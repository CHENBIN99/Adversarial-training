"""
CCG
paper: Consistency Regularization for Adversarial Robustness
"""
import os
import sys

import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch.nn.functional as F
from utils.utils import *
from train.train_base import TrainerBase
from utils.AverageMeter import AverageMeter
from torch.cuda.amp import autocast as autocast


class TrainerCCG(TrainerBase):
    def __init__(self, cfg, writer, device, loss_function=torch.nn.CrossEntropyLoss()):
        super(TrainerCCG, self).__init__(cfg, writer, device, loss_function)

    def train_one_epoch(self, model, train_loaders, optimizer, epoch):
        nat_result = AverageMeter()
        adv_result = AverageMeter()

        # switch to second dataloader
        if epoch > self.cfg.TRAIN.epochs * self.cfg.TRAIN.milestone[0]:
            train_loader = train_loaders[1]
        else:
            train_loader = train_loaders[0]

        with tqdm(total=len(train_loader)) as _tqdm:
            _tqdm.set_description('epoch:{}/{} Training:'.format(epoch + 1, self.cfg.TRAIN.epochs))
            for idx, (datas, labels) in enumerate(train_loader):
                # n = datas.size(0) // 2
                data_aug1, data_aug2 = datas[0].to(self.device), datas[1].to(self.device)
                labels = labels.to(self.device)
                data_pair = torch.cat([data_aug1, data_aug2], dim=0)
                n = data_pair.size(0)

                attack_method = self._get_attack(model, self.cfg.ADV.TRAIN.method, self.cfg.ADV.TRAIN.eps,
                                                 self.cfg.ADV.TRAIN.alpha, self.cfg.ADV.TRAIN.iters)
                adv_data = attack_method(data_pair, labels.repeat(2))

                # Forward
                if self.amp:
                    with autocast():
                        adv_output = model(adv_data)
                        loss_ce = self.loss_fn(adv_output, labels.repeat(2))
                        outputs_adv1, outputs_adv2 = adv_output.chunk(2)
                        loss_con = self.cfg.TRAIN.lamda * self._jensen_shannon_div(outputs_adv1, outputs_adv2,
                                                                                   self.cfg.TRAIN.T)
                        loss = loss_ce + loss_con
                else:
                    adv_output = model(adv_data)
                    loss_ce = self.loss_fn(adv_output, labels.repeat(2))
                    outputs_adv1, outputs_adv2 = adv_output.chunk(2)
                    loss_con = self.cfg.TRAIN.lamda * self._jensen_shannon_div(outputs_adv1, outputs_adv2,
                                                                               self.cfg.TRAIN.T)
                    loss = loss_ce + loss_con

                # Backward
                if self.amp:
                    optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Validation during training
                if (idx + 1) % self.cfg.TRAIN.print_freq == 0 or (idx + 1) == len(train_loader):
                    # clean data
                    with torch.no_grad():
                        nat_output = model(data_pair)
                    nat_correct_num = (torch.max(nat_output, dim=1)[1].cpu().detach().numpy() == labels.repeat(2).cpu().numpy()). \
                        astype(int).sum()
                    nat_result.update(nat_correct_num, n)

                    # adv data
                    with torch.no_grad():
                        adv_output = model(adv_data)
                    adv_correct_num = (torch.max(adv_output, dim=1)[1].cpu().detach().numpy() == labels.repeat(2).cpu().numpy()). \
                        astype(int).sum()
                    adv_result.update(adv_correct_num, n)

                    _tqdm.set_postfix(loss='{:.3f}'.format(loss.item()),
                                      nat_acc='{:.3f}%'.format(nat_result.acc_cur * 100),
                                      rob_acc='{:.3f}%'.format(adv_result.acc_cur * 100))
                    if not idx + 1 == len(train_loader):
                        _tqdm.update(self.cfg.TRAIN.print_freq)
                    else:
                        _tqdm.update(len(train_loader) % self.cfg.TRAIN.print_freq)

                    if self.writer is not None:
                        self.writer.add_scalar('Train/Loss_adv', loss_ce.item(), self._iter)
                        self.writer.add_scalar('Train/Loss_con', loss_con.item(), self._iter)
                        self.writer.add_scalar('Train/Nat._Acc', nat_result.acc_cur * 100, self._iter)
                        self.writer.add_scalar(f'Train/{self._get_attack_name()}_Acc', adv_result.acc_cur * 100,
                                               self._iter)
                        self.writer.add_scalar('Train/Lr', optimizer.param_groups[0]["lr"], self._iter)
                self._iter += 1
                self.scheduler.step()

    def _jensen_shannon_div(self, logit1, logit2, T=1.):
        prob1 = F.softmax(logit1 / T, dim=1)
        prob2 = F.softmax(logit2 / T, dim=1)
        mean_prob = 0.5 * (prob1 + prob2)

        logsoftmax = torch.log(mean_prob.clamp(min=1e-8))
        jsd = F.kl_div(logsoftmax, prob1, reduction='batchmean')
        jsd += F.kl_div(logsoftmax, prob2, reduction='batchmean')
        return jsd * 0.5

