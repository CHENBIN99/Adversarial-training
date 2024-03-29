"""
Adversarial for free!
https://arxiv.org/abs/1904.12843
"""
import os
import sys

import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.utils import *
from train.train_base import TrainerBase
from utils.AverageMeter import AverageMeter


class TrainerFree(TrainerBase):
    def __init__(self, cfg, writer, device, loss_function=torch.nn.CrossEntropyLoss(), m=10):
        super(TrainerFree, self).__init__(cfg, writer, device, loss_function)
        self.m = m
        self.delta = torch.zeros(self.cfg.DATA.batch_size, 3, self.cfg.DATA.crop_size, self.cfg.DATA.crop_size,
                                 device=self.device)

    def train_one_epoch(self, model, train_loader, optimizer, epoch):
        nat_result = AverageMeter()
        adv_result = AverageMeter()
        with tqdm(total=len(train_loader)) as _tqdm:
            _tqdm.set_description('epoch:{}/{} Training:'.format(epoch + 1, self.cfg.TRAIN.epochs))
            for idx, (data, label) in enumerate(train_loader):
                n = data.size(0)
                data, label = data.to(self.device), label.to(self.device)
                for i in range(self.m):
                    optimizer.zero_grad()
                    adv_data = (data + self.delta).detach()
                    adv_data.requires_grad_()
                    adv_output = model(adv_data)

                    # Loss
                    loss = self.loss_fn(adv_output, label)
                    loss.backward()
                    optimizer.step()
                    grad = adv_data.grad.data
                    self.delta = self.delta.detach() + self.cfg.ADV.TRAIN.eps * torch.sign(grad.detach())
                    self.delta = torch.clamp(self.delta, -self.cfg.ADV.TRAIN.eps, self.cfg.ADV.TRAIN.eps)

                # Validation during training
                if (idx + 1) % self.cfg.TRAIN.print_freq == 0 or (idx + 1) == len(train_loader):
                    # clean data
                    with torch.no_grad():
                        nat_output = model(data)
                    nat_correct_num = (torch.max(nat_output, dim=1)[1].cpu().detach().numpy() == label.cpu().numpy()). \
                        astype(int).sum()
                    nat_result.update(nat_correct_num, n)

                    # adv data
                    with torch.no_grad():
                        adv_output = model(adv_data)
                    adv_correct_num = (torch.max(adv_output, dim=1)[1].cpu().detach().numpy() == label.cpu().numpy()). \
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
                        self.writer.add_scalar('Train/Loss_adv', loss.item(), self._iter)
                        self.writer.add_scalar('Train/Nat._Acc', nat_result.acc_cur * 100, self._iter)
                        self.writer.add_scalar(f'Train/{self._get_attack_name()}_Acc',
                                               adv_result.acc_cur * 100,
                                               self._iter)
                        self.writer.add_scalar('Train/Lr', optimizer.param_groups[0]["lr"], self._iter)
                self._iter += 1
                self.scheduler.step()

