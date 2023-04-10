"""
CCG
paper: Consistency Regularization for Adversarial Robustness
"""
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.utils import *
from train.train_base import TrainerBase
from adv_lib.trades_lib import *
from utils.AverageMeter import AverageMeter


class TrainerCCGTRADES(TrainerBase):
    def __init__(self, args, writer, attack_name, device, loss_function=torch.nn.CrossEntropyLoss()):
        super(TrainerCCGTRADES, self).__init__(args, writer, attack_name, device, loss_function)

    def train_one_epoch(self, model, train_loaders, optimizer, epoch):
        nat_result = AverageMeter()
        adv_result = AverageMeter()

        # switch to second dataloader
        if epoch > self.cfg.epochs * self.cfg.ms_1:
            train_loader = train_loaders[1]
        else:
            train_loader = train_loaders[0]

        with tqdm(total=len(train_loader)) as _tqdm:
            _tqdm.set_description('epoch:{}/{} Training:'.format(epoch + 1, self.cfg.TRAIN.epochs))
            for idx, (datas, labels) in enumerate(train_loader):
                n = datas.size(0)
                data_aug1, data_aug2 = datas[0].to(self.device), datas[1].to(self.device)
                labels = labels.to(self.device)
                data_pair = torch.cat([data_aug1, data_aug2], dim=0)

                attack_method = self._get_attack(model, self.cfg.ADV.TRAIN.method, self.cfg.ADV.eps,
                                                 self.cfg.ADV.alpha, self.cfg.ADV.iters_eval)

                adv_data = attack_method(data_pair, labels.repeat(2))
                adv_output = model(adv_data)

                # Loss
                loss_ce = self.loss_fn(adv_output, labels.repeat(2))
                outputs_adv1, outputs_adv2 = adv_output.chunk(2)
                loss_con = self.cfg.TRAIN.lamda * self._jensen_shannon_div(outputs_adv1, outputs_adv2, self.cfg.TRAIN.T)

                loss_nat, loss_trades, adv_data = trades_loss(model=model, x_natural=data_pair, y=labels.repeat(2),
                                                              optimizer=optimizer, step_size=self.cfg.ADV.alpha,
                                                              epsilon=self.cfg.ADV.epsilon,
                                                              perturb_steps=self.cfg.ADV.iters,
                                                              beta=self.cfg.TRAIN.trades_beta, distance='l_inf')

                loss = loss_ce + loss_con + loss_trades

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Validation during training
                if (idx + 1) % self.cfg.TRAIN.print_freq == 0 or (idx + 1) == len(train_loader):
                    # clean data
                    with torch.no_grad():
                        nat_output = model(datas)
                    nat_correct_num = (torch.max(nat_output, dim=1)[1].cpu().detach().numpy() == labels.cpu().numpy()). \
                        astype(int).sum()
                    nat_result.update(nat_correct_num, n)

                    # adv data
                    with torch.no_grad():
                        adv_output = model(adv_data)
                    adv_correct_num = (torch.max(adv_output, dim=1)[1].cpu().detach().numpy() == labels.cpu().numpy()). \
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
                        self.writer.add_scalar('Train/Loss_trades', loss_trades.item(), self._iter)
                        self.writer.add_scalar('Train/Nat._Acc', nat_result.acc_cur * 100, self._iter)
                        self.writer.add_scalar(f'Train/{self._get_attack_name()}_Acc', adv_result.acc_cur * 100,
                                               self._iter)
                        self.writer.add_scalar('Train/Lr', optimizer.param_groups[0]["lr"], self._iter)
                self.adjust_learning_rate(optimizer, len(train_loader), epoch)
                self._iter += 1

    def _jensen_shannon_div(self, logit1, logit2, T=1.):
        prob1 = F.softmax(logit1 / T, dim=1)
        prob2 = F.softmax(logit2 / T, dim=1)
        mean_prob = 0.5 * (prob1 + prob2)

        logsoftmax = torch.log(mean_prob.clamp(min=1e-8))
        jsd = F.kl_div(logsoftmax, prob1, reduction='batchmean')
        jsd += F.kl_div(logsoftmax, prob2, reduction='batchmean')
        return jsd * 0.5

