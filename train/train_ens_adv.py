"""
Standard Adversarial Training
"""
import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.utils import *
from train.train_base import TrainerBase
from utils.AverageMeter import AverageMeter


class TrainerEns(TrainerBase):
    def __init__(self, args, writer, device, loss_function=torch.nn.CrossEntropyLoss(), static_model=None):
        super(TrainerEns, self).__init__(args, writer, device, loss_function)
        self.static_model = static_model
        self.num_static_model = len(self.static_model)

    def train_one_epoch(self, model, train_loader, optimizer, epoch):
        nat_result = AverageMeter()
        adv_result = AverageMeter()
        with tqdm(total=len(train_loader)) as _tqdm:
            _tqdm.set_description('epoch:{}/{} Training:'.format(epoch + 1, self.args.max_epochs))
            for idx, (data, label) in enumerate(train_loader):
                n = data.size(0)
                data, label = data.to(self.device), label.to(self.device)
                # random choice
                selected = np.random.randint(self.num_static_model + 1)
                if selected == self.num_static_model:
                    selected_model = model
                else:
                    selected_model = self.static_model[selected]
                attack_method = self._get_attack(selected_model, self.cfg.ADV.TRAIN.method, self.cfg.ADV.eps,
                                                 self.cfg.ADV.alpha, self.cfg.ADV.iters_eval)
                adv_data = attack_method(data, label)

                # training
                nat_output = model(data)
                nat_loss = self.loss_fn(nat_output, label)

                adv_output = model(adv_data)
                adv_loss = self.loss_fn(adv_output, label)

                # combine
                loss = 0.5 * (nat_loss + adv_loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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
                        self.writer.add_scalar('Train/Loss_adv', adv_loss.item(), self._iter)
                        self.writer.add_scalar('Train/Loss_nat', nat_loss.item(), self._iter)
                        self.writer.add_scalar('Train/Nat._Acc', nat_result.acc_cur * 100, self._iter)
                        self.writer.add_scalar(f'Train/{self._get_attack_name()}_Acc', adv_result.acc_cur * 100,
                                               self._iter)
                        self.writer.add_scalar('Train/Lr', optimizer.param_groups[0]["lr"], self._iter)
                self._iter += 1
                self.scheduler.step()
