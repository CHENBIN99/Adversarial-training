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
from torch.cuda.amp import autocast as autocast


class TrainerEns(TrainerBase):
    def __init__(self, cfg, writer, device, loss_function=torch.nn.CrossEntropyLoss(), static_model=None,
                 holdout_model=None):
        super(TrainerEns, self).__init__(cfg, writer, device, loss_function)
        self.static_model = static_model
        self.holdout_model = holdout_model
        self.num_static_model = len(self.static_model)

    def train_one_epoch(self, model, train_loader, optimizer, epoch):
        nat_result = AverageMeter()
        adv_w_result = AverageMeter()
        adv_b_result = AverageMeter()
        with tqdm(total=len(train_loader)) as _tqdm:
            _tqdm.set_description('epoch:{}/{} Training:'.format(epoch + 1, self.cfg.TRAIN.epochs))
            for idx, (data, label) in enumerate(train_loader):
                n = data.size(0)
                data, label = data.to(self.device), label.to(self.device)
                # Random choice
                selected = np.random.randint(self.num_static_model + 1)
                if selected == self.num_static_model:
                    selected_model = model
                else:
                    selected_model = self.static_model[selected]
                selected_model.eval()

                attack_method = self._get_attack(selected_model, self.cfg.ADV.TRAIN.method, self.cfg.ADV.TRAIN.eps,
                                                 self.cfg.ADV.TRAIN.alpha, self.cfg.ADV.TRAIN.iters)

                model.train()
                # Forward
                if self.amp:
                    with autocast():
                        adv_data = attack_method(data, label)
                        nat_output = model(data)
                        adv_output = model(adv_data)
                        nat_loss = self.loss_fn(nat_output, label)
                        adv_loss = self.loss_fn(adv_output, label)
                        loss = 0.5 * (nat_loss + adv_loss)
                else:
                    adv_data = attack_method(data, label)
                    nat_output = model(data)
                    adv_output = model(adv_data)
                    nat_loss = self.loss_fn(nat_output, label)
                    adv_loss = self.loss_fn(adv_output, label)
                    loss = 0.5 * (nat_loss + adv_loss)

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
                        nat_output = model(data)
                    nat_correct_num = (torch.max(nat_output, dim=1)[1].cpu().detach().numpy() == label.cpu().numpy()). \
                        astype(int).sum()
                    nat_result.update(nat_correct_num, n)

                    # white-box adv attack
                    attack_method = self._get_attack(model, self.cfg.ADV.TRAIN.method, self.cfg.ADV.TRAIN.eps,
                                                     self.cfg.ADV.TRAIN.alpha, self.cfg.ADV.TRAIN.iters)
                    adv_data_w = attack_method(data, label)
                    with torch.no_grad():
                        adv_w_output = model(adv_data_w)
                    adv_w_correct_num = (torch.max(adv_w_output, dim=1)[1].cpu().detach().numpy() == label.cpu().numpy()).\
                        astype(int).sum()
                    adv_w_result.update(adv_w_correct_num, n)

                    # black-box adv attack
                    attack_method = self._get_attack(self.holdout_model, self.cfg.ADV.TRAIN.method, self.cfg.ADV.TRAIN.eps,
                                                     self.cfg.ADV.TRAIN.alpha, self.cfg.ADV.TRAIN.iters)
                    adv_data_b = attack_method(data, label)
                    with torch.no_grad():
                        adv_b_output = model(adv_data_b)
                    adv_b_correct_num = (torch.max(adv_b_output, dim=1)[1].cpu().detach().numpy() == label.cpu().numpy()). \
                        astype(int).sum()
                    adv_b_result.update(adv_b_correct_num, n)

                    _tqdm.set_postfix(loss='{:.3f}'.format(loss.item()),
                                      nat_acc='{:.3f}%'.format(nat_result.acc_cur * 100),
                                      wb_rob_acc='{:.3f}%'.format(adv_w_result.acc_cur * 100),
                                      bb_rob_acc='{:.3f}%'.format(adv_b_result.acc_cur * 100))
                    if not idx + 1 == len(train_loader):
                        _tqdm.update(self.cfg.TRAIN.print_freq)
                    else:
                        _tqdm.update(len(train_loader) % self.cfg.TRAIN.print_freq)

                    if self.writer is not None:
                        self.writer.add_scalar('Train/Loss_adv', adv_loss.item(), self._iter)
                        self.writer.add_scalar('Train/Loss_nat', nat_loss.item(), self._iter)
                        self.writer.add_scalar('Train/Nat._Acc', nat_result.acc_cur * 100, self._iter)
                        self.writer.add_scalar(f'Train/{self._get_attack_name()}_Whitebox_Acc',
                                               adv_w_result.acc_cur * 100, self._iter)
                        self.writer.add_scalar(f'Train/{self._get_attack_name()}_Blackbox_Acc',
                                               adv_b_result.acc_cur * 100, self._iter)
                        self.writer.add_scalar('Train/Lr', optimizer.param_groups[0]["lr"], self._iter)
                self._iter += 1
                self.scheduler.step()

    def train(self, model, train_loader, valid_loader):
        opt = torch.optim.SGD(model.parameters(), self.cfg.TRAIN.lr, weight_decay=self.cfg.TRAIN.weight_decay,
                              momentum=self.cfg.TRAIN.momentum)
        self.scheduler = self.get_lr_scheduler(opt, self.cfg.TRAIN.lr_scheduler_name, len(train_loader))

        for epoch in range(0, self.cfg.TRAIN.epochs):
            # training
            self.train_one_epoch(model, train_loader, opt, epoch)

            # validation
            valid_acc, valid_adv_w_acc, valid_adv_b_acc = self.valid(model, valid_loader)

            if valid_adv_b_acc >= self.best_robust_acc:
                self.best_clean_acc = valid_acc
                self.best_robust_acc = valid_adv_b_acc
                self.best_epoch = epoch
                self.save_checkpoint(model, epoch, is_best=True)

            print(f'[EVAL] [{epoch}]/[{self.cfg.TRAIN.epochs}]:\n'
                  f'nat_acc:{valid_acc * 100}%  adv_w_acc:{valid_adv_w_acc * 100}%  adv_w_acc:{valid_adv_b_acc * 100}%\n'
                  f'best_epoch:{self.best_epoch}\tbest_rob_acc:{self.best_robust_acc * 100}%\n')

            # write to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar('Valid/Nat._Acc', valid_acc, epoch)
                self.writer.add_scalar(f'Valid/{self._get_attack_name(train=False)}_Whitebox_Acc', valid_adv_w_acc,
                                       epoch)
                self.writer.add_scalar(f'Valid/{self._get_attack_name(train=False)}_Blackbox_Acc', valid_adv_b_acc,
                                       epoch)

            # save checkpoint
            if self.cfg.TRAIN.save_ckp_freq != -1 and epoch % self.cfg.TRAIN.save_ckp_freq == 0:
                self.save_checkpoint(model, epoch)

    def valid(self, model, valid_loader):
        nat_result = AverageMeter()
        adv_w_result = AverageMeter()
        adv_b_result = AverageMeter()
        attack_method_w = self._get_attack(model, self.cfg.ADV.TRAIN.method, self.cfg.ADV.TRAIN.eps,
                                           self.cfg.ADV.TRAIN.alpha, self.cfg.ADV.TRAIN.iters)
        attack_method_b = self._get_attack(self.holdout_model, self.cfg.ADV.TRAIN.method, self.cfg.ADV.TRAIN.eps,
                                           self.cfg.ADV.TRAIN.alpha, self.cfg.ADV.TRAIN.iters)

        model.eval()
        with torch.no_grad():
            with tqdm(total=len(valid_loader)) as _tqdm:
                _tqdm.set_description('Validating:')
                for idx, (data, label) in enumerate(valid_loader):
                    data, label = data.to(self.device), label.to(self.device)
                    n = data.size(0)

                    # validation using natural data
                    nat_output = model(data)
                    nat_correct_num = (torch.max(nat_output, dim=1)[1].cpu().detach().numpy() == label.cpu().numpy()). \
                        astype(int).sum()
                    nat_result.update(nat_correct_num, n)

                    # adv attack
                    with torch.enable_grad():
                        adv_w_data = attack_method_w(data, label)
                        adv_b_data = attack_method_b(data, label)

                    adv_w_output = model(adv_w_data)
                    adv_b_output = model(adv_b_data)

                    adv_w_correct_num = (
                                torch.max(adv_w_output, dim=1)[1].cpu().detach().numpy() == label.cpu().numpy()). \
                        astype(int).sum()
                    adv_w_result.update(adv_w_correct_num, n)

                    adv_b_correct_num = (
                            torch.max(adv_b_output, dim=1)[1].cpu().detach().numpy() == label.cpu().numpy()). \
                        astype(int).sum()
                    adv_b_result.update(adv_b_correct_num, n)

                    if self.cfg.method != 'nature':
                        _tqdm.set_postfix(nat_acc='{:.3f}%'.format(nat_result.acc_avg * 100),
                                          rob_w_acc='{:.3f}%'.format(adv_w_result.acc_avg * 100),
                                          rob_b_acc='{:.3f}%'.format(adv_b_result.acc_avg * 100))

                    _tqdm.update(1)
        model.train()
        return nat_result.acc_avg, adv_w_result.acc_avg, adv_b_result.acc_avg
