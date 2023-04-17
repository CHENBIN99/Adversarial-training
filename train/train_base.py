"""
Base implement of adversarial attack training
"""

import torchattacks
import torch
from tqdm import tqdm

from utils.utils import *
from abc import ABC, abstractmethod
from utils.AverageMeter import AverageMeter


class TrainerBase(object):
    def __init__(self, cfg, writer, device, loss_function=torch.nn.CrossEntropyLoss()):
        self.cfg = cfg
        self.writer = writer
        self.device = device
        self.loss_fn = loss_function
        # log
        self.best_epoch = 0
        self.best_clean_acc = 0.
        self.best_robust_acc = 0.
        self._iter = 1

    def _get_attack(self, model, attack_name, epsilon, alpha, iters):
        if attack_name == 'pgd':
            return torchattacks.PGD(model=model, eps=epsilon, alpha=alpha, steps=iters, random_start=True)
        elif attack_name == 'fgsm':
            return torchattacks.FGSM(model=model, eps=epsilon)
        elif attack_name == 'rfgsm':
            return torchattacks.RFGSM(model=model, eps=epsilon, alpha=alpha, steps=iters)
        else:
            raise 'no match attack method'

    def _get_attack_name(self, train=True):
        if train:
            if self.cfg.ADV.TRAIN.method == 'pgd':
                return f'PGD-{self.cfg.ADV.TRAIN.iters}'
            elif self.cfg.ADV.TRAIN.method == 'fgsm':
                return 'FGSM'
            elif self.cfg.ADV.TRAIN.method == 'rfgsm':
                return f'RFGSM-{self.cfg.ADV.TRAIN.iters}'
        else:
            if self.cfg.ADV.EVAL.method == 'pgd':
                return f'PGD-{self.cfg.ADV.EVAL.iters}'
            elif self.cfg.ADV.EVAL.method == 'fgsm':
                return 'FGSM'
            elif self.cfg.ADV.EVAL.method == 'rfgsm':
                return f'RFGSM-{self.cfg.ADV.EVAL.iters}'

    def adjust_learning_rate(self, opt, len_loader, epoch):
        """
        Adjust the learning rate during training.
        :param opt: optimizer
        :param len_loader: the total number of mini-batch
        :param epoch: current epoch
        :return: None
        """
        num_milestone = len(self.cfg.TRAIN.lr_epochs)

        for i in range(0, num_milestone - 1):
            if epoch == self.cfg.TRAIN.lr_epochs[i]:
                self.cfg.TRAIN.lr = self.cfg.TRAIN.lr_values[i]
                break

        for param_group in opt.param_groups:
            param_group["lr"] = self.cfg.TRAIN.lr

    def save_checkpoint(self, model, epoch, is_best=False):
        if not is_best:
            file_name = os.path.join(self.cfg.ckp_path, f'checkpoint_{epoch}.pth')
        else:
            file_name = os.path.join(self.cfg.ckp_path, f'checkpoint_best.pth')
        torch.save(model.state_dict(), file_name)

    def train(self, model, train_loader, valid_loader):
        opt = torch.optim.SGD(model.parameters(), self.cfg.TRAIN.lr, weight_decay=self.cfg.TRAIN.weight_decay,
                              momentum=self.cfg.TRAIN.momentum)

        for epoch in range(0, self.cfg.TRAIN.epochs):
            # training
            self.train_one_epoch(model, train_loader, opt, epoch)

            # validation
            valid_acc, valid_adv_acc = self.valid(model, valid_loader)
            if self.cfg.method == 'nature':
                if valid_acc >= self.best_clean_acc:
                    self.best_clean_acc = valid_acc
                    self.best_robust_acc = valid_adv_acc
                    self.best_epoch = epoch
                    self.save_checkpoint(model, epoch, is_best=True)
            else:
                if valid_adv_acc >= self.best_robust_acc:
                    self.best_clean_acc = valid_acc
                    self.best_robust_acc = valid_adv_acc
                    self.best_epoch = epoch
                    self.save_checkpoint(model, epoch, is_best=True)

            if self.cfg.method == 'nature':
                print(f'[EVAL] [{epoch}]/[{self.cfg.TRAIN.epochs}]:\n'
                      f'nat_acc:{valid_acc * 100}%  adv_acc:{valid_adv_acc * 100}%\n'
                      f'best_epoch:{self.best_epoch}\tbest_nat_acc:{self.best_clean_acc * 100}%\n')
            else:
                print(f'[EVAL] [{epoch}]/[{self.cfg.TRAIN.epochs}]:\n'
                      f'nat_acc:{valid_acc * 100}%  adv_acc:{valid_adv_acc * 100}%\n'
                      f'best_epoch:{self.best_epoch}\tbest_rob_acc:{self.best_robust_acc * 100}%\n')

            # write to TensorBoard
            if self.writer is not None:
                if self.cfg.method != 'nature':
                    self.writer.add_scalar('Valid/Nat._Acc', valid_acc, epoch)
                    self.writer.add_scalar(f'Valid/{self._get_attack_name(train=False)}_Acc', valid_adv_acc, epoch)
                else:
                    self.writer.add_scalar('Valid/Nat._Acc', valid_acc, epoch)

            # save checkpoint
            if self.cfg.TRAIN.save_ckp_every_epoch:
                self.save_checkpoint(model, epoch)

    @abstractmethod
    def train_one_epoch(self, model, train_loader, optimizer, epoch):
        ...

    def valid(self, model, valid_loader):
        nat_result = AverageMeter()
        adv_result = AverageMeter()
        attack_method = self._get_attack(model, self.cfg.ADV.EVAL.method, self.cfg.ADV.EVAL.eps,
                                         self.cfg.ADV.EVAL.alpha, self.cfg.ADV.EVAL.iters)

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

                    if self.cfg.method != 'nature':
                        # validation using adversarial data
                        with torch.enable_grad():
                            adv_data = attack_method(data, label)
                        adv_output = model(adv_data)
                        adv_correct_num = (torch.max(adv_output, dim=1)[1].cpu().detach().numpy() == label.cpu().numpy()). \
                            astype(int).sum()
                        adv_result.update(adv_correct_num, n)

                    if self.cfg.method != 'nature':
                        _tqdm.set_postfix(nat_acc='{:.3f}%'.format(nat_result.acc_avg * 100),
                                          rob_acc='{:.3f}%'.format(adv_result.acc_avg * 100))
                    else:
                        _tqdm.set_postfix(nat_acc='{:.3f}%'.format(nat_result.acc_avg * 100))
                    _tqdm.update(1)
        model.train()
        return nat_result.acc_avg, adv_result.acc_avg
