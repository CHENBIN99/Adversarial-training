"""
正常的对抗训练
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
        self.best_clean_acc = 0.
        self.best_robust_acc = 0.
        self._iter = 1

    def _get_attack(self, model, epsilon, alpha, iters):
        if self.cfg.ADV.method == 'pgd':
            return torchattacks.PGD(model=model, eps=epsilon, alpha=alpha, steps=iters, random_start=True)
        elif self.cfg.ADV.method == 'fgsm':
            return torchattacks.FGSM(model=model, eps=epsilon)
        elif self.cfg.ADV.method == 'rfgsm':
            return torchattacks.RFGSM(model=model, eps=epsilon, alpha=alpha, steps=iters)
        else:
            raise 'no match attack method'

    def _get_attack_name(self, train=True):
        if self.attack_name == 'pgd':
            if train:
                return f'PGD-{self.cfg.ADV.iters}'
            else:
                return f'PGD-{self.cfg.ADV.iters_eval}'
        elif self.attack_name == 'fgsm':
            return 'FGSM'
        elif self.attack_name == 'rfgsm':
            if train:
                return f'RFGSM-{self.cfg.ADV.iters}'
            else:
                return f'RFGSM-{self.cfg.ADV.iters_eval}'

    def adjust_learning_rate(self, opt, cur_iters, len_loader, epoch):
        """
        Adjust the learning rate during training.
        :param opt: optimizer
        :param cur_iters: current iteration
        :param len_loader: the total number of mini-batch
        :param epoch: current epoch
        :return: None
        """
        num_milestone = len(self.cfg.TRAIN.lr_epochs)

        for i in range(1, num_milestone):
            if int(self.cfg.TRAIN.lr_epochs[0]) <= epoch < int(self.cfg.TRAIN.lr_epochs):
                self.cfg.TRAIN.lr *= 0.1

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
            if valid_adv_acc >= self.best_robust_acc:
                self.best_clean_acc = valid_acc
                self.best_robust_acc = valid_adv_acc
                self.save_checkpoint(model, epoch, is_best=True)

            print(f'[EVAL] [{epoch}]/[{self.cfg.TRAIN.epochs}]:\n'
                  f'std_acc:{valid_acc * 100}%  adv_acc:{valid_adv_acc * 100}%\n'
                  f'best_epoch:{epoch}\tbest_rob_acc:{self.best_robust_acc * 100}%\n')

            # write to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar('Valid/Clean_acc', valid_acc, epoch)
                self.writer.add_scalar(f'Valid/{self._get_attack_name(train=False)}_Accuracy', valid_adv_acc, epoch)

            # save checkpoint
            if self.cfg.TRAIN.save_ckp_every_epoch:
                self.save_checkpoint(model, epoch)

    @abstractmethod
    def train_one_epoch(self, model, train_loader, optimizer, epoch):
        ...

    def valid(self, model, valid_loader):
        nat_result = AverageMeter()
        adv_result = AverageMeter()
        attack_method = self._get_attack(model, self.cfg.ADV.eps, self.cfg.ADV.alpha, self.cfg.ADV.iters_eval)

        model.eval()
        with torch.no_grad():
            with tqdm(total=len(valid_loader)) as _tqdm:
                _tqdm.set_description('Validating:')
                for idx, (data, label) in enumerate(valid_loader):
                    data, label = data.to(self.device), label.to(self.device)
                    n = data.size(0)

                    # validation using natural data
                    nat_output = model(data)
                    nat_correct_num = (torch.max(nat_output, dim=1)[1].cpu().numpy() == label.cpu().numpy()).sum()
                    nat_result.update(nat_correct_num, n)

                    # validation using adversarial data
                    with torch.enable_grad():
                        adv_data = attack_method(data, label)
                    adv_output = model(adv_data)
                    adv_correct_num = (torch.max(adv_output, dim=1)[1].cpu().numpy() == label.cpu().numpy()).sum()
                    adv_result.update(adv_correct_num, n)

                    _tqdm.set_postfix(nat_acc='{:.3f}%'.format(nat_result.acc_avg * 100),
                                      rob_acc='{:.3f}%'.format(adv_result.acc_avg * 100))
                    _tqdm.update(1)
        model.train()
        return nat_result.acc_avg, adv_result.acc_avg
