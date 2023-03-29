"""
FAST IS BETTER THAN FREE: REVISITING ADVERSARIAL TRAINING
http://arxiv.org/abs/2001.03994
"""
import os
import sys

import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.utils import *
from train.train_base import TrainerBase
from torch.autograd import Variable
from utils.AverageMeter import AverageMeter


class TrainerFast(TrainerBase):
    def __init__(self, cfg, writer, device, loss_function=torch.nn.CrossEntropyLoss(), random_init=True, m=1):
        super(TrainerFast, self).__init__(cfg, writer, device, loss_function)
        self.random_init = random_init
        self.m = m

        self.global_noise_data = torch.zeros([self.cfg.DATA.batch_size, 3, self.cfg.DATA.crop_size,
                                              self.cfg.DATA.crop_size], device=self.device)
        if self.random_init:
            self.global_noise_data.uniform_(-self.cfg.ADV.epsilon, self.cfg.ADV.epsilon)

    def train_one_epoch(self, model, train_loader, optimizer, epoch):
        nat_result = AverageMeter()
        adv_result = AverageMeter()
        with tqdm(total=len(train_loader)) as _tqdm:
            _tqdm.set_description('epoch:{}/{} Training:'.format(epoch + 1, self.cfg.TRAIN.epochs))
            for idx, (data, label) in enumerate(train_loader):
                n = data.size(0)
                data, label = data.to(self.device), label.to(self.device)
                for i in range(self.m):
                    # Ascend on the global noise
                    noise_batch = Variable(self.global_noise_data[0:data.size(0)], requires_grad=True)
                    adv_data = data + noise_batch
                    adv_data.clamp_(self.cfg.ADV.min_value, self.cfg.ADV.max_value)
                    adv_output = model(adv_data)
                    loss = self.loss_fn(adv_output, label)
                    loss.backward()

                    # Update the noise for the next iteration
                    pert = noise_batch.grad * self.cfg.ADV.epsilon * 1.25
                    self.global_noise_data[0:data.size(0)] += pert.data
                    self.global_noise_data.clamp_(-self.cfg.ADV.epsilon, self.cfg.ADV.epsilon)

                    # Descend on global noise
                    noise_batch = Variable(self.global_noise_data[0:data.size(0)], requires_grad=False)
                    adv_data = data + noise_batch
                    adv_data.clamp_(self.cfg.min_image, self.cfg.max_image)
                    adv_output = model(adv_data)
                    loss = self.loss_fn(adv_output, label)

                    # compute gradient and do SGD step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Validation during training
                    if (idx + 1) % self.cfg.TRAIN.print_freq == 0 or (idx + 1) == len(train_loader):
                        # clean data
                        with torch.no_grad():
                            nat_output = model(data)
                        nat_correct_num = (
                                    torch.max(nat_output, dim=1)[1].cpu().detach().numpy() == label.cpu().numpy()). \
                            astype(int).sum()
                        nat_result.update(nat_correct_num, n)

                        # adv data
                        with torch.no_grad():
                            adv_output = model(adv_data)
                        adv_correct_num = (
                                    torch.max(adv_output, dim=1)[1].cpu().detach().numpy() == label.cpu().numpy()). \
                            astype(int).sum()
                        adv_result.update(adv_correct_num, n)

                        _tqdm.set_postfix(loss='{:.3f}'.format(loss.item()),
                                          nat_acc='{:.3f}'.format(nat_result.acc_cur * 100),
                                          rob_acc='{:.3f}'.format(adv_result.acc_cur * 100))
                        if not idx + 1 == len(train_loader):
                            _tqdm.update(self.cfg.TRAIN.print_freq)
                        else:
                            _tqdm.update(len(train_loader) % self.cfg.TRAIN.print_freq)

                        if self.writer is not None:
                            self.writer.add_scalar('Train/Loss_adv', loss.item(), self._iter)
                            self.writer.add_scalar('Train/Clean_acc', nat_result.acc_cur * 100, self._iter)
                            self.writer.add_scalar(f'Train/{self._get_attack_name()}_accuracy',
                                                   adv_result.acc_cur * 100,
                                                   self._iter)
                            self.writer.add_scalar('Train/Lr', optimizer.param_groups[0]["lr"], self._iter)
                    self.adjust_learning_rate(optimizer, len(train_loader), epoch)
                    self._iter += 1
