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
from train.train_base import Trainer_base


class Trainer_Free(Trainer_base):
    def __init__(self, args, writer, attack_name, device, loss_function=torch.nn.CrossEntropyLoss(), m=10):
        super().__init__(args=args, writer=writer, attack_name=attack_name, device=device, loss_function=loss_function)
        self.m = m

    def train(self, model, train_loader, valid_loader=None, adv_train=True):
        opt = torch.optim.SGD(model.parameters(), self.args.learning_rate,
                              weight_decay=self.args.weight_decay,
                              momentum=self.args.momentum)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt,
                                                         milestones=[int(self.args.max_epochs * self.args.ms_1),
                                                                     int(self.args.max_epochs * self.args.ms_2),
                                                                     int(self.args.max_epochs * self.args.ms_3)],
                                                         gamma=0.1)
        _iter = 1

        delta = torch.zeros(self.args.batch_size, 3, self.args.image_size, self.args.image_size, device=self.device)

        setattr(self.args, 'max_epochs', self.args // self.m)
        for epoch in range(0, self.args.max_epochs):
            # train_file
            with tqdm(total=len(train_loader)) as _tqdm:
                _tqdm.set_description('epoch:{}/{} Training:'.format(epoch+1, self.args.max_epochs))
                for idx, (data, label) in enumerate(train_loader):
                    data, label = data.to(self.device), label.to(self.device)
                    for i in range(self.m):
                        opt.zero_grad()
                        adv_data = (data + delta).detach()
                        adv_data.requires_grad_()
                        adv_output = model(adv_data)

                        # Loss
                        loss = self.loss_fn(adv_output, label)

                        loss.backward()
                        opt.step()
                        grad = adv_data.grad.data
                        delta = delta.detach() + self.args.epsilon * torch.sign(grad.detach())
                        delta = torch.clamp(delta, -self.args.epsilon, self.args.epsilon)

                    if _iter % self.args.n_eval_step == 0:
                        # clean data
                        with torch.no_grad():
                            std_output = model(data)
                        pred = torch.max(std_output, dim=1)[1]
                        std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                        # adv data
                        pred = torch.max(adv_output, dim=1)[1]
                        adv_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                        # print(f'[TRAIN]-[{epoch}]/[{self.args.max_epochs}]-iter:{_iter}/{len(train_loader)}\t'
                        #       f'lr:{opt.param_groups[0]["lr"]}\n'
                        #       f'standard acc: {std_acc:.3f}%, robustness acc: {adv_acc:.3f}%, loss:{loss.item():.3f}\n')

                        _tqdm.set_postfix(loss='{:.3f}'.format(loss.item()), nat_acc='{:.3f}'.format(std_acc),
                                          rob_acc='{:.3f}'.format(adv_acc))
                        _tqdm.update(self.args.n_eval_step)

                        if self.writer is not None:
                            self.writer.add_scalar('Train/Loss', loss.item(),
                                                   epoch * len(train_loader) + idx + 1)
                            self.writer.add_scalar('Train/Clean_acc', std_acc,
                                                   epoch * len(train_loader) + idx + 1)
                            self.writer.add_scalar(f'Train/{self.get_attack_name()}_Accuracy', adv_acc,
                                                   epoch * len(train_loader) + idx + 1)
                            self.writer.add_scalar('Train/Lr', opt.param_groups[0]["lr"],
                                                   epoch * len(train_loader) + idx + 1)
                    _iter += 1

            if valid_loader is not None:
                valid_acc, valid_adv_acc = self.valid(model, valid_loader)
                valid_acc, valid_adv_acc = valid_acc * 100, valid_adv_acc * 100
                if valid_adv_acc >= self.best_robust_acc:
                    self.best_clean_acc = valid_acc
                    self.best_robust_acc = valid_adv_acc
                    self.save_checkpoint(model, epoch, is_best=True)

                print(f'[EVAL] [{epoch}]/[{self.args.max_epochs}]:\n'
                      f'std_acc:{valid_acc}%  adv_acc:{valid_adv_acc}%\n'
                      f'best_epoch:{epoch}\tbest_rob_acc:{self.best_robust_acc}\n')

                if self.writer is not None:
                    self.writer.add_scalar('Valid/Clean_acc', valid_acc, epoch)
                    self.writer.add_scalar(f'Valid/{self.get_attack_name(train=False)}_Accuracy', valid_adv_acc, epoch)

            if self.args.n_checkpoint_step != -1 and epoch % self.args.n_checkpoint_step == 0:
                self.save_checkpoint(model, epoch)

            scheduler.step()

