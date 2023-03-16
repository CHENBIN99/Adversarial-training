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
from train.train_base import Trainer_base
from torch.autograd import Variable


class Trainer_Fast(Trainer_base):
    def __init__(self, args, writer, attack_name, device, loss_function=torch.nn.CrossEntropyLoss(), random_init=True,
                 m=1):
        super().__init__(args=args, writer=writer, attack_name=attack_name, device=device, loss_function=loss_function)
        self.random_init = random_init
        self.m = m

    def train(self, model, train_loader, valid_loader=None, adv_train=True):
        opt = torch.optim.SGD(model.parameters(), self.args.learning_rate,
                              weight_decay=self.args.weight_decay,
                              momentum=self.args.momentum)
        _iter = 1

        # delta = torch.zeros(self.args.batch_size, 3, self.args.image_size, self.args.image_size, device=self.device)
        global_noise_data = torch.zeros([self.args.batch_size, 3, self.args.image_size, self.args.image_size],
                                        device=self.device)
        if self.random_init:
            global_noise_data.uniform_(-self.args.epsilon, self.args.epsilon)

        for epoch in range(0, self.args.max_epochs):
            # train_file
            with tqdm(total=len(train_loader)) as _tqdm:
                _tqdm.set_description('epoch:{}/{} Training:'.format(epoch+1, self.args.max_epochs))
                for idx, (data, label) in enumerate(train_loader):
                    data, label = data.to(self.device), label.to(self.device)
                    for i in range(self.m):
                        # Ascend on the global noise
                        noise_batch = Variable(global_noise_data[0:data.size(0)], requires_grad=True)
                        adv_data = data + noise_batch
                        adv_data.clamp_(self.args.min_image, self.args.max_image)
                        adv_output = model(adv_data)
                        loss = self.loss_fn(adv_output, label)
                        loss.backward()

                        # Update the noise for the next iteration
                        pert = noise_batch.grad * self.args.epsilon * 1.25
                        global_noise_data[0:data.size(0)] += pert.data
                        global_noise_data.clamp_(-self.args.epsilon, self.args.epsilon)

                        # Descend on global noise
                        noise_batch = Variable(global_noise_data[0:data.size(0)], requires_grad=False)
                        adv_data = data + noise_batch
                        adv_data.clamp_(self.args.min_image, self.args.max_image)
                        adv_output = model(adv_data)
                        loss = self.loss_fn(adv_output, label)

                        # compute gradient and do SGD step
                        opt.zero_grad()
                        loss.backward()
                        opt.step()

                    self.adjust_learning_rate(opt, _iter, len(train_loader), epoch)

                    if _iter % self.args.n_eval_step == 0:
                        # clean data
                        with torch.no_grad():
                            std_output = model(data)
                        pred = torch.max(std_output, dim=1)[1]
                        std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                        # adv data
                        pred = torch.max(adv_output, dim=1)[1]
                        adv_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                        _tqdm.set_postfix(loss='{:.3f}'.format(loss.item()), nat_acc='{:.3f}'.format(std_acc),
                                          rob_acc='{:.3f}'.format(adv_acc))
                        _tqdm.update(self.args.n_eval_step)

                        if self.writer is not None:
                            self.writer.add_scalar('Train/Loss', loss.item(),
                                                   epoch * len(train_loader) + idx + 1)
                            self.writer.add_scalar('Train/Clean_acc', std_acc,
                                                   epoch * len(train_loader) + idx + 1)
                            self.writer.add_scalar(f'Train/FGSM_Accuracy', adv_acc,
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

