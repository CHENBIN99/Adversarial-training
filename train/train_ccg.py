"""
CCG
paper: Consistency Regularization for Adversarial Robustness
"""
import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch.nn.functional as F

from utils.utils import *

from train.train_base import Trainer_base


class Trainer_CCG(Trainer_base):
    def __init__(self, args, writer, attack_name, device, loss_function=torch.nn.CrossEntropyLoss()):
        super(Trainer_CCG, self).__init__(args, writer, attack_name, device, loss_function)

    def train(self, model, train_loaders, valid_loader=None, adv_train=True):
        opt = torch.optim.SGD(model.parameters(), self.args.learning_rate,
                              weight_decay=self.args.weight_decay,
                              momentum=self.args.momentum)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt,
                                                         milestones=[int(self.args.max_epochs * self.args.ms_1),
                                                                     int(self.args.max_epochs * self.args.ms_2),
                                                                     int(self.args.max_epochs * self.args.ms_3)],
                                                         gamma=0.1)
        _iter = 0
        for epoch in range(0, self.args.max_epochs):
            print('time start...')
            start = time.time()
            # switch to second dataloader
            if epoch > self.args.max_epochs * self.args.ms_1:
                train_loader = train_loaders[1]
            else:
                train_loader = train_loaders[0]

            # train_file
            for idx, (datas, labels) in enumerate(train_loader):
                data_aug1, data_aug2 = datas[0].to(self.device), datas[1].to(self.device)
                labels = labels.to(self.device)
                data_pair = torch.cat([data_aug1, data_aug2], dim=0)

                attack_method = self.get_attack(model, self.args.epsilon, self.args.alpha, self.args.iters_eval)

                adv_data = attack_method(data_pair, labels.repeat(2))
                adv_output = model(adv_data)

                # Loss
                loss_ce = self.loss_fn(adv_output, labels.repeat(2))
                outputs_adv1, outputs_adv2 = adv_output.chunk(2)
                loss_con = self.args.lamda * self._jensen_shannon_div(outputs_adv1, outputs_adv2, self.args.T)
                loss = loss_ce + loss_con

                opt.zero_grad()
                loss.backward()
                opt.step()

                # if _iter % self.args.n_eval_step == 0:
                #     # clean data
                #     with torch.no_grad():
                #         std_output = model(data_pair)
                #     pred = torch.max(std_output, dim=1)[1]
                #     std_acc = evaluate(pred.cpu().numpy(), labels.repeat(2).cpu().numpy()) * 100
                #
                #     # adv data
                #     pred = torch.max(adv_output, dim=1)[1]
                #     adv_acc = evaluate(pred.cpu().numpy(), labels.repeat(2).cpu().numpy()) * 100
                #
                #     print(f'[TRAIN]-[{epoch}]/[{self.args.max_epochs}]-iter:{_iter}: lr:{opt.param_groups[0]["lr"]}\n'
                #           f'standard acc: {std_acc:.3f}%, robustness acc: {adv_acc:.3f}%, loss:{loss.item():.3f}\n')
                #
                #     if self.writer is not None:
                #         self.writer.add_scalar('Train/Loss', loss.item(),
                #                                epoch * len(train_loader) + idx)
                #         self.writer.add_scalar('Train/Clean_acc', std_acc,
                #                                epoch * len(train_loader) + idx)
                #         self.writer.add_scalar(f'Train/{self.get_attack_name()}_Accuracy', adv_acc,
                #                                epoch * len(train_loader) + idx)
                #         self.writer.add_scalar('Train/Lr', opt.param_groups[0]["lr"],
                #                                epoch * len(train_loader) + idx)
                _iter += 1

            print(f'Use: {time.time() - start}')

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

    def _jensen_shannon_div(self, logit1, logit2, T=1.):
        prob1 = F.softmax(logit1 / T, dim=1)
        prob2 = F.softmax(logit2 / T, dim=1)
        mean_prob = 0.5 * (prob1 + prob2)

        logsoftmax = torch.log(mean_prob.clamp(min=1e-8))
        jsd = F.kl_div(logsoftmax, prob1, reduction='batchmean')
        jsd += F.kl_div(logsoftmax, prob2, reduction='batchmean')
        return jsd * 0.5

