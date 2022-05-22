"""
TRADES
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch.nn.functional as F

from utils.utils import *

from train.train_base import Trainer_base


class Trainer_Trades(Trainer_base):
    def __init__(self, args, writer, attack_name, device, loss_function=torch.nn.CrossEntropyLoss()):
        super(Trainer_Trades, self).__init__(args, writer, attack_name, device, loss_function)

    def train(self, model, train_loader, valid_loader=None, adv_train=True):
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
            # train_file
            for idx, (data, label) in enumerate(train_loader):
                data, label = data.to(self.device), label.to(self.device)
                attack_method = self.get_attack(model)

                adv_data = attack_method(data, label)
                adv_output = model(adv_data)
                clean_output = model(data)

                # TRADES Loss
                criterion_kl = torch.nn.KLDivLoss(reduction='sum')
                loss_robust = (1. / self.args.batch_size) * criterion_kl(F.log_softmax(adv_output, dim=1),
                                                                         F.softmax(clean_output, dim=1))
                loss_natural = self.loss_fn(clean_output, label)
                loss = loss_natural + self.args.beta * loss_robust

                opt.zero_grad()
                loss.backward()
                opt.step()

                if _iter % self.args.n_eval_step == 0:
                    # clean data
                    pred = torch.max(clean_output, dim=1)[1]
                    std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                    # adv data
                    pred = torch.max(adv_output, dim=1)[1]
                    adv_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                    print(f'[TRAIN]-[{epoch}]/[{self.args.max_epochs}]-iter:{_iter}: lr:{opt.param_groups[0]["lr"]}\n'
                          f'standard acc: {std_acc:.3f}%, robustness acc: {adv_acc:.3f}%, loss:{loss.item():.3f}\n')

                    if self.writer is not None:
                        self.writer.add_scalar('Train/Loss', loss.item(),
                                               epoch * len(train_loader) + idx)
                        self.writer.add_scalar('Train/Clean_acc', std_acc,
                                               epoch * len(train_loader) + idx)
                        self.writer.add_scalar('Train/Adv_acc', adv_acc,
                                               epoch * len(train_loader) + idx)
                        self.writer.add_scalar('Train/Lr', opt.param_groups[0]["lr"],
                                               epoch * len(train_loader) + idx)
                _iter += 1

            if epoch % self.args.n_checkpoint_step == 0:
                file_name = os.path.join(self.args.model_folder, f'checkpoint_{epoch}.pth')
                save_model(model, file_name)

            if valid_loader is not None:
                valid_acc, valid_adv_acc = self.valid(model, valid_loader)
                valid_acc, valid_adv_acc = valid_acc * 100, valid_adv_acc * 100
                print(f'[EVAL] [{epoch}]/[{self.args.max_epochs}]:\n'
                      f'std_acc:{valid_acc}%  adv_acc:{valid_adv_acc}%\n')

                if self.writer is not None:
                    self.writer.add_scalar('Valid/Clean_acc', valid_acc, epoch)
                    self.writer.add_scalar('Valid/Adv_acc', valid_adv_acc, epoch)

            scheduler.step()

