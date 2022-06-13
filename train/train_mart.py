"""
MART
paper: IMPROVING ADVERSARIAL ROBUSTNESS REQUIRES REVISITING MISCLASSIFIED EXAMPLES
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch.nn.functional as F

from utils.utils import *

from train.train_base import Trainer_base


class Trainer_Mart(Trainer_base):
    def __init__(self, args, writer, attack_name, device, loss_function=torch.nn.CrossEntropyLoss()):
        super(Trainer_Mart, self).__init__(args, writer, attack_name, device, loss_function)

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
                attack_method = self.get_attack(model, self.args.epsilon, self.args.alpha, self.args.iters)

                model.eval()
                adv_data = attack_method(data, label)
                model.train()
                adv_output = model(adv_data)
                clean_output = model(data)

                # MART Loss
                criterion_kl = torch.nn.KLDivLoss(reduction='none')

                adv_probs = F.softmax(adv_output, dim=1)
                tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
                new_y = torch.where(tmp1[:, -1] == label, tmp1[:, -2], tmp1[:, -1])
                loss_adv = F.cross_entropy(adv_output, label) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

                nat_probs = F.softmax(clean_output, dim=1)
                true_probs = torch.gather(nat_probs, 1, (label.unsqueeze(1)).long()).squeeze()
                loss_robust = (1.0 / len(data)) * torch.sum(
                    torch.sum(criterion_kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))

                loss = loss_adv + self.args.beta * loss_robust

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
                        self.writer.add_scalar('Train/Nature_Accuracy', std_acc,
                                               epoch * len(train_loader) + idx)
                        self.writer.add_scalar(f'Train/{self.get_attack_name()}_Accuracy', adv_acc,
                                               epoch * len(train_loader) + idx)
                        self.writer.add_scalar('Train/Lr', opt.param_groups[0]["lr"],
                                               epoch * len(train_loader) + idx)
                _iter += 1

            if epoch % self.args.n_checkpoint_step == 0:
                self.save_checkpoint(model, epoch)

            if valid_loader is not None:
                valid_acc, valid_adv_acc = self.valid(model, valid_loader)
                valid_acc, valid_adv_acc = valid_acc * 100, valid_adv_acc * 100
                print(f'[EVAL] [{epoch}]/[{self.args.max_epochs}]:\n'
                      f'std_acc:{valid_acc}%  adv_acc:{valid_adv_acc}%\n')

                if self.writer is not None:
                    self.writer.add_scalar('Valid/Clean_acc', valid_acc, epoch)
                    self.writer.add_scalar(f'Valid/{self.get_attack_name(train=False)}_Accuracy', valid_adv_acc, epoch)

            scheduler.step()

