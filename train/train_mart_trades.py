"""
MART
paper: IMPROVING ADVERSARIAL ROBUSTNESS REQUIRES REVISITING MISCLASSIFIED EXAMPLES
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.utils import *

from train.train_base import Trainer_base

from adv_lib.trades_lib import *
from adv_lib.mart_lib import *


class Trainer_Mart_Trades(Trainer_base):
    def __init__(self, args, writer, attack_name, device, loss_function=torch.nn.CrossEntropyLoss()):
        super(Trainer_Mart_Trades, self).__init__(args, writer, attack_name, device, loss_function)

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

                # MART Loss
                loss_adv, loss_mart, adv_data = mart_loss(model=model, x_natural=data, y=label, optimizer=opt,
                                                          step_size=self.args.alpha, epsilon=self.args.epsilon,
                                                          perturb_steps=self.args.iters, beta=self.args.mart_beta,
                                                          distance='l_inf')
                # TRADES Loss
                loss_nat, loss_trades, adv_data = trades_loss(model=model, x_natural=data, y=label, optimizer=opt,
                                                              step_size=self.args.alpha, epsilon=self.args.epsilon,
                                                              perturb_steps=self.args.iters, beta=self.args.trades_beta,
                                                              distance='l_inf')

                loss = loss_adv + loss_mart + loss_trades

                opt.zero_grad()
                loss.backward()
                opt.step()

                if _iter % self.args.n_eval_step == 0:
                    # clean data
                    with torch.no_grad():
                        clean_output = model(data)
                    pred = torch.max(clean_output, dim=1)[1]
                    std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                    # adv data
                    with torch.no_grad():
                        adv_output = model(adv_data)
                    pred = torch.max(adv_output, dim=1)[1]
                    adv_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                    print(f'[TRAIN]-[{epoch}]/[{self.args.max_epochs}]-iter:{_iter}: lr:{opt.param_groups[0]["lr"]}\n'
                          f'standard acc: {std_acc:.3f}%\trobustness acc: {adv_acc:.3f}%\n'
                          f'loss_adv:{loss_adv.item():.3f}\tloss_mart:{loss_mart.item():.3f}\t'
                          f'loss_trades:{loss_trades.item():.3f}\n')

                    if self.writer is not None:
                        self.writer.add_scalar('Train/Loss_adv', loss_adv.item(),
                                               epoch * len(train_loader) + idx)
                        self.writer.add_scalar('Train/Loss_mart', loss_mart.item(),
                                               epoch * len(train_loader) + idx)
                        self.writer.add_scalar('Train/Loss_trades', loss_trades.item(),
                                               epoch * len(train_loader) + idx)
                        self.writer.add_scalar('Train/Nature_Accuracy', std_acc,
                                               epoch * len(train_loader) + idx)
                        self.writer.add_scalar(f'Train/{self.get_attack_name()}_Accuracy', adv_acc,
                                               epoch * len(train_loader) + idx)
                        self.writer.add_scalar('Train/Lr', opt.param_groups[0]["lr"],
                                               epoch * len(train_loader) + idx)
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

