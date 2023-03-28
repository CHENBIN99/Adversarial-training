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


class TrainerEns(TrainerBase):
    def __init__(self, args, writer, device, loss_function=torch.nn.CrossEntropyLoss(), static_model=None):
        super(TrainerEns, self).__init__(args, writer, device, loss_function)
        self.static_model = static_model
        self.num_static_model = len(self.static_model)

    def train_one_epoch(self, model, train_loader, optimizer, epoch):
        nat_result = AverageMeter()
        adv_result = AverageMeter()
        with tqdm(total=len(train_loader)) as _tqdm:
            _tqdm.set_description('epoch:{}/{} Training:'.format(epoch + 1, self.args.max_epochs))
            for idx, (data, label) in enumerate(train_loader):
                n = data.size(0)
                data, label = data.to(self.device), label.to(self.device)
                # random choice
                selected = np.random.randint(self.num_static_model + 1)
                if selected == self.num_static_model:
                    selected_model = model
                else:
                    selected_model = self.static_model[selected]
                attack_method = self._get_attack(selected_model, self.cfg.ADV.eps, self.cfg.ADV.alpha, self.cfg.ADV.iters)
                adv_data = attack_method(data, label)

                # training
                nat_output = model(data)
                nat_loss = self.loss_fn(nat_output, label)

                adv_output = model(adv_data)
                adv_loss = self.loss_fn(adv_output, label)

                # combine
                loss = 0.5 * (nat_loss + adv_loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Validation during training
                if self._iter % self.cfg.TRAIN.print_freq == 0:
                    # clean data
                    with torch.no_grad():
                        nat_output = model(data)
                    nat_correct_num = (torch.max(nat_output, dim=1)[1].cpu().detach().numpy() == label.cpu().numpy()). \
                        astype(int).sum()
                    nat_result.update(nat_correct_num, n)

                    # adv data
                    with torch.no_grad():
                        adv_output = model(adv_data)
                    adv_correct_num = (torch.max(adv_output, dim=1)[1].cpu().detach().numpy() == label.cpu().numpy()). \
                        astype(int).sum()
                    adv_result.update(adv_correct_num, n)

                    _tqdm.set_postfix(loss='{:.3f}'.format(loss.item()), nat_acc='{:.3f}'.format(nat_result.acc_cur),
                                      rob_acc='{:.3f}'.format(adv_result.acc_cur))
                    _tqdm.update(self.cfg.TRAIN.print_freq)

                    if self.writer is not None:
                        self.writer.add_scalar('Train/Loss', loss.item(), epoch * len(train_loader) + idx)
                        self.writer.add_scalar('Train/Clean_acc', nat_result.acc_cur, epoch * len(train_loader) + idx)
                        self.writer.add_scalar(f'Train/{self._get_attack_name()}_accuracy', adv_result.acc_cur,
                                               epoch * len(train_loader) + idx)
                        self.writer.add_scalar('Train/Lr', optimizer.param_groups[0]["lr"],
                                               epoch * len(train_loader) + idx)
                self._iter += 1

    def train(self, model, static_model, train_loader, valid_loader=None, adv_train=True):
        opt = torch.optim.SGD(model.parameters(), self.args.learning_rate,
                              weight_decay=self.args.weight_decay,
                              momentum=self.args.momentum)

        num_static_model = len(static_model)

        _iter = 1
        for epoch in range(0, self.args.max_epochs):
            # train_file
            with tqdm(total=len(train_loader)) as _tqdm:
                _tqdm.set_description('epoch:{}/{} Training:'.format(epoch+1, self.args.max_epochs))
                for idx, (data, label) in enumerate(train_loader):
                    data, label = data.to(self.device), label.to(self.device)

                    # random choice
                    selected = np.random.randint(num_static_model + 1)
                    if selected == num_static_model:
                        selected_model = model
                    else:
                        selected_model = static_model[selected]
                    attack_method = self.get_attack(selected_model, self.args.epsilon, self.args.alpha, self.args.iters)
                    adv_data = attack_method(data, label)

                    # training
                    nat_output = model(data)
                    nat_loss = self.loss_fn(nat_output, label)

                    adv_output = model(adv_data)
                    adv_loss = self.loss_fn(adv_output, label)

                    # combine
                    loss = 0.5 * (nat_loss + adv_loss)

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    if _iter % self.args.n_eval_step == 0:
                        # clean data
                        with torch.no_grad():
                            std_output = model(data)
                        pred = torch.max(std_output, dim=1)[1]
                        std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                        # adv data
                        if selected != num_static_model:
                            attack_method = self.get_attack(model, self.args.epsilon, self.args.alpha, self.args.iters)
                            adv_data = attack_method(data, label)
                            adv_output = model(adv_data)
                        pred = torch.max(adv_output, dim=1)[1]
                        adv_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                        _tqdm.set_postfix(loss='{:.3f}'.format(loss.item()), nat_acc='{:.3f}'.format(std_acc),
                                          rob_acc='{:.3f}'.format(adv_acc))
                        _tqdm.update(self.args.n_eval_step)

                        if self.writer is not None:
                            self.writer.add_scalar('Train/Loss', loss.item(), _iter)
                            self.writer.add_scalar('Train/Clean_acc', std_acc, _iter)
                            self.writer.add_scalar(f'Train/{self.get_attack_name()}_Accuracy', adv_acc, _iter)
                            self.writer.add_scalar('Train/Lr', opt.param_groups[0]["lr"], _iter)
                    _iter += 1
                    self.adjust_learning_rate(opt, _iter, len(train_loader), epoch)

            if valid_loader is not None:
                valid_acc, valid_adv_acc = self.valid(model, valid_loader)
                valid_acc, valid_adv_acc = valid_acc * 100, valid_adv_acc * 100
                if valid_adv_acc >= self.best_robust_acc:
                    self.best_clean_acc = valid_acc
                    self.best_robust_acc = valid_adv_acc
                    self.save_checkpoint(model, epoch, is_best=True)

                if self.writer is not None:
                    self.writer.add_scalar('Valid/Clean_acc', valid_acc, epoch)
                    self.writer.add_scalar(f'Valid/{self.get_attack_name(train=False)}_Accuracy', valid_adv_acc, epoch)

            if self.args.n_checkpoint_step != -1 and epoch % self.args.n_checkpoint_step == 0:
                self.save_checkpoint(model, epoch)
