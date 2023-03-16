"""
Standard Adversarial Training
"""
import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.utils import *
from train.train_base import Trainer_base


class Trainer_Ens(Trainer_base):
    def __init__(self, args, writer, attack_name, device, loss_function=torch.nn.CrossEntropyLoss()):
        super(Trainer_Ens, self).__init__(args, writer, attack_name, device, loss_function)

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
