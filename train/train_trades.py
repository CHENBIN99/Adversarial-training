"""
TRADES
paper: Theoretically Principled Trade-off between Robustness and Accuracy
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.utils import *

import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
from train.train_base import Trainer_base

from adv_lib.trades_lib import *


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss, x_adv


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

                loss, adv_data = trades_loss(model=model, x_natural=data, y=label, optimizer=opt,
                                             step_size=self.args.alpha, epsilon=self.args.epsilon,
                                             perturb_steps=self.args.iters, beta=self.args.beta, distance='l_inf')

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

