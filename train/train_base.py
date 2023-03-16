"""
正常的对抗训练
"""

import torchattacks
import torch
from tqdm import tqdm

from utils.utils import *


class Trainer_base:
    def __init__(self, args, writer, attack_name, device, loss_function=torch.nn.CrossEntropyLoss()):
        self.args = args
        self.writer = writer
        self.device = device
        self.attack_name = attack_name
        self.loss_fn = loss_function
        # log
        self.best_clean_acc = 0.
        self.best_robust_acc = 0.

    def get_attack(self, model, epsilon, alpha, iters):
        if self.attack_name == 'pgd':
            return torchattacks.PGD(model=model,
                                    eps=epsilon,
                                    alpha=alpha,
                                    steps=iters,
                                    random_start=True)
        elif self.attack_name == 'fgsm':
            return torchattacks.FGSM(model=model,
                                     eps=epsilon)
        elif self.attack_name == 'rfgsm':
            return torchattacks.RFGSM(model=model,
                                      eps=epsilon,
                                      alpha=alpha,
                                      steps=iters)
        else:
            raise 'no match attack method'

    def get_attack_name(self, train=True, upper=True):
        if self.attack_name == 'pgd':
            if train:
                return f'PGD-{self.args.iters}'
            else:
                return f'PGD-{self.args.iters_eval}'
        elif self.attack_name == 'fgsm':
            return 'FGSM'
        elif self.attack_name == 'rfgsm':
            if train:
                return f'RFGSM-{self.args.iters}'
            else:
                return f'RFGSM-{self.args.iters_eval}'

    def adjust_learning_rate(self, opt, cur_iters, len_loader, epoch):
        """
        Adjust the learning rate during training.
        :param opt: optimizer
        :param cur_iters: current iteration
        :param len_loader: the total number of mini-batch
        :param epoch: current epoch
        :return: None
        """
        if self.args.lr_schedule == 'milestone':
            if epoch < int(self.args.ms_1 * self.args.max_epochs):
                lr = self.args.learning_rate * 0.1
            elif epoch < int(self.args.ms_2 * self.args.max_epochs):
                lr = self.args.learning_rate * 0.01
            elif epoch < int(self.args.ms_3 * self.args.max_epochs):
                lr = self.args.learning_rate * 0.001
        elif self.args.lr_schedule == 'cycle_1':
            cycle = [0, 2, 12, 24, 30]
            lr_cycle = [1e-6, 0.4, 0.04, 0.004, 0.0004]
            if cur_iters < len_loader * cycle[1]:
                lr = lr_cycle[0] + (lr_cycle[1] - lr_cycle[0]) / (len_loader * (cycle[1] - cycle[0])) * \
                     (cur_iters - len_loader * cycle[0])
            elif cur_iters < len_loader * cycle[2]:
                lr = lr_cycle[1] - (lr_cycle[1] - lr_cycle[2]) / (len_loader * (cycle[2] - cycle[1])) * \
                     (cur_iters - len_loader * cycle[1])
            elif cur_iters < len_loader * cycle[3]:
                lr = lr_cycle[2] - (lr_cycle[2] - lr_cycle[3]) / (len_loader * (cycle[3] - cycle[2])) * \
                     (cur_iters - len_loader * cycle[2])
            elif cur_iters <= len_loader * cycle[4]:
                lr = lr_cycle[3] - (lr_cycle[3] - lr_cycle[4]) / (len_loader * (cycle[4] - cycle[3])) * \
                     (cur_iters - len_loader * cycle[3])
        elif self.args.lr_schedule == 'cycle_2':
            cycle = [0, 1, 10, 45, 50]
            lr_cycle = [1e-6, 0.2, 0.1, 0.01, 0.001]
            if cur_iters < len_loader * cycle[1]:
                lr = lr_cycle[0] + (lr_cycle[1] - lr_cycle[0]) / (len_loader * (cycle[1] - cycle[0])) * \
                     (cur_iters - len_loader * cycle[0])
            elif cur_iters < len_loader * cycle[2]:
                lr = lr_cycle[1] - (lr_cycle[1] - lr_cycle[2]) / (len_loader * (cycle[2] - cycle[1])) * \
                     (cur_iters - len_loader * cycle[1])
            elif cur_iters < len_loader * cycle[3]:
                lr = lr_cycle[2] - (lr_cycle[2] - lr_cycle[3]) / (len_loader * (cycle[3] - cycle[2])) * \
                     (cur_iters - len_loader * cycle[2])
            elif cur_iters <= len_loader * cycle[4]:
                lr = lr_cycle[3] - (lr_cycle[3] - lr_cycle[4]) / (len_loader * (cycle[4] - cycle[3])) * \
                     (cur_iters - len_loader * cycle[3])
        elif self.args.lr_schedule == 'cycle_3':
            cycle = [0, 1, 40, 45, 50]
            lr_cycle = [1e-6, 0.1, 0.1, 0.01, 0.001]
            if cur_iters < len_loader * cycle[1]:
                lr = lr_cycle[0] + (lr_cycle[1] - lr_cycle[0]) / (len_loader * (cycle[1] - cycle[0])) * \
                     (cur_iters - len_loader * cycle[0])
            elif cur_iters < len_loader * cycle[2]:
                lr = lr_cycle[1] - (lr_cycle[1] - lr_cycle[2]) / (len_loader * (cycle[2] - cycle[1])) * \
                     (cur_iters - len_loader * cycle[1])
            elif cur_iters < len_loader * cycle[3]:
                lr = lr_cycle[2] - (lr_cycle[2] - lr_cycle[3]) / (len_loader * (cycle[3] - cycle[2])) * \
                     (cur_iters - len_loader * cycle[2])
            elif cur_iters <= len_loader * cycle[4]:
                lr = lr_cycle[3] - (lr_cycle[3] - lr_cycle[4]) / (len_loader * (cycle[4] - cycle[3])) * \
                     (cur_iters - len_loader * cycle[3])
        else:
            raise NotImplemented

        for param_group in opt.param_groups:
            param_group["lr"] = lr

    def save_checkpoint(self, model, epoch, is_best=False):
        if not is_best:
            file_name = os.path.join(self.args.model_folder, f'checkpoint_{epoch}.pth')
        else:
            file_name = os.path.join(self.args.model_folder, f'checkpoint_best.pth')
        torch.save(model.state_dict(), file_name)

    def train(self, **kwargs):
        pass

    def valid(self, model, valid_loader):
        total_correct_nat = 0
        total_correct_adv = 0
        total_acc_nat = 0.
        total_acc_adv = 0.
        num = 0

        attack_method = self.get_attack(model, self.args.epsilon, self.args.alpha, self.args.iters_eval)

        model.eval()

        with torch.no_grad():
            with tqdm(total=len(valid_loader)) as _tqdm:
                _tqdm.set_description('Validating:')
                for idx, (data, label) in enumerate(valid_loader):
                    data, label = data.to(self.device), label.to(self.device)
                    output = model(data)
                    pred = torch.max(output, dim=1)[1]
                    std_acc_num = evaluate(pred.cpu().numpy(), label.cpu().numpy(), 'sum')

                    with torch.enable_grad():
                        adv_data = attack_method(data, label)
                    adv_output = model(adv_data)
                    adv_pred = torch.max(adv_output, dim=1)[1]
                    adv_acc_num = evaluate(adv_pred.cpu().numpy(), label.cpu().numpy(), 'sum')

                    total_correct_nat += std_acc_num
                    total_correct_adv += adv_acc_num
                    num += output.shape[0]
                    total_acc_nat = total_correct_nat / num
                    total_acc_adv = total_correct_adv / num

                    _tqdm.set_postfix(nat_acc='{:.3f}'.format(total_acc_nat * 100),
                                      rob_acc='{:.3f}'.format(total_acc_adv * 100))
                    _tqdm.update(1)

        model.train()

        return total_acc_nat, total_acc_adv

