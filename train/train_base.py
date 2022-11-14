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
                                     eps=self.args.epsilon)
        else:
            raise 'no match attack method'

    def get_attack_name(self, train=True, upper=True):
        if self.attack_name == 'pgd':
            if train:
                return f'PGD{self.args.iters}'
            else:
                return f'PGD{self.args.iters_eval}'
        elif self.attack_name == 'fgsm':
            return 'FGSM'

    def save_checkpoint(self, model, epoch, is_best=False):
        if not is_best:
            file_name = os.path.join(self.args.model_folder, f'checkpoint_{epoch}.pth')
        else:
            file_name = os.path.join(self.args.model_folder, f'checkpoint_best.pth')
        torch.save(model.state_dict(), file_name)

    def train(self, **kwargs):
        pass

    def valid(self, model, valid_loader, use_pseudo_label=False):
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
                        adv_data = attack_method(data, pred if use_pseudo_label else label)
                    adv_output = model(adv_data)
                    adv_pred = torch.max(adv_output, dim=1)[1]
                    adv_acc_num = evaluate(adv_pred.cpu().numpy(), label.cpu().numpy(), 'sum')

                    total_correct_nat += std_acc_num
                    total_correct_adv += adv_acc_num
                    num += output.shape[0]
                    total_acc_nat = total_correct_nat / num
                    total_acc_adv = total_correct_adv / num

                    _tqdm.set_postfix(nat_acc='{:.3f}'.format(total_acc_nat), rob_acc='{:.3f}'.format(total_acc_adv))
                    _tqdm.update(1)

        model.train()

        return total_acc_nat, total_acc_adv

