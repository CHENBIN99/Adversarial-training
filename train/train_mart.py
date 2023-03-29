"""
MART
paper: IMPROVING ADVERSARIAL ROBUSTNESS REQUIRES REVISITING MISCLASSIFIED EXAMPLES
"""
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train.train_base import TrainerBase
from adv_lib.mart_lib import *
from utils.utils import *
from utils.AverageMeter import AverageMeter


class TrainerMart(TrainerBase):
    def __init__(self, args, writer, attack_name, device, loss_function=torch.nn.CrossEntropyLoss()):
        super(TrainerMart, self).__init__(args, writer, attack_name, device, loss_function)

    def train_one_epoch(self, model, train_loader, optimizer, epoch):
        nat_result = AverageMeter()
        adv_result = AverageMeter()
        with tqdm(total=len(train_loader)) as _tqdm:
            _tqdm.set_description('epoch:{}/{} Training:'.format(epoch + 1, self.cfg.TRAIN.epochs))
            for idx, (data, label) in enumerate(train_loader):
                n = data.size(0)
                data, label = data.to(self.device), label.to(self.device)

                # MART Loss
                loss_adv, loss_mart, adv_data = mart_loss(model=model, x_natural=data, y=label, optimizer=optimizer,
                                                          step_size=self.cfg.ADV.alpha, epsilon=self.cfg.ADV.eps,
                                                          perturb_steps=self.cfg.ADV.iters, beta=self.cfg.TRAIN.beta,
                                                          distance='l_inf', device=self.device)
                loss = loss_adv + loss_mart
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Validation during training
                if (idx + 1) % self.cfg.TRAIN.print_freq == 0 or (idx + 1) == len(train_loader):
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

                    _tqdm.set_postfix(loss='{:.3f}'.format(loss.item()),
                                      nat_acc='{:.3f}'.format(nat_result.acc_cur * 100),
                                      rob_acc='{:.3f}'.format(adv_result.acc_cur * 100))
                    if not idx + 1 == len(train_loader):
                        _tqdm.update(self.cfg.TRAIN.print_freq)
                    else:
                        _tqdm.update(len(train_loader) % self.cfg.TRAIN.print_freq)

                    if self.writer is not None:
                        self.writer.add_scalar('Train/Loss_adv', loss_adv.item(), self._iter)
                        self.writer.add_scalar('Train/Loss_mart', loss_mart.item(), self._iter)
                        self.writer.add_scalar('Train/Clean_acc', nat_result.acc_cur * 100, self._iter)
                        self.writer.add_scalar(f'Train/{self._get_attack_name()}_accuracy', adv_result.acc_cur * 100,
                                               self._iter)
                        self.writer.add_scalar('Train/Lr', optimizer.param_groups[0]["lr"], self._iter)
                self.adjust_learning_rate(optimizer, len(train_loader), epoch)
                self._iter += 1
