"""
Standard Adversarial Training
"""
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.utils import *
from train.train_base import TrainerBase
from utils.AverageMeter import AverageMeter


class TrainerNature(TrainerBase):
    def __init__(self, cfg, writer, device, loss_function=torch.nn.CrossEntropyLoss()):
        super(TrainerNature, self).__init__(cfg, writer, device, loss_function)

    def train_one_epoch(self, model, train_loader, optimizer, epoch):
        nat_result = AverageMeter()
        with tqdm(total=len(train_loader)) as _tqdm:
            _tqdm.set_description('epoch:{}/{} Training:'.format(epoch + 1, self.cfg.TRAIN.epochs))
            for idx, (data, label) in enumerate(train_loader):
                n = data.size(0)
                data, label = data.to(self.device), label.to(self.device)
                adv_output = model(data)
                # Loss
                loss = self.loss_fn(adv_output, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Validation during training
                if self._iter % self.cfg.print_freq == 0:
                    # clean data
                    with torch.no_grad():
                        nat_output = model(data)
                    nat_correct_num = (torch.max(nat_output, dim=1)[1].cpu().detach() == label.cpu().numpy()).sum()
                    nat_result.update(nat_correct_num, n)

                    _tqdm.set_postfix(loss='{:.3f}'.format(loss.item()), nat_acc='{:.3f}'.format(nat_result.acc_cur))
                    _tqdm.update(self.cfg.TRAIN.print_freq)

                    if self.writer is not None:
                        self.writer.add_scalar('Train/Loss', loss.item(), epoch * len(train_loader) + idx)
                        self.writer.add_scalar('Train/Clean_acc', nat_result.acc_cur, epoch * len(train_loader) + idx)
                        self.writer.add_scalar('Train/Lr', optimizer.param_groups[0]["lr"],
                                               epoch * len(train_loader) + idx)
                self._iter += 1
