import argparse
import os
import sys
import torch.nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'model', 'classifiers')))

from tensorboardX import SummaryWriter
from utils.utils import *
from dataloader import get_dataloader
from model.registry import get_model, get_static_model


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('-c', '--config', default='configs.yml', type=str, metavar='Path',
                        help='path to the config file (default: configs.yml)')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        help='training dataset, e.g. cifar10/100 or imagenet')
    parser.add_argument('-m', '--method', default='standard', type=str,
                        help='at method, e.g. standard at, trades, mart, etc.')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--use_log', action='store_true',
                        help='use Tensorboard to log train data')
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='gpu id used for training.')
    return parser.parse_args()


# Parase config file and initiate logging
configs = parse_config_file(parse_args())


def main(cfg):
    same_seeds(cfg.TRAIN.seed)

    # create dir
    if cfg.use_log:
        if not os.path.exists(cfg.log_path):
            os.makedirs(cfg.log_path)
        tb_writer = SummaryWriter(log_dir=cfg.log_path)
    else:
        tb_writer = None
    if not os.path.exists(cfg.ckp_path):
        os.makedirs(cfg.ckp_path)

    # device
    device = torch.device(f'cuda:{cfg.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # get dataloader
    train_dataloader, valid_dataloader = get_dataloader.get_dataloader(configs)

    model = get_model(cfg.TRAIN.arch, cfg.DATA.num_class, cfg.dataset, cfg.TRAIN.pretrain, cfg.TRAIN.compile)
    model.to(device)

    # choose adversarial training method
    if cfg.method == 'nature':
        from train.train_nature import TrainerNature
        trainer = TrainerNature(cfg, tb_writer, device)
    elif cfg.method == 'standard':
        from train.train_standard import TrainerStandard
        trainer = TrainerStandard(cfg, tb_writer, device)
    elif cfg.method == 'at_free':
        from train.train_at_free import TrainerFree
        trainer = TrainerFree(cfg, tb_writer, device, m=cfg.TRAIN.m)
    elif cfg.method == 'at_fast':
        from train.train_fast_at import TrainerFast
        trainer = TrainerFast(cfg, tb_writer, device, m=cfg.TRAIN.m, random_init=True)
    elif cfg.method == 'at_ens':
        from train.train_ens_adv import TrainerEns
        model_static = get_static_model(cfg.TRAIN.static_model_id, cfg.DATA.num_class, device)
        trainer = TrainerEns(cfg, tb_writer, device, static_model=model_static)
    elif cfg.method == 'trades':
        from train.train_trades import TrainerTrades
        trainer = TrainerTrades(cfg, tb_writer, device)
    elif cfg.method == 'mart':
        from train.train_mart import TrainerMart
        trainer = TrainerMart(cfg, tb_writer, device)
    elif cfg.method == 'mart_trades':
        from train.train_mart_trades import TrainerMartTrades
        trainer = TrainerMartTrades(cfg, tb_writer, device)
    elif cfg.method == 'ccg':
        from train.train_ccg import TrainerCCG
        trainer = TrainerCCG(cfg, tb_writer, device)
    elif cfg.method == 'ccg_trades':
        from train.train_ccg_trades import TrainerCCGTRADES
        trainer = TrainerCCGTRADES(cfg, tb_writer, device)
    else:
        raise 'no match at_method'

    trainer.train(model, train_dataloader, valid_dataloader)

    print('Train Finished!')


if __name__ == '__main__':
    main(configs)
