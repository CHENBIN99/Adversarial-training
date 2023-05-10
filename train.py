import init_path
import argparse
import torch.nn
from tensorboardX import SummaryWriter
from utils.utils import *
from dataloader import get_dataloader
from model.registry import get_model, get_static_model
import train


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
        trainer = train.TrainerNature(cfg, tb_writer, device)
    elif cfg.method == 'standard':
        trainer = train.TrainerStandard(cfg, tb_writer, device)
    elif cfg.method == 'at_free':
        trainer = train.TrainerFree(cfg, tb_writer, device, m=cfg.TRAIN.m)
    elif cfg.method == 'at_fast':
        trainer = train.TrainerFast(cfg, tb_writer, device, m=cfg.TRAIN.m, random_init=True)
    elif cfg.method == 'at_ens':
        model_static = get_static_model(cfg.TRAIN.static_model_id, cfg.DATA.num_class, device)
        trainer = train.TrainerEns(cfg, tb_writer, device, static_model=model_static)
    elif cfg.method == 'trades':
        trainer = train.TrainerTrades(cfg, tb_writer, device)
    elif cfg.method == 'mart':
        trainer = train.TrainerMart(cfg, tb_writer, device)
    elif cfg.method == 'mart_trades':
        trainer = train.TrainerMartTrades(cfg, tb_writer, device)
    elif cfg.method == 'ccg':
        trainer = train.TrainerCCG(cfg, tb_writer, device)
    elif cfg.method == 'ccg_trades':
        trainer = train.TrainerCCGTRADES(cfg, tb_writer, device)
    else:
        raise 'no match at_method'

    trainer.train(model, train_dataloader, valid_dataloader)

    print('Train Finished!')


if __name__ == '__main__':
    main(configs)
