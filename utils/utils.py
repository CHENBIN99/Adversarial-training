import os
import numpy as np
import torch
import glob
from shutil import move
import datetime
import time
from easydict import EasyDict
import yaml

import torchvision
from torch.utils.data import DataLoader


# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)                     # 固定随机种子（CPU）
    if torch.cuda.is_available():               # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)            # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)        # 为所有GPU设置
    np.random.seed(seed)                        # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = False      # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True   # 固定网络结构


def get_exp_name(at_method, dataset, config):
    # time
    curr_time = time.strftime("%m%d%H%M")

    # attack method
    if at_method == 'nature':
        exp_name = 'Nature'
    elif at_method == 'standard':
        exp_name = f'Standard'
    elif at_method == 'at_free':
        exp_name = f'AT-Free'
    elif at_method == 'at_fast':
        exp_name = f'AT-Fast'
    elif at_method == 'at_ens':
        exp_name = f'EnsAT_static-{config.static_model}'
    elif at_method == 'trades':
        exp_name = f'TRADES_beta-{config.trades_beta}'
    elif at_method == 'mart':
        exp_name = f'MART_beta-{config.mart_beta}'
    elif at_method == 'mart_trades':
        exp_name = f'MART_beta-{config.mart_beta}_TRADES_beta-{config.trades_beta}'
    elif at_method == 'ccg':
        exp_name = f'CCG'
    elif at_method == 'ccg_trades':
        exp_name = f'CCG_TRADES_beta-{config.trades_beta}'
    else:
        raise 'no match at method'

    exp_name += f'_{dataset}_{config.TRAIN.arch}_lr-{config.TRAIN.lr}_seed-{config.TRAIN.seed}_t-{curr_time}'

    return exp_name


def evaluate(_input, _target, method='mean'):
    correct = (_input == _target).astype(np.float32)
    if method == 'mean':
        return correct.mean()
    else:
        return correct.sum()


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def get_project_path():
    """得到项目路径"""
    project_path = os.path.join(
        os.path.dirname(__file__),
        "..",
    )
    return os.path.abspath(project_path)


def download_tinyimagenet(args):
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    os.system(f'wget -P {os.path.join(args.root_path, args.data_root)} {url}')
    os.system(f'unzip {os.path.join(args.root_path, args.data_root, "tiny-imagenet-200.zip")} '
              f'-d '
              f'{os.path.join(args.root_path, args.data_root, "tiny-imagenet-200")}')


def parse_config_file(args):
    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    # Add args parameters to the dict
    for k, v in vars(args).items():
        config[k] = v

    # Add the output path
    config.exp_name = get_exp_name(args.method, args.dataset, config)
    config.root_path = get_project_path()
    config.log_path = os.path.join(config.root_path, config.SAVE.log, config.exp_name)
    config.ckp_path = os.path.join(config.root_path, config.SAVE.checkpoint, config.exp_name)
    config.ADV.TRAIN.eps = config.ADV.TRAIN.eps / 255.
    config.ADV.TRAIN.alpha = config.ADV.TRAIN.alpha / 255.
    config.ADV.EVAL.eps = config.ADV.EVAL.eps / 255.
    config.ADV.EVAL.alpha = config.ADV.EVAL.alpha / 255.

    return config
