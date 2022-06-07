import os
import numpy as np
import torch
import glob
from shutil import move
import datetime

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


def get_exp_name(args):
    # time
    curr_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

    # model name
    if args.model_name == 'wrn34-10':
        model_name = 'wrn3410'
    elif args.model_name == 'resnet18':
        model_name = 'resnet18'
    elif args.model_name == 'resnet50':
        model_name = 'resnet50'
    elif args.model_name == 'preactresnet18':
        model_name = 'preact18'
    else:
        raise 'no match model'

    # attack method
    if args.at_method == 'standard':
        exp_name = f'Standard_{args.dataset}_{model_name}_{args.learning_rate}_{curr_time}'
    elif args.at_method == 'trades':
        exp_name = f'TRADES_{args.beta}_{args.dataset}_{model_name}_{args.learning_rate}_{curr_time}'
    elif args.at_method == 'mart':
        exp_name = f'MART_{args.beta}_{args.dataset}_{model_name}_{args.learning_rate}_{curr_time}'
    else:
        raise 'no match at method'

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
    os.system(f'wget -p {os.path.join(args.root_path, args.data_root)} {url}')
    os.system(f'unzip {os.path.join(args.root_path, args.data_root, "tiny-imagenet-200.zip")} '
              f'-d '
              f'{os.path.join(args.root_path, args.data_root, "tiny-imagenet-200")}')
    deal_tinyimagenet(os.path.join(args.root_path, args.data_root, "tiny-imagenet-200"))


def deal_tinyimagenet(folder_path):
    target_folder = os.path.join(folder_path, 'val')

    val_dict = {}
    with open('./tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
        for line in f.readlines():
            split_line = line.split('\t')
            val_dict[split_line[0]] = split_line[1]

    paths = glob.glob('./tiny-imagenet-200/val/images/*')
    for path in paths:
        file = path.split('/')[-1]
        folder = val_dict[file]
        if not os.path.exists(target_folder + str(folder)):
            os.mkdir(target_folder + str(folder))
            os.mkdir(target_folder + str(folder) + '/images')

    for path in paths:
        file = path.split('/')[-1]
        folder = val_dict[file]
        dest = target_folder + str(folder) + '/images/' + str(file)
        move(path, dest)

    os.rmdir('./tiny-imagenet-200/val/images')

