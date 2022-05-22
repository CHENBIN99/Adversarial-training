import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import datetime
import torch.nn.functional as F
import torchvision
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms

from model import wideresnet, preactresnet
from utils.args import parser
from utils.utils import *

from train import train_trades


def main(args):
    project_path = get_project_path()
    setattr(args, 'root_path', project_path)

    cur_time = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    if args.at_method == 'standard':
        exp_name = f'Standard_{args.dataset}_{args.learning_rate}_{cur_time}'
    elif args.at_method == 'trades':
        exp_name = f'TRADES_{args.beta}_{args.dataset}_{args.learning_rate}_{cur_time}'
    else:
        raise 'no match at method'

    tb_folder = os.path.join(project_path, args.save_root, args.exp_series, exp_name)
    model_folder = os.path.join(project_path, args.model_root, args.exp_series, exp_name)

    if args.tensorboard:
        if not os.path.exists(tb_folder):
            os.makedirs(tb_folder)
        setattr(args, 'tb_folder', tb_folder)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
        setattr(args, 'model_folder', model_folder)

    if args.tensorboard:
        # add tensorboard
        tb_writer = SummaryWriter(log_dir=tb_folder)
    else:
        tb_writer = None

    # use WRN-34-10
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    if args.model_name == 'wrn34-10':
        model = wideresnet.WideResNet(depth=34,
                                      widen_factor=10,
                                      num_classes=10 if args.dataset == 'cifar10' else 100,
                                      dropRate=0.0).to(device)
    elif args.model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(512, 10 if args.dataset == 'cifar10' else 100)
        model.to(device)
    elif args.model_name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(2048, 10 if args.dataset == 'cifar10' else 100)
        model.to(device)
    elif args.model_name == 'preactresnet18':
        model = preactresnet.PreActResNet18(num_classes=10 if args.dataset == 'cifar10' else 100)
        model.to(device)
    else:
        raise 'no match model'

    if args.at_method == 'trades':
        trainer = train_trades.Trainer_Trades(args, tb_writer, args.attack_method, device)


    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                          (4, 4, 4, 4), mode='reflect').squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    if args.dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(os.path.join(project_path, args.data_root),
                                                     train=True,
                                                     transform=transform_train,
                                                     download=True)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_works)
        # valid
        valid_dataset = torchvision.datasets.CIFAR10(os.path.join(project_path, args.data_root),
                                                     train=False,
                                                     transform=transform_test,
                                                     download=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_works)

    elif args.dataset == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(os.path.join(project_path, args.data_root),
                                                      train=True,
                                                      transform=transform_train,
                                                      download=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_works)
        # valid
        valid_dataset = torchvision.datasets.CIFAR100(os.path.join(project_path, args.data_root),
                                                      train=False,
                                                      transform=transform_test,
                                                      download=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_works)
    elif args.dataset == 'svhn':
        train_dataset = torchvision.datasets.SVHN(os.path.join(project_path, args.data_root),
                                                  split='train',
                                                  transform=transform_train,
                                                  download=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_works)
        valid_dataset = torchvision.datasets.SVHN(os.path.join(project_path, args.data_root),
                                                  split='test',
                                                  transform=transform_test,
                                                  download=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_works)
    else:
        raise 'no match dataset'

    trainer.train(model, train_loader, valid_loader, args.adv_train)

    print('Train Finished!')


if __name__ == '__main__':
    args = parser()
    main(args)

