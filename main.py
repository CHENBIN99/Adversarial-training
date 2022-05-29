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

from train import train_standard, train_trades


# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)                     # 固定随机种子（CPU）
    if torch.cuda.is_available():               # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)            # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)        # 为所有GPU设置
    np.random.seed(seed)                        # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = False      # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True   # 固定网络结构


def main(args):
    same_seeds(args.seed)
    project_path = get_project_path()
    setattr(args, 'root_path', project_path)

    cur_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
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

    # choose model
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'tinyimagenet':
        num_classes = 200
    else:
        raise 'no match dataset'

    if args.model_name == 'wrn34-10':
        model = wideresnet.WideResNet(depth=34,
                                      widen_factor=10,
                                      num_classes=num_classes,
                                      dropRate=0.0).to(device)
    elif args.model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(512, num_classes)
        model.to(device)
    elif args.model_name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(2048, num_classes)
        model.to(device)
    elif args.model_name == 'preactresnet18':
        model = preactresnet.PreActResNet18(num_classes=num_classes)
        model.to(device)
    else:
        raise 'no match model'

    # choose adversarial training method
    if args.at_method == 'standard':
        trainer = train_standard.Trainer_Standard(args, tb_writer, args.attack_method, device)
    elif args.at_method == 'trades':
        trainer = train_trades.Trainer_Trades(args, tb_writer, args.attack_method, device)
    else:
        raise 'no match at_method'

    # dataset transforms
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

    # choose dataset
    if args.dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(os.path.join(project_path, args.data_root),
                                                     train=True,
                                                     transform=transform_train,
                                                     download=True)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_works,
                                  pin_memory=True)
        # valid
        valid_dataset = torchvision.datasets.CIFAR10(os.path.join(project_path, args.data_root),
                                                     train=False,
                                                     transform=transform_test,
                                                     download=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_works,
                                  pin_memory=True)

    elif args.dataset == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(os.path.join(project_path, args.data_root),
                                                      train=True,
                                                      transform=transform_train,
                                                      download=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_works,
                                  pin_memory=True)
        # valid
        valid_dataset = torchvision.datasets.CIFAR100(os.path.join(project_path, args.data_root),
                                                      train=False,
                                                      transform=transform_test,
                                                      download=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_works,
                                  pin_memory=True)

    elif args.dataset == 'svhn':
        train_dataset = torchvision.datasets.SVHN(os.path.join(project_path, args.data_root),
                                                  split='train',
                                                  transform=transform_train,
                                                  download=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_works,
                                  pin_memory=True)
        valid_dataset = torchvision.datasets.SVHN(os.path.join(project_path, args.data_root),
                                                  split='test',
                                                  transform=transform_test,
                                                  download=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_works,
                                  pin_memory=True)

    elif args.dataset == 'tinyimagenet':
        train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(project_path,
                                                                           args.data_root,
                                                                           'tiny-imagenet-200',
                                                                           'train'),
                                                         transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_works, pin_memory=True)
        valid_dataset = torchvision.datasets.ImageFolder(root=os.path.join(project_path,
                                                                           args.data_root,
                                                                           'tiny-imagenet-200',
                                                                           'val'),
                                                         transform=transform_train)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=args.num_works, pin_memory=True)

    else:
        raise 'no match dataset'

    trainer.train(model, train_loader, valid_loader, args.adv_train)

    print('Train Finished!')


if __name__ == '__main__':
    args = parser()
    main(args)

