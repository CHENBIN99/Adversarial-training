import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.utils import *
from dataloader.autoaugment import *


class MultiDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform(sample)
        return x1, x2


def get_dataloader(args):
    # image size
    if args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'svhn':
        image_size = 32
    elif args.dataset == 'tinyimagenet':
        image_size = 64
    else:
        raise NotImplemented

    # dataset transforms
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    if 'ccg' not in args.at_method:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                              (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        # choose dataset
        if args.dataset == 'cifar10':
            train_dataset = torchvision.datasets.CIFAR10(os.path.join(args.root_path, args.data_root),
                                                         train=True,
                                                         transform=transform_train,
                                                         download=True)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_works, pin_memory=True)
            # valid
            valid_dataset = torchvision.datasets.CIFAR10(os.path.join(args.root_path, args.data_root),
                                                         train=False,
                                                         transform=transform_test,
                                                         download=True)
            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_works, pin_memory=True)

        elif args.dataset == 'cifar100':
            train_dataset = torchvision.datasets.CIFAR100(os.path.join(args.root_path, args.data_root),
                                                          train=True, transform=transform_train, download=True)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_works, pin_memory=True)
            # valid
            valid_dataset = torchvision.datasets.CIFAR100(os.path.join(args.root_path, args.data_root),
                                                          train=False, transform=transform_test, download=True)
            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_works, pin_memory=True)
        elif args.dataset == 'tinyimagenet':
            from dataloader.tiny_imagenet import TinyImageNet
            if not os.path.exists(os.path.join(args.root_path, args.data_root, 'tiny-imagenet-200')):
                download_tinyimagenet(args)

            train_dataset = TinyImageNet(root=os.path.join(args.root_path,
                                                           args.data_root,
                                                           'tiny-imagenet-200'),
                                         train=True, transform=transform_train)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                       num_workers=args.num_works, pin_memory=True)

            valid_dataset = TinyImageNet(root=os.path.join(args.root_path,
                                                           args.data_root,
                                                           'tiny-imagenet-200'),
                                         train=False, transform=transform_test)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                                       num_workers=args.num_works, pin_memory=True)
        else:
            raise 'no match dataset'

        return train_loader, valid_loader

    else:
        transform_train_1 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                              (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        train_dataset_1 = torchvision.datasets.CIFAR10(os.path.join(args.root_path, args.data_root),
                                                       train=True,
                                                       transform=MultiDataTransform(transform_train_1),
                                                       download=True)
        train_dataloader_1 = torch.utils.data.DataLoader(train_dataset_1, batch_size=args.batch_size, shuffle=True,
                                                         num_workers=args.num_works, pin_memory=True)
        valid_dataset = torchvision.datasets.CIFAR10(os.path.join(args.root_path, args.data_root),
                                                     train=True,
                                                     transform=transform_train_1,
                                                     download=True)
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                                       num_workers=args.num_works, pin_memory=True)

        if args.dataset == 'cifar10':
            transform_train_2 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                  (4, 4, 4, 4), mode='reflect').squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),
                transforms.ToTensor(),
            ])
            train_dataset_2 = torchvision.datasets.CIFAR10(os.path.join(args.root_path, args.data_root),
                                                           train=True,
                                                           transform=MultiDataTransform(transform_train_2),
                                                           download=True)
            train_dataloader_2 = torch.utils.data.DataLoader(train_dataset_2, batch_size=args.batch_size, shuffle=True,
                                                             num_workers=args.num_works, pin_memory=True)

        elif args.dataset == 'cifar100':
            transform_train_2 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                  (4, 4, 4, 4), mode='reflect').squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),
                transforms.ToTensor(),
            ])
            train_dataset_2 = torchvision.datasets.CIFAR100(os.path.join(args.root_path, args.data_root),
                                                            train=True,
                                                            transform=MultiDataTransform(transform_train_2),
                                                            download=True)
            train_dataloader_2 = DataLoader(train_dataset_2, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_works,
                                        pin_memory=True)
        elif args.dataset == 'svhn':
            transform_train_2 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                  (4, 4, 4, 4), mode='reflect').squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                SVHNPolicy(),
                transforms.ToTensor(),
            ])
            train_dataset_2 = torchvision.datasets.SVHN(os.path.join(args.root_path, args.data_root),
                                                        split='train',
                                                        transform=MultiDataTransform(transform_train_2),
                                                        download=True)
            train_dataloader_2 = DataLoader(train_dataset_2, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_works,
                                        pin_memory=True)
        elif args.dataset == 'tinyimagenet':
            transform_train_2 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                  (4, 4, 4, 4), mode='reflect').squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                ImageNetPolicy(),
                transforms.ToTensor(),
            ])
            train_dataset_2 = torchvision.datasets.ImageFolder(root=os.path.join(args.root_path,
                                                                               args.data_root,
                                                                               'tiny-imagenet-200',
                                                                               'train'),
                                                               transform=MultiDataTransform(transform_train_2))
            train_dataloader_2 = torch.utils.data.DataLoader(train_dataset_2, batch_size=args.batch_size, shuffle=True,
                                                         num_workers=args.num_works, pin_memory=True)
        else:
            raise 'no match dataset'

        return [train_dataloader_1, train_dataloader_2], valid_dataloader



