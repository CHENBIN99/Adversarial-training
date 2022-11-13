import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
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
        if args.image_size == -1:
            image_size = 32
        else:
            image_size = args.image_size
    elif args.dataset == 'tinyimagenet':
        if args.image_size == -1:
            image_size = 64
        else:
            image_size = args.image_size
    elif args.dataset == 'imagenet':
        if args.image_size == -1:
            image_size = 224
        else:
            image_size = args.image_size
    else:
        raise NotImplemented

    setattr(args, 'image_size', image_size)

    # dataset transforms
    transform_test = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    if 'ccg' not in args.at_method:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                              (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop((image_size, image_size)),
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
        elif args.dataset == 'imagenet':
            train_dataset = datasets.ImageFolder(os.path.join(args.root_path, args.data_root, 'ImageNet', 'train'),
                                                 transform_train)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                           num_workers=args.num_works, pin_memory=True, drop_last=True)
            valid_dataset = datasets.ImageFolder(os.path.join(args.root_path, args.data_root, 'ImageNet', 'val/'),
                                                 transform_test)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True,
                                                           num_workers=args.num_works, pin_memory=True, drop_last=True)
        else:
            raise NotImplemented

        return train_loader, valid_loader

    else:
        transform_train_1 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                              (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        if args.dataset == 'cifar10':
            transform_train_2 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((image_size, image_size)),
                transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                  (4, 4, 4, 4), mode='reflect').squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),
                transforms.ToTensor(),
            ])
            train_dataset_1 = torchvision.datasets.CIFAR10(os.path.join(args.root_path, args.data_root),
                                                           train=True,
                                                           transform=MultiDataTransform(transform_train_1),
                                                           download=True)
            train_dataset_2 = torchvision.datasets.CIFAR10(os.path.join(args.root_path, args.data_root),
                                                           train=True,
                                                           transform=MultiDataTransform(transform_train_2),
                                                           download=True)
            valid_dataset = torchvision.datasets.CIFAR10(os.path.join(args.root_path, args.data_root),
                                                         train=True,
                                                         transform=transform_test,
                                                         download=True)

            train_dataloader_1 = torch.utils.data.DataLoader(train_dataset_1, batch_size=args.batch_size, shuffle=True,
                                                             num_workers=args.num_works, pin_memory=True)
            train_dataloader_2 = torch.utils.data.DataLoader(train_dataset_2, batch_size=args.batch_size, shuffle=True,
                                                             num_workers=args.num_works, pin_memory=True)
            valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                                           num_workers=args.num_works, pin_memory=True)

        elif args.dataset == 'cifar100':
            transform_train_2 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((image_size, image_size)),
                transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                  (4, 4, 4, 4), mode='reflect').squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),
                transforms.ToTensor(),
            ])
            train_dataset_1 = torchvision.datasets.CIFAR100(os.path.join(args.root_path, args.data_root),
                                                            train=True,
                                                            transform=MultiDataTransform(transform_train_1),
                                                            download=True)
            train_dataset_2 = torchvision.datasets.CIFAR100(os.path.join(args.root_path, args.data_root),
                                                            train=True,
                                                            transform=MultiDataTransform(transform_train_2),
                                                            download=True)
            valid_dataset = torchvision.datasets.CIFAR100(os.path.join(args.root_path, args.data_root),
                                                          train=True,
                                                          transform=transform_test,
                                                          download=True)
            train_dataloader_1 = torch.utils.data.DataLoader(train_dataset_1, batch_size=args.batch_size, shuffle=True,
                                                             num_workers=args.num_works, pin_memory=True)
            train_dataloader_2 = torch.utils.data.DataLoader(train_dataset_2, batch_size=args.batch_size, shuffle=True,
                                                             num_workers=args.num_works, pin_memory=True)
            valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                                           num_workers=args.num_works, pin_memory=True)
        elif args.dataset == 'tinyimagenet':
            from dataloader.tiny_imagenet import TinyImageNet
            if not os.path.exists(os.path.join(args.root_path, args.data_root, 'tiny-imagenet-200')):
                download_tinyimagenet(args)

            transform_train_2 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((image_size, image_size)),
                transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                  (4, 4, 4, 4), mode='reflect').squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),
                transforms.ToTensor(),
            ])

            train_dataset_1 = TinyImageNet(root=os.path.join(args.root_path, args.data_root, 'tiny-imagenet-200'),
                                           train=True, transform=MultiDataTransform(transform_train_1))
            train_dataset_2 = TinyImageNet(root=os.path.join(args.root_path, args.data_root, 'tiny-imagenet-200'),
                                           train=True, transform=MultiDataTransform(transform_train_2))
            valid_dataset = TinyImageNet(root=os.path.join(args.root_path, args.data_root, 'tiny-imagenet-200'),
                                           train=False, transform=transform_test)

            train_dataloader_1 = torch.utils.data.DataLoader(train_dataset_1, batch_size=args.batch_size, shuffle=True,
                                                         num_workers=args.num_works, pin_memory=True)
            train_dataloader_2 = torch.utils.data.DataLoader(train_dataset_2, batch_size=args.batch_size, shuffle=True,
                                                         num_workers=args.num_works, pin_memory=True)
            valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True,
                                                       num_workers=args.num_works, pin_memory=True)
        else:
            raise 'no match dataset'

        return [train_dataloader_1, train_dataloader_2], valid_dataloader



