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


def get_dataloader(cfg):
    if 'ccg' in cfg.method:
        return get_dataloader_ccg(cfg)
    else:
        if not cfg.DATA.aug:
            # transform_train = transforms.Compose([
            #     transforms.Resize(cfg.DATA.image_size),
            #     transforms.RandomResizedCrop(cfg.DATA.crop_size),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=cfg.DATA.mean, std=cfg.DATA.std),
            # ])
            transform_train = transforms.Compose([
                transforms.RandomCrop(cfg.DATA.crop_size, padding=cfg.DATA.padding),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=cfg.DATA.mean, std=cfg.DATA.std),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.Resize(cfg.DATA.image_size),
                transforms.RandomResizedCrop(cfg.DATA.crop_size),
                policy[cfg.dataset],
                transforms.ToTensor(),
                transforms.Normalize(mean=cfg.DATA.mean, std=cfg.DATA.std),
            ])

        transform_test = transforms.Compose([
            transforms.Resize((cfg.DATA.crop_size, cfg.DATA.crop_size)),
            transforms.ToTensor(),
        ])

        if cfg.dataset == 'cifar10':
            train_dataset = torchvision.datasets.CIFAR10(os.path.join(cfg.root_path, cfg.DATA.data_path), train=True,
                                                         transform=transform_train, download=True)
            train_loader = DataLoader(train_dataset, batch_size=cfg.DATA.batch_size, shuffle=True,
                                      num_workers=cfg.DATA.workers, pin_memory=True)

            valid_dataset = torchvision.datasets.CIFAR10(os.path.join(cfg.root_path, cfg.DATA.data_path), train=False,
                                                         transform=transform_test, download=True)
            valid_loader = DataLoader(valid_dataset, batch_size=cfg.DATA.batch_size, shuffle=False,
                                      num_workers=cfg.DATA.workers, pin_memory=True)
        elif cfg.dataset == 'cifar100':
            train_dataset = torchvision.datasets.CIFAR100(os.path.join(cfg.root_path, cfg.DATA.data_path), train=True,
                                                          transform=transform_train, download=True)
            train_loader = DataLoader(train_dataset, batch_size=cfg.DATA.batch_size, shuffle=True,
                                      num_workers=cfg.DATA.workers, pin_memory=True)

            valid_dataset = torchvision.datasets.CIFAR100(os.path.join(cfg.root_path, cfg.DATA.data_path), train=False,
                                                          transform=transform_test, download=True)
            valid_loader = DataLoader(valid_dataset, batch_size=cfg.DATA.batch_size, shuffle=False,
                                      num_workers=cfg.DATA.workers, pin_memory=True)
        elif cfg.dataset == 'imagenet':
            train_dataset = datasets.ImageFolder(os.path.join(cfg.root_path, cfg.DATA.data_path, 'ImageNet', 'train'),
                                                 transform_train)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.DATA.batch_size, shuffle=True,
                                                       num_workers=cfg.DATA.workers, pin_memory=True, drop_last=True)
            valid_dataset = datasets.ImageFolder(os.path.join(cfg.root_path, cfg.DATA.data_path, 'ImageNet', 'val/'),
                                                 transform_test)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=cfg.DATA.batch_size, shuffle=True,
                                                       num_workers=cfg.DATA.workers, pin_memory=True, drop_last=True)

        else:
            raise NotImplemented

        return train_loader, valid_loader


def get_dataloader_ccg(cfg):
    transform_train_1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((cfg.DATA.image_size, cfg.DATA.image_size)),
        transforms.RandomResizedCrop((cfg.DATA.crop_size, cfg.DATA.crop_size), scale=(0.64, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=cfg.DATA.mean, std=cfg.DATA.std)
    ])

    transform_train_2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((cfg.DATA.image_size, cfg.DATA.image_size)),
        transforms.RandomResizedCrop((cfg.DATA.crop_size, cfg.DATA.crop_size), scale=(0.64, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=cfg.DATA.mean, std=cfg.DATA.std)
    ])

    transform_test = transforms.Compose([
        transforms.Resize((cfg.DATA.crop_size, cfg.DATA.crop_size)),
        transforms.ToTensor(),
    ])

    if cfg.dataset == 'cifar10':
        train_dataset_1 = torchvision.datasets.CIFAR10(os.path.join(cfg.root_path, cfg.DATA.data_path), train=True,
                                                       transform=MultiDataTransform(transform_train_1), download=True)
        train_dataset_2 = torchvision.datasets.CIFAR10(os.path.join(cfg.root_path, cfg.DATA.data_path), train=True,
                                                       transform=MultiDataTransform(transform_train_2), download=True)
        valid_dataset = torchvision.datasets.CIFAR10(os.path.join(cfg.root_path, cfg.DATA.data_path), train=True,
                                                     transform=transform_test, download=True)

        train_dataloader_1 = torch.utils.data.DataLoader(train_dataset_1, batch_size=cfg.DATA.batch_size, shuffle=True,
                                                         num_workers=cfg.DATA.workers, pin_memory=True)
        train_dataloader_2 = torch.utils.data.DataLoader(train_dataset_2, batch_size=cfg.DATA.batch_size, shuffle=True,
                                                         num_workers=cfg.DATA.workers, pin_memory=True)
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=cfg.DATA.batch_size, shuffle=False,
                                                       num_workers=cfg.DATA.workers, pin_memory=True)
    elif cfg.dataset == 'cifar100':
        train_dataset_1 = torchvision.datasets.CIFAR100(os.path.join(cfg.root_path, cfg.DATA.data_path), train=True,
                                                        transform=MultiDataTransform(transform_train_1), download=True)
        train_dataset_2 = torchvision.datasets.CIFAR100(os.path.join(cfg.root_path, cfg.DATA.data_path), train=True,
                                                        transform=MultiDataTransform(transform_train_2), download=True)
        valid_dataset = torchvision.datasets.CIFAR100(os.path.join(cfg.root_path, cfg.DATA.data_path), train=True,
                                                      transform=transform_test, download=True)

        train_dataloader_1 = torch.utils.data.DataLoader(train_dataset_1, batch_size=cfg.DATA.batch_size, shuffle=True,
                                                         num_workers=cfg.DATA.workers, pin_memory=True)
        train_dataloader_2 = torch.utils.data.DataLoader(train_dataset_2, batch_size=cfg.DATA.batch_size, shuffle=True,
                                                         num_workers=cfg.DATA.workers, pin_memory=True)
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=cfg.DATA.batch_size, shuffle=False,
                                                       num_workers=cfg.DATA.workers, pin_memory=True)
    else:
        raise NotImplemented

    return [train_dataloader_1, train_dataloader_2], valid_dataloader

