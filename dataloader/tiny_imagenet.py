import os
import sys
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from torchvision import transforms


def download_tinyimagenet(args):
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    os.system(f'wget -P {os.path.join(args.root_path, args.data_root)} {url}')
    os.system(f'unzip {os.path.join(args.root_path, args.data_root, "tiny-imagenet-200.zip")}'
              f' -d '
              f'{os.path.join(args.root_path, args.data_root)}')


class data(Dataset):
    def __init__(self, args, type, transform, labels_t, image_names, val_names, val_labels):
        self.type = type
        if type == 'train':
            i = 0
            self.images = []
            for label in labels_t:
                image = []
                for image_name in image_names[i]:
                    image_path = os.path.join(args.root_path, 'data/tiny-imagenet-200/train', label, 'images', image_name)
                    image.append(cv2.imread(image_path))
                self.images.append(image)
                i = i + 1
            self.images = np.array(self.images)
            self.images = self.images.reshape(-1, 64, 64, 3)
        elif type == 'val':
            self.val_images = []
            for val_image in val_names:
                val_image_path = os.path.join(args.root_path, 'data/tiny-imagenet-200/val/images', val_image)
                self.val_images.append(cv2.imread(val_image_path))
            self.val_images = np.array(self.val_images)
        self.transform = transform
        self.val_labels = val_labels

    def __getitem__(self, index):
        label = []
        image = []
        if self.type == 'train':
            label = index // 500
            image = self.images[index]
        if self.type == 'val':
            label = self.val_labels[index]
            image = self.val_images[index]
        return self.transform(image), label

    def __len__(self):
        len = 0
        if self.type == 'train':
            len = self.images.shape[0]
        if self.type == 'val':
            len = self.val_images.shape[0]
        return len


def get_tiny(args):
    labels_t = []
    image_names = []
    with open(os.path.join(args.root_path, 'data/tiny-imagenet-200/wnids.txt')) as wnid:
        for line in wnid:
            labels_t.append(line.strip('\n'))
    for label in labels_t:
        txt_path = os.path.join(args.root_path, 'data/tiny-imagenet-200/train/', label, f'{label}_boxes.txt')
        image_name = []
        with open(txt_path) as txt:
            for line in txt:
                image_name.append(line.strip('\n').split('\t')[0])
        image_names.append(image_name)
    labels = np.arange(200)
    val_labels_t = []
    val_labels = []
    val_names = []
    with open(os.path.join(args.root_path, 'data/tiny-imagenet-200/val/val_annotations.txt')) as txt:
        for line in txt:
            val_names.append(line.strip('\n').split('\t')[0])
            val_labels_t.append(line.strip('\n').split('\t')[1])
    for i in range(len(val_labels_t)):
        for i_t in range(len(labels_t)):
            if val_labels_t[i] == labels_t[i_t]:
                val_labels.append(i_t)
    val_labels = np.array(val_labels)

    transform_train = transforms.Compose([transforms.ToPILImage(),
                                          transforms.RandomCrop(64, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.Resize(224),
                                          transforms.ToTensor()
    ])

    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize(224)])

    train_dataset = data(args, 'train', transform=transform_train, labels_t=labels_t, image_names=image_names,
                         val_names=val_names, val_labels=val_labels)
    val_dataset = data(args, 'val', transform=transform_test, labels_t=labels_t, image_names=image_names,
                       val_names=val_names, val_labels=val_labels)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                  num_workers=args.num_worker)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                num_workers=args.num_worker)
    return train_dataloader, val_dataloader

