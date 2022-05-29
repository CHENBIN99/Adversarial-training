import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import copy

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from torchvision import datasets, transforms
from tqdm import tqdm

import torchattacks
import numpy as np
import argparse


def parser():
    parser = argparse.ArgumentParser(description='AT')
    parser.add_argument('--attack_method', required=True)
    parser.add_argument('--model_name', type=str, default='wideresnet')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], required=True)
    parser.add_argument('--batch_size', default=128)
    return parser.parse_args()


def evaluate(_input, _target, method='mean'):
    correct = (_input == _target).astype(np.float32)
    if method == 'mean':
        return correct.mean()
    else:
        return correct.sum()


def valid(args, model, valid_loader, adv_test=False, use_pseudo_label=False):
    total_acc = 0.
    num = 0
    total_adv_acc = 0.

    if args.attack_method == 'pgd':
        attack_method = torchattacks.PGD(model=model, eps=8 / 255, alpha=2 / 255, steps=20, random_start=True)
    elif args.attack_method == 'aa':
        attack_method = torchattacks.AutoAttack(model, eps=8/255, n_classes=10 if args.dataset == 'cifar10' else 100)

    with torch.no_grad():
        for idx, (data, label) in enumerate(tqdm(valid_loader)):
            data, label = data.to(device), label.to(device)

            # output = t(f(data))
            output = model(data)

            pred = torch.max(output, dim=1)[1]
            std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy(), 'sum')

            total_acc += std_acc
            num += output.shape[0]

            if adv_test:
                with torch.enable_grad():
                    adv_data = attack_method(data,
                                             pred if use_pseudo_label else label)
                # adv_output = t(f(adv_data))
                adv_output = model(adv_data)

                adv_pred = torch.max(adv_output, dim=1)[1]
                adv_acc = evaluate(adv_pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                total_adv_acc += adv_acc
            else:
                total_adv_acc = -num

    return total_acc / num, total_adv_acc / num


if __name__ == '__main__':
    args = parser()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    if args.dataset == 'cifar10':
        test_set = datasets.CIFAR10(root='./data/', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    elif args.dataset == 'cifar100':
        test_set = datasets.CIFAR100(root='./data/', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # load model
    if args.model_name == 'wideresnet':
        from model.wideresnet import WideResNet
        model = WideResNet(depth=34, widen_factor=10, num_classes=10 if args.dataset == 'cifar10' else 100)
        model.load_state_dict(torch.load(args.model_path, map_location='cuda:0'))
        model.to(device)
        model.eval()
    elif args.model_name == 'preact':
        from model.preactresnet import PreActResNet18
        model = PreActResNet18()
        model.load_state_dict(torch.load(args.model_path, map_location='cuda:0'))
        model.to(device)
        model.eval()

    # loss function
    loss_func = nn.CrossEntropyLoss()

    clean_acc, adv_acc = valid(args, model, test_loader, True)

    inf = f'RESULT:\n' \
          f'clean acc: {clean_acc}\n' \
          f'{args.attack_method} acc:   {adv_acc}\n'

    print(inf)




