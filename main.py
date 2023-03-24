import sys
import os

import timm
import torch.nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from tensorboardX import SummaryWriter


from model import wideresnet, preactresnet
from utils.args import parser
from utils.utils import *
from dataloader import get_dataloader


def main(args):
    same_seeds(args.seed)
    project_path = get_project_path()
    setattr(args, 'root_path', project_path)
    setattr(args, 'epsilon', args.epsilon / 255.0)
    setattr(args, 'alpha', args.alpha / 255.0)

    exp_name = get_exp_name(args)
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
    elif args.dataset == 'imagenet':
        num_classes = 1000
    else:
        raise 'no match dataset'

    train_dataloader, valid_dataloader = get_dataloader.get_dataloader(args)

    if args.model_name == 'wrn3410':
        model = wideresnet.WideResNet(depth=34,
                                      widen_factor=10,
                                      num_classes=num_classes,
                                      dropRate=0.0,
                                      stride=1 if args.dataset != 'tinyimagenet' else 2).to(device)
    elif args.model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(512, num_classes)
        model.to(device)
    elif args.model_name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(2048, num_classes)
        model.to(device)
    elif args.model_name == 'resnet101':
        model = torchvision.models.resnet101(pretrained=False)
        model.fc = torch.nn.Linear(2048, num_classes)
        model.to(device)
    elif args.model_name == 'preactresnet18':
        model = preactresnet.PreActResNet18(num_classes=num_classes, stride=1 if args.dataset != 'tinyimagenet' else 2)
        model.to(device)
    elif args.model_name == 'resnet18_224':
        model = timm.create_model('resnet18', pretrained=False)
        model.to(device)
    elif args.model_name == 'efficientnet':
        model = timm.create_model('efficientnet_b0', pretrained=False)
        model.to(device)
    elif args.model_name == 'incv3':
        model = timm.create_model('inception_v3', num_classes=num_classes, pretrained=False)
        model.to(device)
    elif args.model_name == 'incv2':
        model = timm.create_model('inception_resnet_v2', num_classes=num_classes, pretrained=False)
        model.to(device)
    elif args.model_name == 'swin_t':
        model = timm.create_model('swin_tiny_patch4')
    else:
        raise 'no match model'

    if args.at_method == 'at_ens':
        if args.static_model == 1:
            static_model = [
                timm.create_model('inception_v3', pretrained=True).to(device),
                timm.create_model('resnetv2_50', pretrained=True).to(device),
            ]
        elif args.static_model == 2:
            static_model = [
                timm.create_model('inception_v3', pretrained=True).to(device),
                timm.create_model('resnetv2_50', pretrained=True).to(device),
                timm.create_model('inception_resnet_v2', pretrained=True).to(device),
            ]
        elif args.static_model == 3:
            static_model = [
                timm.create_model('inception_v3', pretrained=True).to(device),
                timm.create_model('inception_resnet_v2', pretrained=True).to(device),
            ]
        elif args.static_model == 4:
            static_model = [
                timm.create_model('inception_v3', num_classes=num_classes,
                                  checkpoint_path='./static_checkpoint/cifar10/inception_v3.pth.tar').to(device),
                timm.create_model('vit_tiny_patch16_224', num_classes=num_classes,
                                  checkpoint_path='./static_checkpoint/cifar10/vit_tiny.pth.tar').to(device),
            ]
        elif args.static_model == 5:
            static_model = [
                timm.create_model('resnet18', num_classes=num_classes,
                                  checkpoint_path='./static_checkpoint/cifar10/resnet18.pth.tar').to(device),
                timm.create_model('inception_v3', num_classes=num_classes,
                                  checkpoint_path='./static_checkpoint/cifar10/inception_v3.pth.tar').to(device),
                timm.create_model('vit_tiny_patch16_224', num_classes=num_classes,
                                  checkpoint_path='./static_checkpoint/cifar10/vit_tiny.pth.tar').to(device),
            ]
        elif args.static_model == 6:
            static_model = [
                timm.create_model('inception_v3', num_classes=num_classes
                                  , checkpoint_path='./static_checkpoint/cifar10/inception_v3.pth.tar').to(device),
                timm.create_model('deit_tiny_patch16_224', num_classes=num_classes,
                                  checkpoint_path='./static_checkpoint/cifar10/deit_tiny.pth.tar').to(device),
            ]
        else:
            raise NotImplemented

    # choose adversarial training method
    if args.at_method == 'nature':
        from train.train_nature import Trainer_Nature
        trainer = Trainer_Nature(args, tb_writer, args.attack_method, device)
    elif args.at_method == 'standard':
        from train.train_standard import Trainer_Standard
        trainer = Trainer_Standard(args, tb_writer, args.attack_method, device)
    elif args.at_method == 'at_free':
        from train.train_at_free import Trainer_Free
        trainer = Trainer_Free(args, tb_writer, args.attack_method, device, m=args.m)
    elif args.at_method == 'at_fast':
        from train.train_fast_at import Trainer_Fast
        trainer = Trainer_Fast(args, tb_writer, args.attack_method, device, m=1, random_init=True)
    elif args.at_method == 'at_ens':
        from train.train_ens_adv import Trainer_Ens
        trainer = Trainer_Ens(args, tb_writer, args.attack_method, device)
    elif args.at_method == 'trades':
        from train.train_trades import Trainer_Trades
        trainer = Trainer_Trades(args, tb_writer, args.attack_method, device)
    elif args.at_method == 'mart':
        from train.train_mart import Trainer_Mart
        trainer = Trainer_Mart(args, tb_writer, args.attack_method, device)
    elif args.at_method == 'mart_trades':
        from train.train_mart_trades import Trainer_Mart_Trades
        trainer = Trainer_Mart_Trades(args, tb_writer, args.attack_method, device)
    elif args.at_method == 'ccg':
        from train.train_ccg import Trainer_CCG
        trainer = Trainer_CCG(args, tb_writer, args.attack_method, device)
    elif args.at_method == 'ccg_trades':
        from train.train_ccg_trades import Trainer_CCG_TRADES
        trainer = Trainer_CCG_TRADES(args, tb_writer, args.attack_method, device)
    else:
        raise 'no match at_method'

    if args.at_method != 'at_ens':
        trainer.train(model, train_dataloader, valid_dataloader, args.adv_train)
    else:
        trainer.train(model, static_model, train_dataloader, valid_dataloader, args.adv_train)

    print('Train Finished!')


if __name__ == '__main__':
    args = parser()
    main(args)

