import torch
import timm
from model import wideresnet, preactresnet, resnet


def get_model(model_name, num_classes, dataset, input_size, device, pretrain=False, compile=False):
    if model_name == 'wrn3410':
        model = wideresnet.WideResNet(depth=34, widen_factor=10, num_classes=num_classes, dropRate=0.0,
                                      stride=1 if dataset != 'tinyimagenet' else 2)
        model.to(device)
    elif model_name == 'preactresnet18':
        model = preactresnet.PreActResNet18(num_classes=num_classes, stride=1 if dataset != 'tinyimagenet' else 2)
        model.to(device)
    elif model_name == 'resnet18' and dataset in ['cifar10', 'cifar100'] and input_size == 32:
        model = resnet.ResNet18()
        model.to(device)
    else:
        model = timm.create_model(model_name, num_classes=num_classes, pretrained=pretrain)
        model.to(device)

    if compile:
        model = torch.compile(model)

    return model

def get_static_model(static_model_id, num_class, device):
    if static_model_id == 1:
        static_model = [
            timm.create_model('inception_v3', pretrained=True).to(device),
            timm.create_model('resnetv2_50', pretrained=True).to(device),
        ]
    elif static_model_id == 2:
        static_model = [
            timm.create_model('inception_v3', pretrained=True).to(device),
            timm.create_model('resnetv2_50', pretrained=True).to(device),
            timm.create_model('inception_resnet_v2', pretrained=True).to(device),
        ]
    elif static_model_id == 3:
        static_model = [
            timm.create_model('inception_v3', pretrained=True).to(device),
            timm.create_model('inception_resnet_v2', pretrained=True).to(device),
        ]
    elif static_model_id == 4:
        static_model = [
            timm.create_model('inception_v3', num_classes=num_class,
                              checkpoint_path='./static_checkpoint/cifar10/inception_v3.pth.tar').to(device),
            timm.create_model('vit_tiny_patch16_224', num_classes=num_class,
                              checkpoint_path='./static_checkpoint/cifar10/vit_tiny.pth.tar').to(device),
        ]
    elif static_model_id == 5:
        static_model = [
            timm.create_model('resnet18', num_classes=num_class,
                              checkpoint_path='./static_checkpoint/cifar10/resnet18.pth.tar').to(device),
            timm.create_model('inception_v3', num_classes=num_class,
                              checkpoint_path='./static_checkpoint/cifar10/inception_v3.pth.tar').to(device),
            timm.create_model('vit_tiny_patch16_224', num_classes=num_class,
                              checkpoint_path='./static_checkpoint/cifar10/vit_tiny.pth.tar').to(device),
        ]
    elif static_model_id == 6:
        static_model = [
            timm.create_model('inception_v3', num_classes=num_class
                              , checkpoint_path='./static_checkpoint/cifar10/inception_v3.pth.tar').to(device),
            timm.create_model('deit_tiny_patch16_224', num_classes=num_class,
                              checkpoint_path='./static_checkpoint/cifar10/deit_tiny.pth.tar').to(device),
        ]
    else:
        raise NotImplemented

    return static_model

