import torch

from model import classifiers
import timm


def get_model(model_name, num_classes, dataset_name, pre_train=False, compile=False):
    if dataset_name in ['cifar10', 'cifar100']:
        if model_name == 'resnet18':
            model = classifiers.resnet.resnet18(num_classes=num_classes)
        elif model_name == 'resnet34':
            model = classifiers.resnet.resnet34(num_classes=num_classes)
        elif model_name == 'resnet50':
            model = classifiers.resnet.resnet50(num_classes=num_classes)
        elif model_name == 'resnet101':
            model = classifiers.resnet.resnet101(num_classes=num_classes)
        elif model_name == 'resnet152':
            model = classifiers.resnet.resnet152(num_classes=num_classes)
        elif model_name == 'preactresnet18':
            model = classifiers.preresnet.PreActResNet18(num_classes=num_classes)
        elif model_name == 'preactresnet34':
            model = classifiers.preresnet.PreActResNet34(num_classes=num_classes)
        elif model_name == 'preactresnet50':
            model = classifiers.preresnet.PreActResNet50(num_classes=num_classes)
        elif model_name == 'preactresnet101':
            model = classifiers.preresnet.PreActResNet50(num_classes=num_classes)
        elif model_name == 'wrn16_1':
            model = classifiers.wresnet.wrn_16_1(num_classes=num_classes)
        elif model_name == 'wrn16_2':
            model = classifiers.wresnet.wrn_16_2(num_classes=num_classes)
        elif model_name == 'wrn40_1':
            model = classifiers.wresnet.wrn_40_1(num_classes=num_classes)
        elif model_name == 'wrn40_2':
            model = classifiers.wresnet.wrn_40_2(num_classes=num_classes)
        elif model_name == 'wrn34_10':
            model = classifiers.wresnet.wrn_34_10(num_classes=num_classes)
        elif model_name == 'inc_v3':
            model = classifiers.inception_v3.inception_v3_cifar(num_classes=num_classes)
        else:
            raise NotImplemented
    elif dataset_name in ['imagenet']:
        model = timm.create_model(model_name, pretrained=pre_train, num_classes=1000)
    else:
        raise NotImplemented

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

