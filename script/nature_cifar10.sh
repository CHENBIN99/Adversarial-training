#!/bin/bash

echo "use gpu $1"

python train.py -c config/nature_cifar10_resnet18.yml --dataset cifar10 --method nature --use_log --gpu_id $1

python train.py -c config/nature_cifar10_resnet34.yml --dataset cifar10 --method nature --use_log --gpu_id $1

python train.py -c config/nature_cifar10_resnet50.yml --dataset cifar10 --method nature --use_log --gpu_id $1

python train.py -c config/nature_cifar10_wrn16_1.yml --dataset cifar10 --method nature --use_log --gpu_id $1

python train.py -c config/nature_cifar10_wrn40_1.yml --dataset cifar10 --method nature --use_log --gpu_id $1

python train.py -c config/nature_cifar10_wrn3410.yml --dataset cifar10 --method nature --use_log --gpu_id $1
