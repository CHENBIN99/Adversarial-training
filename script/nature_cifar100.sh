#!/bin/bash

echo "use gpu $1"

python train.py -c config/nature/nature_cifar100_resnet18.yml --dataset cifar100 --method nature --use_log --gpu_id $1

python train.py -c config/nature/nature_cifar100_resnet34.yml --dataset cifar100 --method nature --use_log --gpu_id $1

python train.py -c config/nature/nature_cifar100_resnet50.yml --dataset cifar100 --method nature --use_log --gpu_id $1

python train.py -c config/nature/nature_cifar100_wrn16_1.yml --dataset cifar100 --method nature --use_log --gpu_id $1

python train.py -c config/nature/nature_cifar100_wrn40_1.yml --dataset cifar100 --method nature --use_log --gpu_id $1

python train.py -c config/nature/nature_cifar100_wrn3410.yml --dataset cifar100 --method nature --use_log --gpu_id $1

python train.py -c config/nature/nature_cifar100_incv3.yml --dataset cifar100 --method nature --use_log --gpu_id $1