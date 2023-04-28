#!/bin/bash

echo "use gpu $1"

python train.py -c config/standard/standard_cifar10_resnet18.yml --dataset cifar10 --method standard --use_log --gpu_id $1

python train.py -c config/standard/standard_cifar10_resnet50.yml --dataset cifar10 --method standard --use_log --gpu_id $1

python train.py -c config/standard/standard_cifar10_wrn3410.yml --dataset cifar10 --method standard --use_log --gpu_id $1

python train.py -c config/standard/standard_cifar10_incv3.yml --dataset cifar10 --method standard --use_log --gpu_id $1
