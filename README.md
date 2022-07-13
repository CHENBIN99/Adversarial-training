# SOTA_AT
 Implement of some SOTA method of adversarial training



## Usage

The option for the training method is as follow:

> + <at_method> : {standard, trades, mart, ccg}
> + <dataset> : {cifar10, cifar100, tinyimagent}
> + <model> : {wrn34-10, resnet18, resnet50, preactresnet18}



### Training Scripts

+ Standard Adversarial Training

  > \# standard adversarial training on cifar10 using Wide-ResNet50
  >
  > `python main.py --at_method standard --model_name wrn34-10 --dataset cifar10 --tensorboard`

+ Trades

  paper: https://arxiv.org/abs/1901.08573

  > \# Trades with beta 6.0
  >
  > `python main.py --at_method trades --beta 6.0 --model_name wrn34-10 --dataset cifar10 --tensorboard`

+ Mart

  > \# Mart with beta 6.0
  >
  > `python main.py --at_method mart --beta 6.0 --model_name wrn34-10 --dataset cifar10 --tensorboard`
