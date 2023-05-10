# 🛡️Adversarial Training

 Implement of some SOTA method of adversarial training



## File Structure

> . \
> ├── adv_lib  
> ├── checkpoint  
> ├── config  
> │   ├── ccg  
> │   ├── free-at  
> │   ├── mart  
> │   ├── nature  
> │   ├── standard  
> │   └── trades  
> ├── data  
> ├── dataloader  
> ├── log  
> ├── model  
> │   ├── classifiers  
> ├── script  
> ├── static_checkpoint  
> ├── train  
> └── utils  


## Usage

### Toy example

`python train.py -c config/standard/standard_cifar10_resnet18.yml --dataset cifar10 --method standard --use_log`

 This command above will run standard adversarial training on CIFAR-10 using ResNet-18

### Args

`-c` : path of the config file

`--dataset`: Training dataset, e.g. cifar10/cifar100/Imagenet

`-m`: method of adversarial training, e.g. standard/trades/mart

`-e`: evaluate model on validation set

`--pretrained`: use pretrained model

`--use_log`: use Tensorboard to log train data

`--gpu_id`: gpu id used for training



### More examples

train wrn34-10 on cifar-10 using trades

`python train.py -c config/trades/trades_cifar10_wrn3410.yml --dataset cifar10 --method trades --use_log`

train wrn34-10 on cifar-10 using mart

`python train.py -c config/mart/mart_cifar10_wrn3410.yml --dataset cifar10 --method mart --use_log`

......



The config folder already has a number of configuration files that you can use to complete specific training sessions, or you can create a new config to complete your own custom training sessions.



## Results

release soon



## Mile stone
- [ ] release training results
- [x] amp

