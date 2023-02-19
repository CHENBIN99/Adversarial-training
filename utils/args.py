import argparse


def parser():
    parser = argparse.ArgumentParser(description='AT')
    parser.add_argument('--seed', default=1,
                        help='random seed')
    parser.add_argument('--data_root', default='data',
                        help='the directory to save the dataset')
    parser.add_argument('--adv_train', action='store_true')
    parser.add_argument('--save_root', default='runs',
                        help='the directory to save the logs or other imformations (e.g. images)')
    parser.add_argument('--exp_series', type=str, default='exp_default',
                        help='the series of the exp')
    parser.add_argument('--model_root', default='checkpoint', help='the directory to save the models')
    parser.add_argument('--load_checkpoint', default='./model/default/model.pth')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Log progress to TensorBoard')

    # parameters for generating adversarial examples
    parser.add_argument('--attack_method', type=str, default='pgd',
                        choices=['pgd', 'fgsm'])
    parser.add_argument('--epsilon', '-e', type=float, default=8,
                        help='maximum perturbation of adversaries (4/255=0.0157)')
    parser.add_argument('--alpha', '-a', type=float, default=2,
                        help='movement multiplier per iteration when generating adversarial examples (2/255=0.00784)')
    parser.add_argument('--iters', type=int, default=10,
                        help='maximum iteration when generating adversarial examples')
    parser.add_argument('--iters_eval', type=int, default=20,
                        help='maximum iteration when generating adversarial examples')
    parser.add_argument('--max_image', type=float, default=1., help='max value of the image')
    parser.add_argument('--min_image', type=float, default=0., help='min value of the image')

    # training
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='batch size')
    parser.add_argument('--max_epoch', '-m_e', type=int, default=100,
                        help='the maximum numbers of the model see a sample')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='momentum for optimizer')
    parser.add_argument('--weight_decay', '-w', type=float, default=2e-4,
                        help='the parameter of l2 restriction for weights')
    parser.add_argument('--dataset', type=str, default='cifar10', help='training dataset')
    parser.add_argument('--image_size', type=int, default=-1, help='resize the image in dataset')
    parser.add_argument('--max_epochs', type=int, default=100, help='total epochs need to run')
    parser.add_argument('--num_works', type=int, default=4, help='numbers of the workers')
    parser.add_argument('--n_eval_step', type=int, default=10,
                        help='number of iteration per one evaluation')
    parser.add_argument('--n_checkpoint_step', type=int, default=-1,
                        help='number of iteration to save a checkpoint')
    parser.add_argument('--n_store_image_step', type=int, default=4000,
                        help='number of iteration to save adversaries')
    parser.add_argument('--perturbation_type', '-p', choices=['linf', 'l2'], default='linf',
                        help='the type of the perturbation (linf or l2)')
    parser.add_argument('--gpu_id', '-g', default='0', help='which gpu to use')
    parser.add_argument('--multi-gpu', action="store_true",
                        help='use if machine have muti-gpu')
    parser.add_argument('--ms_1', type=float, default=0.5,
                        help='mile stone 1 of learning schedule')
    parser.add_argument('--ms_2', type=float, default=0.75,
                        help='mile stone 2 of learning schedule')
    parser.add_argument('--ms_3', type=float, default=0.9,
                        help='mile stone 3 of learning schedule')
    parser.add_argument('--log_time', action='store_true')

    # model
    parser.add_argument('--model_name', type=str, default='wrn34-10')
    parser.add_argument('--depth', type=int, default=34)
    parser.add_argument('--widen_factor', type=int, default=10)
    parser.add_argument('--num_classes', type=int, default=10)

    # free
    # default: m=8 for cifar10 and m=4 for imagenet
    parser.add_argument('--m', type=int, default=4,
                        help='the hyper-parameter of the free-at')

    # ens
    parser.add_argument('--static_model', type=int, default=1)

    # TRADES
    parser.add_argument('--trades_beta', type=float, default=6.0)

    # MART
    parser.add_argument('--mart_beta', type=float, default=6.0)

    # CCG
    parser.add_argument('--lamda', type=float, default=1.0)
    parser.add_argument('--T', type=float, default=1.)

    # Choose at method
    parser.add_argument('--at_method', type=str, default='standard')

    return parser.parse_args()

def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.info('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))


