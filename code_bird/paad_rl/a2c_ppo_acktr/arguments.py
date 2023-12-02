import argparse
import sys
import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--algo', default='a2c', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument(
        '--gail',
        action='store_true',
        default=False,
        help='do imitation learning with gail')
    parser.add_argument(
        '--gail-experts-dir',
        default='./gail_experts',
        help='directory that contains expert demonstrations for gail')
    parser.add_argument(
        '--gail-batch-size',
        type=int,
        default=128,
        help='gail batch size (default: 128)')
    parser.add_argument(
        '--gail-epoch', type=int, default=5, help='gail epochs (default: 5)')
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=16,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=5,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=32,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=10e6,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--env-name',
        default='LunarLander-v2',
        help='environment to train on (default: LunarLander-v2)')
    parser.add_argument(
        '--log-dir',
        default='./data/log/',
        help='directory to save agent logs (default: ./data/log/)')
    parser.add_argument(
        '--save-dir',
        default='./learned_models/',
        help='directory to save agent models (default: ./learned_models/)')
    parser.add_argument(
        '--victim-dir',
        default='./released_models/a2c_victim/',
        help='directory to save agent models (default: ./released_models/a2c_victim/)')
    parser.add_argument(
        '--clean-dir',
        default='./released_models/a2c_victim/',
        help='directory to clean models')
    parser.add_argument(
        '--adv-dir',
        default='./learned_adv/',
        help='directory to save adversary models (default: ./learned_adv/)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')
    parser.add_argument(
        '--imitate',
        action='store_true',
        default=False,
        help='Start with Imitation Learning')
    parser.add_argument(
        '--imitate-steps',
        type=int,
        default=int(1e+6))
    parser.add_argument(
        '--collect-data',
        action='store_true',
        default=False,
        help='collect expert data during robust attack model attacks')
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=False,
        help='stop printing information')
    parser.add_argument(
        '--cuda-id',
        type=int,
        default=0)
    parser.add_argument(
        '--test-episodes',
        type=int,
        default=1000,
        help='number of episodes to test return (default: 1000)')
    parser.add_argument(
        '--attacker',
        type=str,
        default=None,
        help='the attacker algorithm (default: None)')
    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.01,
        help='the attack budget')
    parser.add_argument(
        '--rs-eps',
        type=float,
        default=0.02,
        help='the epsilon for training robust sarsa')
    parser.add_argument(
        '--attack-lr',
        type=float,
        default=0.01,
        help='PGD attack learning rate')
    parser.add_argument(
        '--beta-pa',
        type=float,
        default=10,
        help="Beta value used in pa attacker"
    )
    parser.add_argument(
        '--attack-steps',
        type=int,
        default=10,
        help='PGD attack learning steps')
    parser.add_argument(
        '--rand-init',
        action='store_true',
        default=False,
        help='whether to use a random initialization for pgd attacks')
    parser.add_argument(
        '--res-dir',
        default='./data/a2c_results/',
        help='directory to save agent rewards (default: ./data/a2c_results/)')
    parser.add_argument(
        '--load',
        action='store_true',
        default=False,
        help='load pretrained model')
    parser.add_argument(
        '--train-nn',
        action='store_true',
        default=False,
        help='train obs attack nn')
    parser.add_argument(
        '--fgsm',
        action='store_true',
        default=False,
        help='whether to use fgsm')
    parser.add_argument(
        '--momentum',
        action='store_true',
        default=False,
        help='whether to use momentum fgm')
    parser.add_argument(
        '--train-freq',
        type=int,
        default=1)
    parser.add_argument(
        '--no-attack',
        action='store_true',
        default=False)
    parser.add_argument(
        '--det',
        action='store_true',
        default=False,
        help='whether to use deterministic policy')
    parser.add_argument(
        '--v-det',
        action='store_true',
        default=False,
        help='whether victim uses deterministic policy')
    parser.add_argument(
        '--norm-env',
        action='store_true',
        default=False,
        help='whether normalize environment')
    parser.add_argument(
        '--use-nn',
        action='store_true',
        default=False,
        help='whether to use neural network observation attacker for pa attacks')
    parser.add_argument('--nn-hiddens', nargs='+', type=int)
    parser.add_argument(
        '--v-algo', 
        default='a2c', 
        help='algorithm to attack: a2c | ppo | acktr')
    parser.add_argument(
        '--plot',
        action='store_true',
        default=False)
    parser.add_argument(
        '--test',
        action='store_true',
        default=False)
    parser.add_argument(
        '--epoch_smoothed',
        type=int,
        default=10,
        help='rewards smoothed over epochs for plotting')
    parser.add_argument(
        '--reward-bonus',
        type=float,
        default=0,
        help='rewards added to a misclassification')
    parser.add_argument('--rew-weights', nargs='+', type=float, 
                        help='weights for the original reward of action, mask size reward, action deviation rew and mask similarity rew')
    parser.add_argument('--mask-interval', type=int, default=1000,
                        help='save mask interval, one save per n updates (default: 1000)')
    parser.add_argument('--reg_l1', type=float, default=0.01,
                        help='coefficient for L1 regularization of mask size')
    parser.add_argument('--reg_l2', type=float, default=0.0,
                        help='coefficient for L2 regularization of mask size')
    parser.add_argument('--smooth', type=float, default=0.0,
                        help='coefficient for smoothness regularization of mask size')
    parser.add_argument('--beta', action='store_true', default=False, 
                        help='use beta distribution to solve mask')
    parser.add_argument('--trigger-path', default='trigger.png',
                        help='path to the trigger image')                    
    parser.add_argument('--stop-rew', type=float, default=1000000,
                        help='weight of original agent reward')
    parser.add_argument('--patience', type=float, default=50,
                        help='patience for counting down the weights of mask size regularization')
    parser.add_argument('--init-alpha', type=float, default=50,
                        help='initial value of alpha')
    parser.add_argument('--init-delta', type=float, default=50,
                        help='initial value of delta')
    parser.add_argument('--mask-bound', type=float, default=200,
                        help='threshold for mask size during optimization')
    parser.add_argument('--mask-lowbound', type=float, default=400,
                        help='threshold for lower mask size during optimization')
    parser.add_argument('--max-adv', action='store_true', default=False, 
                        help='whether we maximize the advantage of victim or not')
    parser.add_argument('--use-value', action='store_true', default=False, 
                        help='whether we maximize the value of victim or not')
    parser.add_argument('--use-return', action='store_true', default=False, 
                        help='whether we maximize the return of victim or not')
    parser.add_argument('--kl', action='store_true', default=False, 
                        help='whether add kl divergence between retrained and clean policy')

    return parser

def add_trojan_args(parser):
    parser.add_argument('--poison', help="poison the training data", dest='poison', action="store_true")
    parser.add_argument('--color', default=100, required='--poison' in sys.argv, type=int,
                        help="color of the poisoned pixels")
    parser.add_argument('--start_position', default="0, 0", required='--poison' in sys.argv,
                        help='delimited input of x, y where the poisoning will start',
                        type=lambda s: [int(el) for el in s.split(',')])
    parser.add_argument('--pixels_to_poison_h', required='--poison' in sys.argv, default=3, type=int,
                        help="Number of pixels to be poisoned horizontally")
    parser.add_argument('--pixels_to_poison_v', required='--poison' in sys.argv, default=3, type=int,
                        help="Number of pixels to be poisoned vertically")
    parser.add_argument('--attack_method', required='--poison' in sys.argv, default='strong_targeted', type=str,
                        choices=['strong_targeted', 'weak_targeted', 'untargeted'],
                        help="which method will be used to attack")
    parser.add_argument('--action', required='--poison' in sys.argv, default=2, type=int,
                        help="specify the target action for targeted attacks")
    parser.add_argument('--budget', required='--poison' in sys.argv, default=20000, type=int,
                        help="how many states/actions/rewards will be poisoned")
    parser.add_argument('--when_to_poison', required='--poison' in sys.argv, default="uniformly", type=str,
                        choices=['uniformly', 'first', 'middle', 'last'],
                        help="Number of pixels to be poisoned vertically")
    parser.add_argument('--max_global_steps', default=80000000, type=int, help="Max. number of training steps")
    parser.set_defaults(poison=False)

    return parser