import argparse
from collections import defaultdict
import copy
import gym
import operator
import os
import sys
import pickle
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.agent import Agent
from core.rl_algs import PPO
from core.experiment import Experiment
from infra.log import create_logger, visualize_parameters
from infra.env_config import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from models.primitives import Feedforward, CompositePolicy, WeightNetwork, PrimitivePolicy, CompositeTransferPolicy, GoalEmbedder, LatentPolicy, LatentTransferPolicy
from utils import *

parser = argparse.ArgumentParser(description='PyTorch PPO example')
parser.add_argument('--env-name', default="Ant-v3", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--env-type', default="vel",
                    help='vel | goal; n-: normalized; m-: multitask')
parser.add_argument('--model-path', metavar='G',
                    help='path of pre-trained model')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--log-std', type=float, default=-0.0, metavar='G',
                    help='log std for the policy (default: -0.0)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--plr', type=float, default=1e-5, metavar='G',
                    help='policy learning rate (default: 1e-5)')
parser.add_argument('--clr', type=float, default=1e-4, metavar='G',
                    help='critic learning rate (default: 1e-4)')
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                    help='clipping epsilon for PPO')
parser.add_argument('--num-threads', type=int, default=1, metavar='N',
                    help='number of threads for agent (default: 1)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=4096, metavar='N',
                    help='minimal batch size per PPO update (default: 4096)')
parser.add_argument('--max-iter-num', type=int, default=5000, metavar='N',
                    help='maximal number of main iterations (default: 5000)')
parser.add_argument('--log-every', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 1)')
parser.add_argument('--save-every', type=int, default=100, metavar='N',
                    help='interval between saving (default: 100)')
parser.add_argument('--visualize-every', type=int, default=500, metavar='N',
                    help='interval between visualizing (default: 500)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')

parser.add_argument('--klp', type=float, default=0.0002,
                    help='KL for info bottlneeck for primitive')
parser.add_argument('--klw', type=float, default=0.0004,
                    help='KL for info bottlneeck for weight')
parser.add_argument('--entropy-coeff', type=float, default=0.005,
                    help='KL for info bottlneeck for weight')
parser.add_argument('--weight-entropy-coeff', type=float, default=0.001,
                    help='KL for info bottlneeck for weight')

parser.add_argument('--maxeplen', type=int, default=200, metavar='N',
                    help='maximal number of main iterations (default: 10000)')
parser.add_argument('--num-test', type=int, default=100,
                    help='number of test trajectories (default: 100)')
parser.add_argument('--resume', type=str, default='',
                    help='.tar path of saved model')
parser.add_argument('--outputdir', type=str, default='runs',
                    help='outputdir')
parser.add_argument('--policy', type=str, default='vanilla',
                    help='vanilla | primitive | mixture')
parser.add_argument('--opt', type=str, default='adam',
                    help='adam | sgd')
parser.add_argument('--debug', action='store_true',
                    help='debug')
parser.add_argument('--printf', action='store_true',
                    help='printf')
parser.add_argument('--fixed-std', type=float, default=0.1,
                    help='fixed std')
parser.add_argument('--vwght', type=str, default='0 0',
                    help='weight for xy: 1 0 is x vel forward, 0 -1 is y vel backward')
parser.add_argument('--goal-dist', type=float, default=10,
                    help='goal distance (default: 10)')

parser.add_argument('--control-weight', type=float, default=0.1,
                    help='control weight(default: 0.1)')
parser.add_argument('--contact-weight', type=float, default=0.1,
                    help='contact weight (default: 0.1)')
parser.add_argument('--task-weight', type=float, default=0.7,
                    help='task weight (default: 0.7)')
parser.add_argument('--healthy-weight', type=float, default=0.1,
                    help='healthy weight (default: 0.1)')
parser.add_argument('--task-scale', type=float, default=1,
                    help='task scale (default: 1)')

parser.add_argument('--running-state', action='store_true',
                    help='running state')
parser.add_argument('--multitask', action='store_true', default=True,
                    help='multitask')
parser.add_argument('--multitask-for-transfer', action='store_true',
                    help='multitask for transfer')

parser.add_argument('--nprims', type=int, default=1, metavar='N',
                    help='number of primitives (default: 1)')
parser.add_argument('--for-transfer', action='store_true', default=True,
                    help='multitask for transfer')

parser.add_argument('--tasks', type=str, default='1234_1234',
                    help='quadrants for training_testing (default: 1234_1234)')
parser.add_argument('--nsamp', type=int, default=2, metavar='N',
                    help='number of times we sample from test (default: 1)')

args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
    os.system('export OMP_NUM_THREADS=1')

def build_expname(args, ext=''):
    expname = 'env-{}'.format(args.env_name)
    expname += '_p-{}'.format(args.policy)
    expname += '_np-{}'.format(args.nprims)
    expname += '_tt-{}'.format(args.tasks)
    expname += '_wef-{}'.format(args.weight_entropy_coeff)
    expname += '_klp-{}'.format(args.klp)
    expname += '_s{}'.format(args.seed)
    expname += ext
    if args.debug: expname+= '_debug'
    return expname

def initialize_logger(logger):
    logger.add_variable('i_iter')
    logger.add_variable('min_reward', include_running_avg=True)
    logger.add_variable('avg_reward', include_running_avg=True)
    logger.add_variable('max_reward', include_running_avg=True)
    # only works if include_running_avg for 'return'!
    logger.add_metric('running_min_reward', -np.inf, operator.ge)
    logger.add_metric('running_avg_reward', -np.inf, operator.ge)
    logger.add_metric('running_max_reward', -np.inf, operator.ge)

def process_args(args):
    if args.debug:
        args.maxeplen = 100
        args.max_iter_num = 5
        args.save_every = 1
        args.visualize_every = 5
        args.min_batch_size = 128
        args.num_threads = 1
        args.num_test = 2
        args.nprims = 2
        args.plr = 1e-2
        args.nsamp = 2
    return args

def initialize_actor_critic(env, device):
    state_dim = env.observation_space.shape[0]
    is_disc_action = len(env.action_space.shape) == 0

    action_dim = env.action_space.n if is_disc_action else env.action_space.shape[0]

    """define actor and critic"""
    if args.model_path is None:
        if is_disc_action:
            policy_net = DiscretePolicy(state_dim, action_dim)
            value_net = Value(state_dim)
        else:
            if args.policy == 'vanilla':
                policy_net = Policy(state_dim, action_dim, log_std=args.log_std)
                value_net = Value(state_dim)
            elif args.policy == 'primitive':
                goal_dim = env.env.goal_dim
                hdim = 64 if args.debug else 128
                encoder = Feedforward([state_dim+goal_dim, hdim], out_act=F.relu)
                policy_net = PrimitivePolicy(encoder=encoder, bottleneck_dim=hdim, decoder_dims=[hdim, action_dim], device=device, id=0)
                value_net = Value(state_dim+goal_dim)
            elif args.policy == 'composite':
                num_primitives = args.nprims
                goal_dim = env.env.goal_dim
                hdim = 64 if args.debug else 128
                encoders = [Feedforward([state_dim, hdim], out_act=F.relu) for i in range(num_primitives)]
                primitive_builder = lambda e, i: PrimitivePolicy(encoder=e, bottleneck_dim=hdim, decoder_dims=[hdim, action_dim], device=device, id=i)
                weight_network = WeightNetwork(state_dim=state_dim, goal_dim=goal_dim, encoder_dims=[hdim], bottleneck_dim=hdim, decoder_dims=[hdim, num_primitives], device=device)
                policy_net = CompositePolicy(weight_network=weight_network, primitives=nn.ModuleList([primitive_builder(e, i) for i, e in enumerate(encoders)]), obs_dim=state_dim, device=device) 
                value_net = Value(state_dim+goal_dim)
            elif args.policy == 'latent':
                goal_dim = env.env.goal_dim
                hdim = 64 if args.debug else 128
                zdim = args.nprims
                goal_embedder = GoalEmbedder(dims=[goal_dim, hdim, hdim, zdim], obs_dim=state_dim, device=device) # good
                policy_net = LatentPolicy(goal_embedder=goal_embedder, network_dims=[state_dim+zdim, hdim, hdim], outdim=action_dim, obs_dim=state_dim, device=device)
                value_net = Value(state_dim+goal_dim)
            else:
                False
    ######################################################
    # TODO verify that this works
    # TODO: make this separate
    if args.resume:
        policy_net, value_net, running_state = pickle.load(open(args.model_path, "rb"))
        # TODO: load from checkpoint
        ckpt = torch.load(args.resume)
        policy_net.load_state_dict(ckpt['policy'])
        value_net.load_state_dict(ckpt['valuefn'])
        running_state = ckpt['running_state']
    #######################################################
    policy_net.to(device)
    value_net.to(device)
    return policy_net, value_net

def reset_for_transfer(env, policy_net, value_net, device, args):
    if args.policy == 'composite':
        state_dim = env.observation_space.shape[0]
        goal_dim = env.env.goal_dim
        num_primitives = args.nprims
        hdim = 64 if args.debug else 128
        weight_network = WeightNetwork(state_dim=state_dim, goal_dim=goal_dim, encoder_dims=[hdim], bottleneck_dim=hdim, decoder_dims=[hdim, num_primitives], device=device, fixed_std=args.fixed_std)
        policy_net = CompositeTransferPolicy(weight_network=weight_network, primitives=policy_net.primitives, obs_dim=state_dim, device=device)
        value_net = Value(state_dim+goal_dim)
        policy_net.to(device)
        value_net.to(device)
    elif args.policy == 'primitive':
        policy_net.zero_grad()
        value_net.zero_grad()
    elif args.policy == 'latent':
        state_dim = env.observation_space.shape[0]
        goal_dim = env.env.goal_dim
        hdim = 64 if args.debug else 128
        zdim = args.nprims
        goal_embedder = GoalEmbedder(dims=[state_dim+goal_dim, hdim, hdim, zdim], obs_dim=state_dim, device=device, fixed_std=args.fixed_std, ignore_obs=False)
        policy_net = LatentTransferPolicy(goal_embedder=goal_embedder, decoder=policy_net.decoder, obs_dim=state_dim, device=device)
        value_net = Value(state_dim+goal_dim)
        policy_net.to(device)
        value_net.to(device)
    else:
        assert False
    return policy_net, value_net

def setup_experiment(args):
    args = process_args(args)

    """environment"""
    env = initialize_environment(args)

    """seeding"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)
    return env

def initialize_experiment(env, policy_net, value_net, device, args, running_state, ext=''):
    agent = Agent(env, policy_net, value_net, device, args, running_state=running_state, num_threads=args.num_threads)
    logger = create_logger(lambda params: build_expname(args=params, ext=ext), args)
    initialize_logger(logger)
    return agent, logger

def run_experiment(agent, env, logger, device, dtype, running_state, args):
    rl_alg = PPO(agent=agent, args=args, dtype=dtype, device=device)
    exp = Experiment(agent, env, rl_alg, logger, running_state, args)
    exp.main_loop()

def main(args):
    setup_experiment(args)
    if args.running_state:
        running_state = ZFilter((state_dim,), clip=5)
    else:
        running_state = None
    # running_reward = ZFilter((1,), demean=False, clip=10)
    policy_net, value_net = initialize_actor_critic(env, device)
    env.env.train_mode()
    agent, logger = initialize_experiment(env, policy_net, value_net, device, args, running_state)
    run_experiment(agent, env, logger, device, dtype, running_state, args)

def visualize_params(dict_of_models, pfunc):
    for k, v in dict_of_models.items():
        pfunc('#'*20 + ' {} '.format(k) + '#'*(80-len(k)-2-20))
        visualize_parameters(v, pfunc)
    pfunc('#'*80)

def pretrain_transfer(args, device, vis_p):
    env = setup_experiment(args)
    policy_net, value_net = initialize_actor_critic(env, device)
    env.env.train_mode()
    agent, logger = initialize_experiment(env, policy_net, value_net, device, args, None)
    vis_p('Initial', policy_net, value_net, logger)
    run_experiment(agent, env, logger, device, dtype, None, args)
    vis_p('After Training', policy_net, value_net, logger)
    ######################################################################
    policy_net, value_net = reset_for_transfer(env, policy_net, value_net, device, args)
    env.env.test_mode()
    agent, logger = initialize_experiment(env, policy_net, value_net, device, args, None, ext='_transfer')
    vis_p('After Reset', policy_net, value_net, logger)
    run_experiment(agent, env, logger, device, dtype, None, args)
    vis_p('After Transfer', policy_net, value_net, logger)

def main_transfer_composite(args):
    def vis_p(label, policy_net, value_net, logger):
        logger.printf(label)
        visualize_params({
            'Weight Network': policy_net.weight_network,
            'Value Network': value_net,
            'Primitive 0': policy_net.primitives[0]},
            pfunc=lambda x: logger.printf(x))
    pretrain_transfer(args, device, vis_p)

def main_transfer_latent(args):
    def vis_p(label, policy_net, value_net, logger):
        logger.printf(label)
        visualize_params({
            'Decoder Network': policy_net.decoder,
            'Encoder Network': policy_net.goal_embedder,
            'Value Network': value_net},
            pfunc=lambda x: logger.printf(x))
    pretrain_transfer(args, device, vis_p)

def main_transfer_primitive(args):
    def vis_p(label, policy_net, value_net, logger):
        logger.printf(label)
        visualize_params({
            'Policy Network': policy_net,
            'Value Network': value_net},
            pfunc=lambda x: logger.printf(x))
    pretrain_transfer(args, device, vis_p)

if __name__ == '__main__':
    if args.for_transfer:
        if args.policy == 'primitive':
            main_transfer_primitive(args)
        elif args.policy == 'composite':
            main_transfer_composite(args)
        elif args.policy == 'latent':
            main_transfer_latent(args)
        else:
            assert False
    else:
        main(args)




