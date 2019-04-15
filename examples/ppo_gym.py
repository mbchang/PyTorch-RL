import argparse
from collections import defaultdict
import copy
import gym
from moviepy.editor import ImageSequenceClip
import operator
import os
import sys
import pickle
import time
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.agent import Agent
from core.rl_algs import PPO
from infra.log import create_logger, display_stats, merge_log, visualize_parameters
from infra.env_config import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from models.primitives import Feedforward, CompositePolicy, WeightNetwork, PrimitivePolicy
from utils import *

parser = argparse.ArgumentParser(description='PyTorch PPO example')
parser.add_argument('--env-name', default="Hopper-v2", metavar='G',
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

parser.add_argument('--maxeplen', type=int, default=10000, metavar='N',
                    help='maximal number of main iterations (default: 10000)')
parser.add_argument('--num-test', type=int, default=100,
                    help='number of test trajectories (default: 100)')
parser.add_argument('--resume', type=str, default='',
                    help='.tar path of saved model')
parser.add_argument('--outputdir', type=str, default='runs',
                    help='outputdir')
parser.add_argument('--policy', type=str, default='vanilla',
                    help='vanilla | primitive | mixture')
parser.add_argument('--opt', type=str, default='sgd',
                    help='adam | sgd')
parser.add_argument('--debug', action='store_true',
                    help='debug')
parser.add_argument('--printf', action='store_true',
                    help='printf')
parser.add_argument('--fixed-var', action='store_true',
                    help='fixed variance')
parser.add_argument('--vwght', type=str, default='1 0',
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
parser.add_argument('--multitask', action='store_true',
                    help='multitask')
parser.add_argument('--multitask-for-transfer', action='store_true',
                    help='multitask for transfer')

parser.add_argument('--nprims', type=int, default=1, metavar='N',
                    help='number of primitives (default: 1)')
parser.add_argument('--for-transfer', action='store_true',
                    help='multitask for transfer')

parser.add_argument('--tasks', type=str, default='1234_1234',
                    help='quadrants for training_testing (default: 1234_1234)')
parser.add_argument('--nsamp', type=int, default=1, metavar='N',
                    help='number of times we sample from test (default: 1)')

args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
    os.system('export OMP_NUM_THREADS=1')

class Experiment():
    def __init__(self, agent, env, rl_alg, logger, running_state, args):
        self.agent = agent
        self.env = env
        self.rl_alg = rl_alg
        self.logger = logger
        self.running_state = running_state
        self.args = args

    def sample_single_trajectory(self, render):
        raise NotImplementedError
        episode_data = []
        state = self.env.reset()
        reward_episode = 0
        for t in range(self.args.maxeplen):  # Don't infinite loop while learning
            state_var = tensor(state).unsqueeze(0)
            with torch.no_grad():
                action = self.agent.policy.select_action(state_var)[0].numpy()
            action = int(action) if self.agent.policy.is_disc_action else action.astype(np.float64)
            next_state, reward, done, info = self.env.step(action)
            reward_episode += reward
            mask = 0 if done else 1
            e = copy.deepcopy(info)
            e.update({'reward_total': reward})
            if render:
                frame = self.env.render(mode='rgb_array')
                e['frame'] = frame
            episode_data.append(e)
            if done:
                break
            state = next_state
        return episode_data

    def sample_trajectory(self, render):
        raise NotImplementedError
        with torch.no_grad():
            best_reward_episode = -np.inf
            best_episode_data = {}
            for i in range(self.args.num_test):
                episode_data = self.sample_single_trajectory(render)
                reward_episode = np.sum([e['reward_total'] for e in episode_data])
                if reward_episode > best_reward_episode:
                    best_reward_episode = reward_episode
                    best_episode_data = copy.deepcopy(episode_data)
        return best_episode_data

    def visualize(self, policy_name, i_episode, episode_data, mode, sample_num):
        frames = np.array([e['frame'] for e in episode_data])
        if self.env.env.multitask:
            # goal = episode_data[0]['goal']
            # label = '_g[{:.3f},{:.3f}]'.format(goal[0], goal[1])
            goal_x, goal_y = episode_data[0]['goal'] # radians
            goal_angle = np.arctan2(goal_y, goal_x)*180/np.pi
            label = '_g{:.3f}'.format(goal_angle)
        else:
            label = ''
        clip = ImageSequenceClip(list(frames), fps=30).resize(0.5)
        clip.write_gif('{}/{}-{}-{}{}_{}.gif'.format(self.logger.logdir, policy_name, mode, i_episode, label, sample_num), fps=30)

    def save(self, i_iter):
        metric_keys = [
            'running_min_reward', 
            'running_avg_reward',
            'running_max_reward',
            'min_reward', 
            'avg_reward',
            'max_reward',
            ]
        self.logger.printf('Saving to {}'.format(self.logger.logdir))
        self.logger.save_csv(clear_data=False)
        self.logger.plot_from_csv(var_pairs=[('i_iter', k) for k in metric_keys])
        ckpt = {
            # or you should probably actually just save the whole thing
            'policy': self.logger.to_cpu(self.agent.policy.state_dict()),
            'valuefn': self.logger.to_cpu(self.agent.valuefn.state_dict()),
            'i_iter': i_iter,
            'running_state': self.running_state
        }
        self.logger.save_checkpoint(ckpt_data=ckpt, current_metric_keys=metric_keys, i_iter=i_iter, ext='_train')
        self.logger.clear_data() # to save memory
        to_device(torch.device('cpu'), self.agent.policy, self.agent.valuefn)
        pickle.dump((self.agent.policy, self.agent.valuefn, self.running_state),
                    open(os.path.join(assets_dir(), 'learned_models/{}_ppo.p'.format(args.env_name)), 'wb'))
        to_device(device, self.agent.policy, self.agent.valuefn)

    def test(self, policy, i_iter, hide_goal, sample_num=0):
        to_device(torch.device('cpu'), policy)
        with torch.no_grad():
            test_batch, test_log = self.agent.collect_samples(
                policy=policy,
                min_batch_size=args.min_batch_size, 
                deterministic=True, 
                render=True,
                hide_goal=hide_goal)

        best_episode_data = self.log(test_log, i_iter, policy.name)
        self.logger.printf('Test {} Sample {}\tT_sample {:.4f}\tR_min {:.2f}\tR_max {:.2f}\tR_avg {:.2f}'.format(
        i_iter, sample_num, test_log['sample_time'], test_log['min_reward'], test_log['max_reward'], test_log['avg_reward']))
        to_device(self.agent.device, policy)
        self.visualize(policy_name=policy.name, i_episode=i_iter, episode_data=best_episode_data, mode='test', sample_num=sample_num)

    def log(self, log, i_iter, policy_name):
        best_episode_data = log['best_episode_data']
        best_merged_episode_data = merge_log(best_episode_data)
        self.logger.printf('Best Episode Data: {}'.format(policy_name))
        self.logger.printf(display_stats(best_merged_episode_data))
        worst_episode_data = log['worst_episode_data']
        worst_merged_episode_data = merge_log(worst_episode_data)
        self.logger.printf('Worst Episode Data: {}'.format(policy_name))
        self.logger.printf(display_stats(worst_merged_episode_data))
        return best_episode_data

    def main_loop(self):
        for i_iter in range(args.max_iter_num+1):
            self.logger.update_variable(name='i_iter', index=i_iter, value=i_iter)
            should_log = i_iter % self.args.log_every == 0
            should_save = i_iter % self.args.save_every == 0
            should_visualize = i_iter % self.args.visualize_every == 0

            if should_visualize:
                for sn in range(args.nsamp):
                    self.test(policy=self.agent.policy, i_iter=i_iter, hide_goal=False, sample_num=sn)
                if self.args.policy == 'composite':
                    for p in self.agent.policy.primitives:
                        self.test(policy=p, i_iter=i_iter, hide_goal=True)

            """generate multiple trajectories that reach the minimum batch_size"""
            batch, log = self.agent.collect_samples(
                policy=self.agent.policy, 
                min_batch_size=args.min_batch_size)  # here you can record the action mean and std

            for metric in ['min_reward', 'avg_reward', 'max_reward']:
                self.logger.update_variable(
                    name=metric, index=i_iter, value=log[metric], include_running_avg=True)

            t0 = time.time()
            self.rl_alg.update_params(batch, i_iter, self.agent)
            t1 = time.time()
            self.logger.printf(display_stats(self.rl_alg.aggregate_stats()))

            if should_log:
                best_episode_data = self.log(log, i_iter, self.agent.policy.name)
                self.logger.printf('{}\tT_sample {:.4f}\tT_update {:.4f}\tR_min {:.2f}\tR_max {:.2f}\tR_avg {:.2f}'.format(
                    i_iter, log['sample_time'], t1-t0, log['min_reward'], log['max_reward'], log['avg_reward']))

            if should_save:
                self.save(i_iter)

            """clean up gpu memory"""
            torch.cuda.empty_cache()

def build_expname(args, ext=''):
    expname = 'env-{}'.format(args.env_name)
    expname += '_opt-{}'.format(args.opt)
    expname += '_plr-{}'.format(args.plr)
    expname += '_clr-{}'.format(args.clr)
    expname += '_eplen-{}'.format(args.maxeplen)
    expname += '_ntest-{}'.format(args.num_test)
    expname += '_p-{}'.format(args.policy)

    expname += '_gd-{}'.format(args.goal_dist)
    expname += '_vw-{}'.format(args.vwght.replace(' ', ''))
    expname += '_ctlw-{}'.format(args.control_weight)
    expname += '_cntw-{}'.format(args.contact_weight)
    expname += '_tw-{}'.format(args.task_weight)
    expname += '_hw-{}'.format(args.healthy_weight)
    expname += '_ts-{}'.format(args.task_scale)
    expname += '_np-{}'.format(args.nprims)
    expname += '_tt-{}'.format(args.tasks)

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
        args.num_test = 5
        args.nprims = 4
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
                # if args.debug:
                #     encoder = Feedforward([state_dim, 64, 64], out_act=F.relu)
                #     policy_net = PrimitivePolicy(encoder=encoder, bottleneck_dim=64, decoder_dims=[64, action_dim], device=device, id=0, fixed_var=args.fixed_var, vib=False)
                # else:
                #     encoder = Feedforward([state_dim, 128], out_act=F.relu)
                #     policy_net = PrimitivePolicy(encoder=encoder, bottleneck_dim=128, decoder_dims=[128, action_dim], device=device, id=0)
                # value_net = Value(state_dim)

                goal_dim = env.env.goal_dim
                if args.debug:
                    encoder = Feedforward([state_dim+goal_dim, 64, 64], out_act=F.relu)
                    policy_net = PrimitivePolicy(encoder=encoder, bottleneck_dim=64, decoder_dims=[64, action_dim], device=device, id=0, fixed_var=args.fixed_var, vib=False)
                else:
                    encoder = Feedforward([state_dim+goal_dim, 128], out_act=F.relu)
                    policy_net = PrimitivePolicy(encoder=encoder, bottleneck_dim=128, decoder_dims=[128, action_dim], device=device, id=0)
                value_net = Value(state_dim+goal_dim)


            elif args.policy == 'composite':
                num_primitives = args.nprims
                goal_dim = env.env.goal_dim
                hdim = 64 if args.debug else 128
                encoders = [Feedforward([state_dim, hdim], out_act=F.relu) for i in range(num_primitives)]
                primitive_builder = lambda e, i: PrimitivePolicy(encoder=e, bottleneck_dim=hdim, decoder_dims=[hdim, action_dim], device=device, id=i)
                weight_network = WeightNetwork(state_dim=state_dim, goal_dim=goal_dim, encoder_dims=[hdim], bottleneck_dim=hdim, decoder_dims=[hdim, num_primitives], device=device)
                policy_net = CompositePolicy(weight_network=weight_network, primitives=nn.ModuleList([primitive_builder(e, i) for i, e in enumerate(encoders)]), obs_dim=state_dim) 
                value_net = Value(state_dim+goal_dim)
            else:
                False
    ######################################################
    # TODO verify that this works
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

def reset_weightnet_critic(env, composite_policy, device):
    state_dim = env.observation_space.shape[0]
    goal_dim = env.env.goal_dim
    num_primitives = args.nprims
    hdim = 64 if args.debug else 128
    weight_network = WeightNetwork(state_dim=state_dim, goal_dim=goal_dim, encoder_dims=[hdim], bottleneck_dim=hdim, decoder_dims=[hdim, num_primitives], device=device)
    composite_policy.weight_network = weight_network
    value_net = Value(state_dim+goal_dim)
    composite_policy.freeze_primitives = True
    composite_policy.to(device)
    value_net.to(device)
    return composite_policy, value_net

def main(args):
    args = process_args(args)

    """environment"""
    env = initialize_environment(args)

    """seeding"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)

    if args.running_state:
        running_state = ZFilter((state_dim,), clip=5)
    else:
        running_state = None
    # running_reward = ZFilter((1,), demean=False, clip=10)

    # """define actor and critic"""
    policy_net, value_net = initialize_actor_critic(env, device)
    """create agent"""
    env.env.train_mode()
    agent = Agent(env, policy_net, value_net, device, args, running_state=running_state, num_threads=args.num_threads)
    logger = create_logger(build_expname, args)
    initialize_logger(logger)
    rl_alg = PPO(agent=agent, args=args, dtype=dtype, device=device)
    exp = Experiment(agent, env, rl_alg, logger, running_state, args)
    exp.main_loop()

def visualize_params(dict_of_models, pfunc):
    for k, v in dict_of_models.items():
        pfunc('#'*20 + ' {} '.format(k) + '#'*(80-len(k)-2-20))
        visualize_parameters(v, pfunc)
    pfunc('#'*80)

def main_transfer_composite(args):
    args = process_args(args)
    assert args.policy == 'composite'

    """environment"""
    env = initialize_environment(args)

    """seeding"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)

    # """define actor and critic"""
    policy_net, value_net = initialize_actor_critic(env, device)

    """create agent"""
    env.env.train_mode()
    agent = Agent(env, policy_net, value_net, device, args, running_state=None, num_threads=args.num_threads)
    logger = create_logger(build_expname, args)
    initialize_logger(logger)

    logger.printf('Initial')
    visualize_params({
        'Weight Network': policy_net.weight_network,
        'Value Network': value_net,
        'Primitive 0': policy_net.primitives[0]},
        pfunc=lambda x: logger.printf(x))

    rl_alg = PPO(agent=agent, args=args, dtype=dtype, device=device)
    exp = Experiment(agent, env, rl_alg, logger, None, args)
    exp.main_loop()

    logger.printf('After Training')
    visualize_params({
        'Weight Network': policy_net.weight_network,
        'Value Network': value_net,
        'Primitive 0': policy_net.primitives[0]},
        pfunc=lambda x: logger.printf(x))

    ######################################################################

    # now reset the weight network and the value function.
    policy_net, value_net = reset_weightnet_critic(env, agent.policy, device)

    env.env.test_mode()
    agent = Agent(env, policy_net, value_net, device, args, running_state=None, num_threads=args.num_threads)
    logger = create_logger(lambda params: build_expname(args=params, ext='_transfer'), args)  # with transfer tag
    initialize_logger(logger)  # should save in the same folder as before

    logger.printf('After Reset')
    visualize_params({
        'Weight Network': policy_net.weight_network,
        'Value Network': value_net,
        'Primitive 0': policy_net.primitives[0]},
        pfunc=lambda x: logger.printf(x))

    rl_alg = PPO(agent=agent, args=args, dtype=dtype, device=device)
    exp = Experiment(agent, env, rl_alg, logger, None, args)
    exp.main_loop()

    logger.printf('After Transfer')
    visualize_params({
        'Weight Network': policy_net.weight_network,
        'Value Network': value_net,
        'Primitive 0': policy_net.primitives[0]},
        pfunc=lambda x: logger.printf(x))

def main_transfer_primitive(args):
    args = process_args(args)
    assert args.policy == 'primitive'

    """environment"""
    env = initialize_environment(args)

    """seeding"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)

    # """define actor and critic"""
    policy_net, value_net = initialize_actor_critic(env, device)

    """create agent"""
    env.env.train_mode()
    agent = Agent(env, policy_net, value_net, device, args, running_state=None, num_threads=args.num_threads)
    logger = create_logger(build_expname, args)
    initialize_logger(logger)

    logger.printf('Initial')
    visualize_params({
        'Policy Network': policy_net,
        'Value Network': value_net},
        pfunc=lambda x: logger.printf(x))

    rl_alg = PPO(agent=agent, args=args, dtype=dtype, device=device)
    exp = Experiment(agent, env, rl_alg, logger, None, args)
    exp.main_loop()

    logger.printf('After Training')
    visualize_params({
        'Policy Network': policy_net,
        'Value Network': value_net},
        pfunc=lambda x: logger.printf(x))

    ######################################################################

    # now reset the weight network and the value function.
    # policy_net, value_net = reset_weightnet_critic(env, agent.policy, device)
    policy_net.zero_grad()
    value_net.zero_grad()

    env.env.test_mode()
    agent = Agent(env, policy_net, value_net, device, args, running_state=None, num_threads=args.num_threads)
    logger = create_logger(lambda params: build_expname(args=params, ext='_transfer'), args)  # with transfer tag
    initialize_logger(logger)  # should save in the same folder as before

    logger.printf('Initial for Transfer')
    visualize_params({
        'Policy Network': policy_net,
        'Value Network': value_net},
        pfunc=lambda x: logger.printf(x))

    rl_alg = PPO(agent=agent, args=args, dtype=dtype, device=device)
    exp = Experiment(agent, env, rl_alg, logger, None, args)
    exp.main_loop()

    logger.printf('After Transfer')
    visualize_params({
        'Policy Network': policy_net,
        'Value Network': value_net},
        pfunc=lambda x: logger.printf(x))

if __name__ == '__main__':
    if args.for_transfer:
        if args.policy == 'primitive':
            main_transfer_primitive(args)
        else:
            main_transfer_composite(args)
    else:
        main(args)




