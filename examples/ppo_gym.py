import argparse
import gym
import os
import sys
import pickle
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from models.primitives import Feedforward, PrimitivePolicy, CompositePolicy, WeightNetwork
from core.agent import Agent

from moviepy.editor import ImageSequenceClip
import copy
import operator

import torch.nn as nn
import torch.nn.functional as F



from infra.log import create_logger
from core.rl_algs import PPO


parser = argparse.ArgumentParser(description='PyTorch PPO example')
parser.add_argument('--env-name', default="Hopper-v2", metavar='G',
                    help='name of the environment to run')
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
parser.add_argument('--plr', type=float, default=4e-5, metavar='G',
                    help='policy learning rate (default: 4e-5)')
parser.add_argument('--clr', type=float, default=5e-3, metavar='G',
                    help='critic learning rate (default: 5e-3)')
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

parser.add_argument('--vwght', type=str, default='1 0',
                    help='weight for xy: 1 0 is x vel forward, 0 -1 is y vel backward')


args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
    os.system('export OMP_NUM_THREADS=1')

def merge_log(log_list):
    """ This is domain specific """
    log = dict()
    metrics = [
        'reward_forward', 
        'reward_ctrl', 
        'reward_contact', 
        'reward_survive', 
        'x_position', 
        'y_position', 
        'distance_from_origin',
        'x_velocity',
        'y_velocity',
        'reward_total']
    aggregators = {'total': np.sum, 'avg': np.mean, 'max': np.max, 'min': np.min, 'std': np.std}
    for m in metrics:
        metric_data = [x[m] for x in log_list]
        for a in aggregators:
            log['{}_{}'.format(a, m)] = aggregators[a](metric_data)
    return log

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

    def visualize(self, i_episode, episode_data, mode):
        frames = np.array([e['frame'] for e in episode_data])
        clip = ImageSequenceClip(list(frames), fps=30).resize(0.5)
        clip.write_gif('{}/{}-{}.gif'.format(self.logger.logdir, mode, i_episode), fps=30)

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

    def test(self, i_iter):
        to_device(torch.device('cpu'), self.agent.policy)
        with torch.no_grad():
            test_batch, test_log = self.agent.collect_samples(
                args.min_batch_size, deterministic=True, render=True)
        episode_data = test_log['episode_data']
        merged_episode_data = merge_log(episode_data)
        self.logger.pprintf(merged_episode_data)
        self.logger.printf('Test {}\tT_sample {:.4f}\tR_min {:.2f}\tR_max {:.2f}\tR_avg {:.2f}'.format(
        i_iter, test_log['sample_time'], test_log['min_reward'], test_log['max_reward'], test_log['avg_reward']))
        to_device(self.agent.device, self.agent.policy)
        self.visualize(i_iter, episode_data, mode='test')

    def main_loop(self):
        for i_iter in range(args.max_iter_num+1):
            self.logger.update_variable(name='i_iter', index=i_iter, value=i_iter)
            should_log = i_iter % self.args.log_every == 0
            should_save = i_iter % self.args.save_every == 0
            should_visualize = i_iter % self.args.visualize_every == 0

            """generate multiple trajectories that reach the minimum batch_size"""
            batch, log = self.agent.collect_samples(args.min_batch_size)

            for metric in ['min_reward', 'avg_reward', 'max_reward']:
                self.logger.update_variable(
                    name=metric, index=i_iter, value=log[metric], include_running_avg=True)

            if should_visualize:
                self.test(i_iter)

            t0 = time.time()
            self.rl_alg.update_params(batch, i_iter, self.agent)
            t1 = time.time()

            if should_log:
                self.logger.printf('{}\tT_sample {:.4f}\tT_update {:.4f}\tR_min {:.2f}\tR_max {:.2f}\tR_avg {:.2f}'.format(
                    i_iter, log['sample_time'], t1-t0, log['min_reward'], log['max_reward'], log['avg_reward']))

            if should_save:
                self.save(i_iter)

            """clean up gpu memory"""
            torch.cuda.empty_cache()

def visualize_parameters(model, aString=None):
    if aString:
        print(aString)
    for n, p in model.named_parameters():
        if p.grad is None:
            print(n, p.size(), p.data.norm(), "No grad")
        else:
            print(n, p.size(), p.data.norm(), p.grad.data.norm(), torch.max(p.grad.data))

def build_expname(args):
    expname = 'env-{}'.format(args.env_name)
    expname += '_opt-{}'.format(args.opt)
    expname += '_plr-{}'.format(args.plr)
    expname += '_clr-{}'.format(args.clr)
    expname += '_eplen-{}'.format(args.maxeplen)
    expname += '_ntest-{}'.format(args.num_test)
    expname += '_vw-{}'.format(args.vwght.replace(' ', ''))
    expname += '_p-{}'.format(args.policy)
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
    return args

def make_renderer_track_agent(env):
    viewer = env.env._get_viewer('rgb_array')
    viewer.cam.type = 1
    viewer.cam.trackbodyid = 0

def initialize_environment(args):
    vw_str = args.vwght
    xw, yw = map(int, vw_str.split())
    vw = {'x': xw, 'y': yw}
    env = gym.make(args.env_name, velocity_weight=vw)
    state_dim = env.observation_space.shape[0]
    is_disc_action = len(env.action_space.shape) == 0
    make_renderer_track_agent(env)
    return env, state_dim, is_disc_action

def main(args):
    args = process_args(args)
    logger = create_logger(build_expname, args)
    initialize_logger(logger)

    """environment"""
    env, state_dim, is_disc_action = initialize_environment(args)

    """seeding"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)

    running_state = ZFilter((state_dim,), clip=5)
    # running_reward = ZFilter((1,), demean=False, clip=10)

    """define actor and critic"""
    if args.model_path is None:
        if is_disc_action:
            policy_net = DiscretePolicy(state_dim, env.action_space.n)
        else:
            if args.policy == 'vanilla':
                policy_net = Policy(state_dim, env.action_space.shape[0], log_std=args.log_std)
            elif args.policy == 'primitive':
                if args.debug:
                    encoder = Feedforward([state_dim, 64, 64], out_act=F.relu)
                    policy_net = PrimitivePolicy(encoder=encoder, ib_dims=[64, 64], hdim=64, outdim=env.action_space.shape[0], device=device)
                else:
                    # encoder = Feedforward([state_dim, 512, 256], out_act=F.relu)
                    # policy_net = PrimitivePolicy(encoder=encoder, ib_dims=[256, 128], hdim=256, outdim=env.action_space.shape[0], device=device)
                    encoder = Feedforward([state_dim, 128], out_act=F.relu)
                    policy_net = PrimitivePolicy(encoder=encoder, ib_dims=[128, 128], hdim=128, outdim=env.action_space.shape[0], device=device)
            elif args.policy == 'composite':
                num_primitives = 3
                if args.debug:
                    encoders = [Feedforward([state_dim, 64, 64], out_act=F.relu) for i in range(num_primitives)]
                    primitive_builder = lambda e: PrimitivePolicy(encoder=e, ib_dims=[64, 64], hdim=64, outdim=env.action_space.shape[0], device=device)
                    weight_network = WeightNetwork(state_dim=state_dim, goal_dim=state_dim, encoder_dims=[64, 64], bottleneck_dim=64, decoder_dims=[64, num_primitives], device=device)
                    policy_net = CompositePolicy(weight_network=weight_network, primitives=nn.ModuleList([primitive_builder(e) for e in encoders]))          
                else:
                    goal_dim = 2 # TODO
                    encoders = [Feedforward([state_dim, 512, 256], out_act=F.relu) for i in range(num_primitives)]
                    primitive_builder = lambda e: PrimitivePolicy(encoder=e, ib_dims=[256, 128], hdim=256, outdim=env.action_space.shape[0], device=device)
                    weight_network = WeightNetwork(state_dim=state_dim, goal_dim=goal_dim, encoder_dims=[512, 256], bottleneck_dim=128, decoder_dims=[256, num_primitives], device=device)
                    policy_net = CompositePolicy(weight_network=weight_network, primitives=nn.ModuleList([primitive_builder(e) for e in encoders]))      
            else:
                False
        value_net = Value(state_dim)
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

    """create agent"""
    agent = Agent(env, policy_net, value_net, device, args, running_state=running_state, num_threads=args.num_threads)

    rl_alg = PPO(agent=agent, args=args, dtype=dtype, device=device)

    exp = Experiment(agent, env, rl_alg, logger, running_state, args)
    exp.main_loop()

if __name__ == '__main__':
    main(args)




