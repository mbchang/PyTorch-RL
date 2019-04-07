import gym
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.tdm.envs.ant_env import GoalXYPosAnt
from rlkit.torch.tdm.envs.multitask_env import MultitaskToFlatEnv

def make_renderer_track_agent(env, args):
    if type(env) == NormalizedBoxEnv:
        viewer = env.wrapped_env._get_viewer('rgb_array')
    elif type(env) == MultitaskToFlatEnv:
        viewer = env.wrapped_env.wrapped_env._get_viewer('rgb_array')
    elif type(env) == gym.wrappers.time_limit.TimeLimit:
        viewer = env.env._get_viewer('rgb_array')
    else:
        assert False
    viewer.cam.type = 1
    viewer.cam.trackbodyid = 0
    if 'goal' in args.env_type:
        viewer.cam.distance = max(4, 1.5*args.goal_dist+2)

def get_env_root():
    terminal_output = os.popen('pip show gym').readlines()
    output_data = {}
    for line in terminal_output:
        delimiter = ': '
        delimiter_loc = line.find(delimiter)
        key = line[:delimiter_loc]
        value = line[delimiter_loc+len(delimiter):].strip('\n')
        output_data[key] = value
    root = os.path.join(output_data['Location'], 'gym/envs/mujoco')
    return root

def replace_file(root, orig_file, new_file):
    orig_file = os.path.join(root, orig_file)
    new_file = os.path.join(root, new_file)
    command = 'cp -r {} {}'.format(new_file, orig_file)
    os.system(command)

def initialize_environment(args):
    vw_str = args.vwght
    xw, yw = map(int, vw_str.split())
    vw = {'x': xw, 'y': yw}

    # NOTE: you cannot have multiple parallel runs work with the different environments!
    new_file = args.env_name.replace('-', '_').lower()

    if '-m-' in args.env_type:
        # multitask
        pass

    if '-n-' in args.env_type:
        # normalize
        new_file += '_norm'

    if 'vel' in args.env_type:
        # velocity
        new_file += '_vel'
        env_args = {'velocity_weight': vw, 
                    'multitask': args.multitask, 
                    'multitask_for_transfer': args.multitask_for_transfer,
                    }

    if 'goal' in args.env_type:
        # goal
        new_file += '_goal'
        env_args = {'goal_distance': args.goal_dist,
                    'exclude_current_positions_from_observation': False
                    }

    if args.debug:
        root = get_env_root()
        replace_file(root, orig_file='ant_v3.py', new_file=new_file+'.py')

    ######################################################
    env = gym.make(args.env_name, **env_args)
    # env = multitask(env)  # this should append the task to the state

    # env = gym.make(args.env_name, 
    #     velocity_weight=vw, 
    #     goal_distance=args.goal_dist,

    #     control_weight=args.control_weight,
    #     contact_weight=args.contact_weight,
    #     task_weight=args.task_weight,
    #     healthy_weight=args.healthy_weight,

    #     control_scale=4,
    #     contact_scale=250,
    #     task_scale=args.task_scale,  # don't need to change

    #     # if going to goal
    #     )
    ######################################################
    # env = NormalizedBoxEnv(GoalXYPosAnt(max_distance=1))
    # env = MultitaskToFlatEnv(env)
    # assert False
    ######################################################
    # state_dim = env.observation_space.shape[0]
    # is_disc_action = len(env.action_space.shape) == 0
    make_renderer_track_agent(env, args)
    return env#, state_dim, is_disc_action