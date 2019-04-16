import multiprocessing
from utils.replay_memory import Memory
from utils.torch import *
import math
import time

import torch.optim as optim
import copy

# This thing should just take in a policy and environment and just run it.
def sample_single_trajectory(env, policy, custom_reward, mean_action, render, running_state, maxeplen, memory, hide_goal):

    ######################################################
    # if render:
    #     goal = env.sample_goal_for_rollout()
    #     env.set_goal(goal)
    ######################################################
    
    state = env.reset()
    ############################
    if not hide_goal:
        goal = env.env.reset_goal()
        state = env.env.append_goal_to_state(state, goal)
    ############################

    if running_state is not None:
        state = running_state(state)
    reward_episode = 0

    episode_data = []

    for t in range(maxeplen):
        state_var = tensor(state).unsqueeze(0)
        with torch.no_grad():
            action = policy.select_action(state_var, deterministic=mean_action)
        # action = int(action) if policy.is_disc_action else action.astype(np.float64)
        env_action = policy.post_process(state_var, action)[0].numpy()  # but why doesn't it work if I do torch no grad in the post_process step of CompositeTransferPolicy?
        env_action = int(env_action) if policy.is_disc_action else env_action.astype(np.float64)
        next_state, reward, done, info = env.step(env_action)

        ############################
        if not hide_goal:
            next_state = env.env.append_goal_to_state(next_state, goal)
        ############################

        reward_episode += reward
        if running_state is not None:
            next_state = running_state(next_state)

        if custom_reward is not None:
            reward = custom_reward(state, action)
            total_c_reward += reward
            min_c_reward = min(min_c_reward, reward)
            max_c_reward = max(max_c_reward, reward)

        mask = 0 if done else 1

        memory.push(state, action[0].numpy(), mask, next_state, reward)

        e = copy.deepcopy(info)
        e.update({'reward_total': reward})
        if render:
            frame = env.render(mode='rgb_array')
            e['frame'] = frame
            if env.env.multitask or env.env.multitask_for_transfer:
                e['goal'] = env.env.goal
        episode_data.append(e)

        if done:
            break

        state = next_state

    assert np.allclose(reward_episode, np.sum([e['reward_total'] for e in episode_data]))
    return episode_data, t

# This thing should just take in a policy and environment and just run it.
# Could I imagine calling this for the primitives?
def collect_samples(pid, queue, env, policy, custom_reward, mean_action, render, running_state, min_batch_size, maxeplen, hide_goal):
    torch.randn(pid)
    log = dict()
    memory = Memory()
    num_steps = 0
    total_reward = 0
    min_reward = 1e6
    max_reward = -1e6
    total_c_reward = 0
    min_c_reward = 1e6
    max_c_reward = -1e6
    num_episodes = 0

    best_episode_data = []
    worst_episode_data = []
    # avg_episode_data = []

    while num_steps < min_batch_size:
        episode_data, t = sample_single_trajectory(env, policy, custom_reward, mean_action, render, running_state, maxeplen, memory, hide_goal)

        reward_episode = np.sum([e['reward_total'] for e in episode_data])

        # log stats
        num_steps += (t + 1)
        num_episodes += 1
        total_reward += reward_episode

        if reward_episode > max_reward:
            best_episode_data = copy.deepcopy(episode_data)  # actually if you are not rendering you should just get the average.

        if reward_episode < min_reward:
            worst_episode_data = copy.deepcopy(episode_data)

        min_reward = min(min_reward, reward_episode)
        max_reward = max(max_reward, reward_episode)

    log['num_steps'] = num_steps
    log['num_episodes'] = num_episodes
    log['total_reward'] = total_reward
    log['avg_reward'] = total_reward / num_episodes
    log['max_reward'] = max_reward
    log['min_reward'] = min_reward
    if custom_reward is not None:
        log['total_c_reward'] = total_c_reward
        log['avg_c_reward'] = total_c_reward / num_steps
        log['max_c_reward'] = max_c_reward
        log['min_c_reward'] = min_c_reward

    # if render:
    log['best_episode_data'] = best_episode_data
    log['worst_episode_data'] = worst_episode_data
 
    if queue is not None:
        queue.put([pid, memory, log])
    else:
        return memory, log


def merge_log(log_list):
    log = dict()
    log['total_reward'] = sum([x['total_reward'] for x in log_list])
    log['num_episodes'] = sum([x['num_episodes'] for x in log_list])
    log['num_steps'] = sum([x['num_steps'] for x in log_list])
    log['avg_reward'] = log['total_reward'] / log['num_episodes']
    log['max_reward'] = max([x['max_reward'] for x in log_list])
    log['min_reward'] = min([x['min_reward'] for x in log_list])
    if 'total_c_reward' in log_list[0]:
        log['total_c_reward'] = sum([x['total_c_reward'] for x in log_list])
        log['avg_c_reward'] = log['total_c_reward'] / log['num_steps']
        log['max_c_reward'] = max([x['max_c_reward'] for x in log_list])
        log['min_c_reward'] = min([x['min_c_reward'] for x in log_list])

    return log


class Agent:

    def __init__(self, env, policy, valuefn, device, args, custom_reward=None, running_state=None, num_threads=1):
        self.env = env
        self.policy = policy
        self.valuefn = valuefn
        self.device = device
        self.custom_reward = custom_reward
        self.running_state = running_state
        self.num_threads = num_threads

        self.args = args
        self.initialize_optimizer(args)

    def initialize_optimizer(self, args):
        if args.opt == 'adam':
            self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.args.plr)
            self.value_optimizer = optim.Adam(self.valuefn.parameters(), lr=self.args.clr)
        elif args.opt == 'sgd':
            self.policy_optimizer = optim.SGD(self.policy.parameters(), lr=self.args.plr, momentum=0.9)
            self.value_optimizer = optim.SGD(self.valuefn.parameters(), lr=self.args.clr, momentum=0.9)
        else:
            assert False
        self.optimizer = {'policy_opt': self.policy_optimizer, 'value_opt': self.value_optimizer}

    def collect_samples(self, policy, min_batch_size, deterministic=False, render=False, hide_goal=False):
        t_start = time.time()
        to_device(torch.device('cpu'), policy)
        thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
        queue = multiprocessing.Queue()
        workers = []

        for i in range(self.num_threads-1):
            worker_args = (i+1, queue, self.env, policy, self.custom_reward, determinisitic, render, self.running_state, thread_batch_size, self.args.maxeplen, hide_goal)
            workers.append(multiprocessing.Process(target=collect_samples, args=worker_args))
        for worker in workers:
            worker.start()

        memory, log = collect_samples(0, None, self.env, policy, self.custom_reward, 
            deterministic, render, self.running_state, thread_batch_size, self.args.maxeplen, hide_goal)

        worker_logs = [None] * len(workers)
        worker_memories = [None] * len(workers)
        for _ in workers:
            pid, worker_memory, worker_log = queue.get()
            worker_memories[pid - 1] = worker_memory
            worker_logs[pid - 1] = worker_log
        for worker_memory in worker_memories:
            memory.append(worker_memory)
        batch = memory.sample()
        assert self.num_threads == 1
        if self.num_threads > 1:
            log_list = [log] + worker_logs
            log = merge_log(log_list)
        to_device(self.device, policy)
        t_end = time.time()
        log['sample_time'] = t_end - t_start
        log['action_mean'] = np.mean(np.vstack(batch.action), axis=0)
        log['action_min'] = np.min(np.vstack(batch.action), axis=0)
        log['action_max'] = np.max(np.vstack(batch.action), axis=0)
        return batch, log
