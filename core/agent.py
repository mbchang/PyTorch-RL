import multiprocessing
from utils.replay_memory import Memory
from utils.torch import *
import math
import time

import torch.optim as optim
import copy


def collect_samples(pid, queue, env, policy, custom_reward,
                    mean_action, render, running_state, min_batch_size, maxeplen):
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

    while num_steps < min_batch_size:
        state = env.reset()
        if running_state is not None:
            state = running_state(state)
        reward_episode = 0

        episode_data = []

        for t in range(maxeplen):
            state_var = tensor(state).unsqueeze(0)
            with torch.no_grad():
                if mean_action:
                    action = policy(state_var)[0][0].numpy()
                else:
                    action = policy.select_action(state_var)[0].numpy()
            action = int(action) if policy.is_disc_action else action.astype(np.float64)
            next_state, reward, done, info = env.step(action)
            reward_episode += reward
            if running_state is not None:
                next_state = running_state(next_state)

            if custom_reward is not None:
                reward = custom_reward(state, action)
                total_c_reward += reward
                min_c_reward = min(min_c_reward, reward)
                max_c_reward = max(max_c_reward, reward)

            mask = 0 if done else 1

            memory.push(state, action, mask, next_state, reward)

            e = copy.deepcopy(info)
            e.update({'reward_total': reward})
            if render:
                frame = env.render(mode='rgb_array')
                e['frame'] = frame
            episode_data.append(e)


            if done:
                break

            state = next_state

        assert np.allclose(reward_episode, np.sum([e['reward_total'] for e in episode_data]))
        # log stats
        num_steps += (t + 1)
        num_episodes += 1
        total_reward += reward_episode

        if reward_episode > max_reward:
            best_episode_data = copy.deepcopy(episode_data)

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

    if render:
        log['episode_data'] = best_episode_data

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

    def __init__(self, env, policy, valuefn, device, args, custom_reward=None,
                 mean_action=False, render=False, running_state=None, num_threads=1):
        self.env = env
        self.policy = policy
        self.valuefn = valuefn
        self.device = device
        self.custom_reward = custom_reward
        self.mean_action = mean_action
        self.running_state = running_state
        self.render = render
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

    def collect_samples(self, min_batch_size, render=False):
        t_start = time.time()
        to_device(torch.device('cpu'), self.policy)
        thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
        queue = multiprocessing.Queue()
        workers = []

        for i in range(self.num_threads-1):
            worker_args = (i+1, queue, self.env, self.policy, self.custom_reward, self.mean_action,
                           False, self.running_state, thread_batch_size, self.args.maxeplen)
            workers.append(multiprocessing.Process(target=collect_samples, args=worker_args))
        for worker in workers:
            worker.start()

        memory, log = collect_samples(0, None, self.env, self.policy, self.custom_reward, self.mean_action,
                                      # self.render, 
                                      render,
                                      self.running_state, thread_batch_size, self.args.maxeplen)

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
        to_device(self.device, self.policy)
        t_end = time.time()
        log['sample_time'] = t_end - t_start
        log['action_mean'] = np.mean(np.vstack(batch.action), axis=0)
        log['action_min'] = np.min(np.vstack(batch.action), axis=0)
        log['action_max'] = np.max(np.vstack(batch.action), axis=0)
        return batch, log
