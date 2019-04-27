import copy
import math
import numpy as np
import torch
from collections import defaultdict

from core.common import estimate_advantages
from utils import *

class PPO():
    def __init__(self, agent, dtype, device, args, optim_epochs=10, optim_batch_size=256):
        self.agent = agent

        self.dtype = dtype
        self.device = device
        self.args = args

        self.optim_epochs = optim_epochs
        self.optim_batch_size = optim_batch_size
        self.optim_value_iternum = 1

        self.reset_record()

    def record(self, minibatch_log, epoch, iter):
        self.log[epoch][iter] = minibatch_log

    def reset_record(self):
        self.log = defaultdict(dict)

    # def aggregate_stats_per_epoch(self):
    #     stats = defaultdict(dict)
    #     aggregators = {'total': np.sum, 'avg': np.mean, 'max': np.max, 'min': np.min, 'std': np.std}

    #     # first merge within the epoch and see how that goes
    #     for e in self.log:
    #         for m in ['num_clipped', 'ratio_clipped', 'kl', 'value_loss', 'policy_surr', 'policy_loss']:
    #             metric_data = [v[m] for k, v in self.log[e].items()]
    #             for a in aggregators:
    #                 stats[e]['{}_{}'.format(a, m)] = aggregators[a](metric_data)
    #     return stats

    def aggregate_stats(self):
        stats = defaultdict(dict)
        aggregators = {'avg': np.mean, 'max': np.max, 'min': np.min, 'std': np.std}
        metrics = copy.deepcopy(list(self.log[0][0].keys()))
        metrics.remove('bsize')
        for m in metrics:
            metric_data = []
            for e in self.log:
                epoch_metric_data = [v[m] for k, v in self.log[e].items()]
                metric_data.extend(epoch_metric_data)
            for a in aggregators:
                stats[m][a] = aggregators[a](metric_data)
        return stats

    def update_params(self, batch, i_iter, agent):
        self.reset_record()
        states = torch.from_numpy(np.stack(batch.state)).to(self.dtype).to(self.device)
        actions = torch.from_numpy(np.stack(batch.action)).to(self.dtype).to(self.device)
        rewards = torch.from_numpy(np.stack(batch.reward)).to(self.dtype).to(self.device)
        masks = torch.from_numpy(np.stack(batch.mask)).to(self.dtype).to(self.device)
        with torch.no_grad():
            values = self.agent.valuefn(states)
            info = self.agent.policy.get_log_prob(states, actions)
            fixed_log_probs = info['log_prob']

        """get advantage estimation from the trajectories"""
        advantages, returns = estimate_advantages(rewards, masks, values, self.args.gamma, self.args.tau, self.device)

        """perform mini-batch PPO update"""
        optim_iter_num = int(math.ceil(states.shape[0] / self.optim_batch_size))
        for j in range(self.optim_epochs):
            perm = np.arange(states.shape[0])
            np.random.shuffle(perm)
            perm = LongTensor(perm).to(self.device)

            states, actions, returns, advantages, fixed_log_probs = \
                states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), fixed_log_probs[perm].clone()

            for i in range(optim_iter_num):
                ind = slice(i * self.optim_batch_size, min((i + 1) * self.optim_batch_size, states.shape[0]))
                states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                    states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

                minibatch_log = self.ppo_step(states_b, actions_b, returns_b, advantages_b, fixed_log_probs_b)
                self.record(minibatch_log=minibatch_log, epoch=j, iter=i)

    def ppo_step(self, states, actions, returns, advantages, fixed_log_probs):

        """update critic"""
        for _ in range(self.optim_value_iternum):
            values_pred = self.agent.valuefn(states)
            value_loss = (values_pred - returns).pow(2).mean()
            # weight decay
            for param in self.agent.valuefn.parameters():
                value_loss += param.pow(2).sum() * self.args.l2_reg
            self.agent.value_optimizer.zero_grad()
            value_loss.backward()
            self.agent.value_optimizer.step()

        """update policy"""
        info = self.agent.policy.get_log_prob(states, actions)
        log_probs = info['log_prob']
        entropy = info['entropy']
        kl = info['kl']

        ratio = torch.exp(log_probs - fixed_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.args.clip_epsilon, 1.0 + self.args.clip_epsilon) * advantages
        policy_surr = -torch.min(surr1, surr2).mean()  # mean over batch
        entropy_penalty = self.args.entropy_coeff * entropy.mean()  # mean over batch
        ib_penalty = self.args.klp * kl.mean()  # mean over batch
        policy_loss = policy_surr + ib_penalty - entropy_penalty  # TODO: add regularization of action mean

        if 'weight_entropy' in info:
            weight_entropy = info['weight_entropy']
            policy_loss -= self.args.weight_entropy_coeff * weight_entropy.mean()

        self.agent.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.policy.parameters(), 40)
        self.agent.policy_optimizer.step()

        """log"""
        num_clipped = (surr1-surr2).nonzero().size(0)
        ratio_clipped = num_clipped / states.size(0)
        log = {}
        log['num_clipped'] = num_clipped
        log['ratio_clipped'] = ratio_clipped
        log['entropy'] = entropy.mean().item()
        log['kl'] = kl.mean().item()
        log['bsize'] = states.size(0)
        log['value_loss'] = value_loss.item()
        log['policy_surr'] = policy_surr.item()
        log['policy_loss'] = policy_loss.item()
        if 'weight_entropy' in info:
            log['weight_entropy'] = weight_entropy.mean().item()
        if 'weight_std' in info:
            log['weight_std'] = info['weight_std'].mean().item()
        return log


