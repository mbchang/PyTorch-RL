import math
import numpy as np
import torch

from core.common import estimate_advantages
from utils import *

class PPO():
    def __init__(self, agent, dtype, device, args, optim_epochs=10, optim_batch_size=64):
        self.agent = agent

        self.dtype = dtype
        self.device = device
        self.args = args

        self.args.l2_reg = self.args.l2_reg
        self.optim_epochs = optim_epochs
        self.optim_batch_size = optim_batch_size
        self.optim_value_iternum = 1


    def update_params(self, batch, i_iter, agent):
        states = torch.from_numpy(np.stack(batch.state)).to(self.dtype).to(self.device)
        actions = torch.from_numpy(np.stack(batch.action)).to(self.dtype).to(self.device)
        rewards = torch.from_numpy(np.stack(batch.reward)).to(self.dtype).to(self.device)
        masks = torch.from_numpy(np.stack(batch.mask)).to(self.dtype).to(self.device)
        with torch.no_grad():
            values = self.agent.valuefn(states)
            fixed_log_probs = self.agent.policy.get_log_prob(states, actions)

        """get advantage estimation from the trajectories"""
        advantages, returns = estimate_advantages(rewards, masks, values, self.args.gamma, self.args.tau, self.device)

        """perform mini-batch PPO update"""
        optim_iter_num = int(math.ceil(states.shape[0] / self.optim_batch_size))
        for _ in range(self.optim_epochs):
            perm = np.arange(states.shape[0])
            np.random.shuffle(perm)
            perm = LongTensor(perm).to(self.device)

            states, actions, returns, advantages, fixed_log_probs = \
                states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), fixed_log_probs[perm].clone()

            for i in range(optim_iter_num):
                ind = slice(i * self.optim_batch_size, min((i + 1) * self.optim_batch_size, states.shape[0]))
                states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                    states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

                self.ppo_step(states_b, actions_b, returns_b, advantages_b, fixed_log_probs_b)

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
        log_probs = self.agent.policy.get_log_prob(states, actions)
        ratio = torch.exp(log_probs - fixed_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.args.clip_epsilon, 1.0 + self.args.clip_epsilon) * advantages
        policy_surr = -torch.min(surr1, surr2).mean()
        self.agent.policy_optimizer.zero_grad()
        policy_surr.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.policy.parameters(), 40)
        self.agent.policy_optimizer.step()