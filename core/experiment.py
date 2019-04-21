from moviepy.editor import ImageSequenceClip
import os
import pickle
import time

from infra.log import display_stats, merge_log
from utils import *

class Experiment():
    def __init__(self, agent, env, rl_alg, logger, running_state, args):
        self.agent = agent
        self.env = env
        self.rl_alg = rl_alg
        self.logger = logger
        self.running_state = running_state
        self.args = args

    def visualize(self, policy_name, i_episode, episode_data, mode, sample_num):
        frames = np.array([e['frame'] for e in episode_data])
        if self.env.env.multitask:
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
                    open(os.path.join(assets_dir(), 'learned_models/{}_ppo.p'.format(self.args.env_name)), 'wb'))
        to_device(self.agent.device, self.agent.policy, self.agent.valuefn)

    def test(self, policy, i_iter, hide_goal, sample_num=0):
        to_device(torch.device('cpu'), policy)
        with torch.no_grad():
            test_batch, test_log = self.agent.collect_samples(
                policy=policy,
                min_batch_size=self.args.min_batch_size, 
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
        for i_iter in range(self.args.max_iter_num+1):
            self.logger.update_variable(name='i_iter', index=i_iter, value=i_iter)
            should_log = i_iter % self.args.log_every == 0
            should_save = i_iter % self.args.save_every == 0
            should_visualize = i_iter % self.args.visualize_every == 0

            if should_visualize:
                for sn in range(self.args.nsamp):
                    self.test(policy=self.agent.policy, i_iter=i_iter, hide_goal=False, sample_num=sn)
                if self.args.policy == 'composite':
                    for p in self.agent.policy.primitives:
                        self.test(policy=p, i_iter=i_iter, hide_goal=True)

            """generate multiple trajectories that reach the minimum batch_size"""
            batch, log = self.agent.collect_samples(
                policy=self.agent.policy, 
                min_batch_size=self.args.min_batch_size)  # here you can record the action mean and std

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