import itertools
import os


"""
3/17/19
Adam vs SGD
Actor Learning Rate
Critic Learning Rate
"""
optimizer = ['sgd', 'adam']#, 'sgd']
plr = [5e-4, 5e-5]
clr = [5e-4]
envs = ['Ant-v3']

outputdir = 'runs/ant_test_optimizer_lr'

gpu = True
num_gpus = 2
i = 0
for o, p, c, e in itertools.product(optimizer, plr, clr, envs):
    prefix = 'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
    command = 'python examples/ppo_gym.py --opt {} --plr {} --clr {} --env-name {}'.format(o, p, c, e)
    command += ' --outputdir {}'.format(outputdir)
    command += ' --printf'
    command += ' &'
    print(prefix + command)
    os.system(prefix + command)
    i += 1
    if i >= num_gpus:
        i = 0