import itertools
import os


"""
3/17/19
Adam vs SGD
Actor Learning Rate
Critic Learning Rate
"""
optimizer = ['sgd', 'adam']
plr = [5e-5]
clr = [5e-4, 5e-5]
envs = ['Ant-v3']
eplen = [200, 1000]

outputdir = 'runs/ant_test_optimizer_lr_eplen'

gpu = True
num_gpus = 2
i = 0
for o, p, c, e, l in itertools.product(optimizer, plr, clr, envs, eplen):
    prefix = 'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
    command = 'python examples/ppo_gym.py --opt {} --plr {} --clr {} --env-name {} --maxeplen {}'.format(o, p, c, e, l)
    command += ' --outputdir {}'.format(outputdir)
    command += ' --printf'
    command += ' &'
    print(prefix + command)
    os.system(prefix + command)
    i += 1
    if i >= num_gpus:
        i = 0