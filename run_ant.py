import itertools
import os


"""
3/17/19
Adam vs SGD
Actor Learning Rate
Critic Learning Rate

Conclusion: 
eplen of 200 is better for debugging. 
Before with eplen of 10000, since the episode was so long, it took an extremely long time for the ant to learn because of the credit assignment problem. It saturated max reward around 6000, but most of this is just from survival.
By using eplen of 200, it saturates the survival reward at 200, after which in order to get larger reward it has to start moving forward.
So what is likely a better strategy is to train for 200 but test on 10000.
"""
# optimizer = ['sgd', 'adam']
# plr = [5e-5]
# clr = [5e-4, 5e-5]
# envs = ['Ant-v3']
# eplen = [200, 1000]

# outputdir = 'runs/ant_test_optimizer_lr_eplen'

# gpu = True
# num_gpus = 2
# i = 0

# if gpu:
#     os.system('export OMP_NUM_THREADS=1')

# for o, p, c, e, l in itertools.product(optimizer, plr, clr, envs, eplen):
#     prefix = 'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
#     command = 'python examples/ppo_gym.py --opt {} --plr {} --clr {} --env-name {} --maxeplen {}'.format(o, p, c, e, l)
#     command += ' --outputdir {}'.format(outputdir)
#     command += ' --printf'
#     command += ' &'
#     print(prefix + command)
#     os.system(prefix + command)
#     i += 1
#     if i >= num_gpus:
#         i = 0


"""
3/18/19
Take best of test run.
"""
optimizer = ['sgd', 'adam']
plr = [5e-5]
clr = [5e-4, 5e-5]
envs = ['Ant-v3']
eplen = [200]

outputdir = 'runs/ant_test_optimizer_lr_eplen_testbest'

gpu = False
num_gpus = 2
i = 0

if gpu:
    os.system('export OMP_NUM_THREADS=1')

for o, p, c, e, l in itertools.product(optimizer, plr, clr, envs, eplen):
    prefix = 'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
    command = 'python examples/ppo_gym.py --opt {} --plr {} --clr {} --env-name {} --maxeplen {}'.format(o, p, c, e, l)
    command += ' --outputdir {}'.format(outputdir)
    command += ' --printf'
    command += ' &'
    print(prefix + command)
    # os.system(prefix + command)
    i += 1
    if i >= num_gpus:
        i = 0