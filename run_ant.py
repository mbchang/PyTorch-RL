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


# """
# 3/18/19
# Take best of test run.

# It works now, when I use the collect_samples() to test. I wonder wny.
# """
# optimizer = ['adam']
# plr = [5e-5]
# clr = [5e-4, 5e-5]
# envs = ['Ant-v3']
# eplen = [200]
# numtest = [4096]

# outputdir = 'runs/ant_test_optimizer_lr_eplen_testbest2'

# gpu = True
# num_gpus = 2
# i = 0

# if gpu:
#     os.system('export OMP_NUM_THREADS=1')

# for o, p, c, e, l, n in itertools.product(optimizer, plr, clr, envs, eplen, numtest):
#     prefix = 'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
#     command = 'python examples/ppo_gym.py --opt {} --plr {} --clr {} --env-name {} --maxeplen {} --num-test {}'.format(o, p, c, e, l, n)
#     command += ' --outputdir {}'.format(outputdir)
#     command += ' --printf'
#     command += ' &'
#     print(prefix + command)
#     # os.system(prefix + command)
#     i += 1
#     if i >= num_gpus:
#         i = 0


"""
3/20/19
4 directions Ant
"""
# optimizer = ['adam']
# plr = [5e-5]
# clr = [5e-4]
# envs = ['Ant-v3']
# eplen = [200]
# numtest = [100]
# vweights = ['1 0', '-1 0', '0 1', '0 -1']

# outputdir = 'runs/ant_test_optimizer_lr_vw'

# gpu = True
# num_gpus = 2
# i = 0

# if gpu:
#     os.system('export OMP_NUM_THREADS=1')

# for o, p, c, e, l, n, vw in itertools.product(optimizer, plr, clr, envs, eplen, numtest, vweights):
#     prefix = 'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
#     command = 'python examples/ppo_gym.py --opt {} --plr {} --clr {} --env-name {} --maxeplen {} --num-test {} --vwght \"{}\"'.format(o, p, c, e, l, n, vw)
#     command += ' --outputdir {}'.format(outputdir)
#     command += ' --printf'
#     command += ' &'
#     print(prefix + command)
#     # os.system(prefix + command)
#     i += 1
#     if i >= num_gpus:
#         i = 0

# """
# 3/23/19

# 1 direction Ant primitive
# learns. 1e-5 learns too slowly.
# """
# optimizer = ['adam']
# plr = [5e-5]
# clr = [5e-4]
# envs = ['Ant-v3']
# policy = ['primitive']
# eplen = [200]
# numtest = [100]
# vweights = ['1 0']

# outputdir = 'runs/ant_test_optimizer_lr_prim'

# gpu = True
# num_gpus = 2
# i = 0

# if gpu:
#     os.system('export OMP_NUM_THREADS=1')

# for o, p, c, e, l, n, vw in itertools.product(optimizer, plr, clr, envs, eplen, numtest, vweights):
#     prefix = 'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
#     command = 'python examples/ppo_gym.py --opt {} --plr {} --clr {} --env-name {} --maxeplen {} --num-test {} --vwght \"{}\"'.format(o, p, c, e, l, n, vw)
#     command += ' --outputdir {}'.format(outputdir)
#     command += ' --printf'
#     command += ' &'
#     print(prefix + command)
#     # os.system(prefix + command)
#     i += 1
#     if i >= num_gpus:
#         i = 0

"""
3/23/19

1 direction Ant primitive. With entropy and IB penalty
Pretty unstable. Trying to lower the learning rate.
# seems like 1e-5 and 1e-4 learns the most stably
"""
# optimizer = ['adam']
# plr = [5e-5, 1e-5]
# clr = [5e-4, 1e-4]
# envs = ['Ant-v3']
# policy = ['primitive']
# eplen = [200]
# numtest = [100]
# vweights = ['1 0']

# outputdir = 'runs/ant_test_optimizer_lr_prim_eib'

# gpu = True
# num_gpus = 2
# i = 0

# if gpu:
#     os.system('export OMP_NUM_THREADS=1')

# for o, p, c, e, l, n, vw, pi in itertools.product(optimizer, plr, clr, envs, eplen, numtest, vweights, policy):
#     prefix = 'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
#     command = 'python examples/ppo_gym.py --opt {} --plr {} --clr {} --env-name {} --maxeplen {} --num-test {} --vwght \"{}\" --policy {}'.format(o, p, c, e, l, n, vw, pi)
#     command += ' --outputdir {}'.format(outputdir)
#     command += ' --printf'
#     command += ' &'
#     print(prefix + command)
#     os.system(prefix + command)
#     i += 1
#     if i >= num_gpus:
#         i = 0


"""
3/23/19

1 direction Ant composite deterministic.
"""
# optimizer = ['adam']
# plr = [1e-5]
# clr = [5e-4, 1e-4]
# envs = ['Ant-v3']
# policy = ['composite']
# eplen = [200]
# numtest = [100]
# vweights = ['1 0']

# outputdir = 'runs/ant_test_optimizer_lr_compdet'

# gpu = True
# num_gpus = 2
# i = 0

# if gpu:
#     os.system('export OMP_NUM_THREADS=1')

# for o, p, c, e, l, n, vw, pi in itertools.product(optimizer, plr, clr, envs, eplen, numtest, vweights, policy):
#     prefix = 'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
#     command = 'python examples/ppo_gym.py --opt {} --plr {} --clr {} --env-name {} --maxeplen {} --num-test {} --vwght \"{}\" --policy {}'.format(o, p, c, e, l, n, vw, pi)
#     command += ' --outputdir {}'.format(outputdir)
#     command += ' --printf'
#     command += ' &'
#     print(prefix + command)
#     os.system(prefix + command)
#     i += 1
#     if i >= num_gpus:
#         i = 0


"""
3/28/19

1 goal Ant primitive deterministic.

3/31/19
Running it again, but this time add a constant to the reward.
"""
optimizer = ['adam']
plr = [1e-5]
clr = [1e-4, 1e-5]
envs = ['Ant-v3']
policy = ['primitive']
eplen = [200]
numtest = [100]
vweights = ['1 0']
goal_dist = [10]

outputdir = 'runs/ant_test_optimizer_lr_gd_const'

gpu = True
num_gpus = 2
i = 0

if gpu:
    os.system('export OMP_NUM_THREADS=1')

for o, p, c, e, l, n, vw, pi, gd in itertools.product(optimizer, plr, clr, envs, eplen, numtest, vweights, policy, goal_dist):
    prefix = 'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
    command = 'python examples/ppo_gym.py --opt {} --plr {} --clr {} --env-name {} --maxeplen {} --num-test {} --vwght \"{}\" --policy {} --goal-dist {}'.format(o, p, c, e, l, n, vw, pi, gd)
    command += ' --outputdir {}'.format(outputdir)
    command += ' --printf'
    command += ' &'
    print(prefix + command)
    os.system(prefix + command)
    i += 1
    if i >= num_gpus:
        i = 0

