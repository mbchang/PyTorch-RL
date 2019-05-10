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
# optimizer = ['adam']
# plr = [1e-5]
# clr = [1e-4, 1e-5]
# envs = ['Ant-v3']
# policy = ['primitive']
# eplen = [200]
# numtest = [100]
# vweights = ['1 0']
# goal_dist = [10]

# outputdir = 'runs/ant_test_optimizer_lr_gd_const_blog'

# gpu = True
# num_gpus = 2
# i = 0

# if gpu:
#     os.system('export OMP_NUM_THREADS=1')

# for o, p, c, e, l, n, vw, pi, gd in itertools.product(optimizer, plr, clr, envs, eplen, numtest, vweights, policy, goal_dist):
#     prefix = 'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
#     command = 'python examples/ppo_gym.py --opt {} --plr {} --clr {} --env-name {} --maxeplen {} --num-test {} --vwght \"{}\" --policy {} --goal-dist {}'.format(o, p, c, e, l, n, vw, pi, gd)
#     command += ' --outputdir {}'.format(outputdir)
#     command += ' --printf'
#     command += ' &'
#     print(prefix + command)
#     os.system(prefix + command)
#     i += 1
#     if i >= num_gpus:
#         i = 0


# # """
# # 3/31/19

# # normalized rewards

# # seems like the control cost is too big.
# # """
# optimizer = ['adam']
# plr = [1e-5]
# clr = [1e-4]
# envs = ['Ant-v3']
# policy = ['primitive']
# eplen = [200]
# numtest = [100]
# vweights = ['1 0']

# control_weight = [1e-3, 1e-2, 1e-1, 1]
# contact_weight = [0.01, 0.001]
# healthy_weight = [1,2]
# task_weight = [10]

# outputdir = 'runs/ant_test_optimizer_normalized_vel2'

# gpu = True
# num_gpus = 4
# i = 0

# if gpu:
#     os.system('export OMP_NUM_THREADS=1')

# for o, p, c, e, l, n, vw, pi, ctrl_wt, cnct_wt, h_wt, t_wt in itertools.product(optimizer, plr, clr, envs, eplen, numtest, vweights, policy, control_weight, contact_weight, healthy_weight, task_weight):
#     prefix = 'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
#     command = 'python examples/ppo_gym.py --opt {} --plr {} --clr {} --env-name {} --maxeplen {} --num-test {} --vwght \"{}\" --policy {} --control-weight {} --contact-weight {} --healthy-weight {} --task-weight {}'.format(o, p, c, e, l, n, vw, pi, ctrl_wt, cnct_wt, h_wt, t_wt)
#     command += ' --outputdir {}'.format(outputdir)
#     command += ' --printf'
#     command += ' &'
#     print(prefix + command)
#     os.system(prefix + command)
#     i += 1
#     if i >= num_gpus:
#         i = 0


# """
# # 4/2/19

# Making the costs all 0

# """
# optimizer = ['adam']
# plr = [1e-5]
# clr = [1e-4]
# envs = ['Ant-v3']
# policy = ['primitive']
# eplen = [200]
# numtest = [100]
# vweights = ['1 0']

# control_weight = [0]
# contact_weight = [0]
# healthy_weight = [0]
# task_weight = [1]
# task_scale = [1e-3, 1e-2, 1e-1, 1, 10, 100]

# outputdir = 'runs/ant_test_optimizer_normalized_vel3'

# gpu = True
# num_gpus = 8
# i = 0

# if gpu:
#     os.system('export OMP_NUM_THREADS=1')

# for o, p, c, e, l, n, vw, pi, ctrl_wt, cnct_wt, h_wt, t_wt, t_s in itertools.product(optimizer, plr, clr, envs, eplen, numtest, vweights, policy, control_weight, contact_weight, healthy_weight, task_weight, task_scale):
#     prefix = 'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
#     command = 'python examples/ppo_gym.py --opt {} --plr {} --clr {} --env-name {} --maxeplen {} --num-test {} --vwght \"{}\" --policy {} --control-weight {} --contact-weight {} --healthy-weight {} --task-weight {} --task-scale {}'.format(o, p, c, e, l, n, vw, pi, ctrl_wt, cnct_wt, h_wt, t_wt, t_s)
#     command += ' --outputdir {}'.format(outputdir)
#     command += ' --printf'
#     command += ' &'
#     print(prefix + command)
#     os.system(prefix + command)
#     i += 1
#     if i >= num_gpus:
#         i = 0


# """
# # 4/6/19

# Multitask for velocity

# Conclusion: Seemes like it is reasonable.

# """
# optimizer = ['adam']
# plr = [1e-5]
# clr = [1e-4]
# envs = ['Ant-v3']
# policy = ['primitive', 'composite']
# eplen = [200]
# numtest = [100]
# vweights = ['0 0']

# # control_weight = [0]
# # contact_weight = [0]
# # healthy_weight = [0]
# # task_weight = [1]
# # task_scale = [1e-3, 1e-2, 1e-1, 1, 10, 100]

# outputdir = 'runs/ant_test_multitask_vel'

# gpu = True
# num_gpus = 2
# i = 0

# if gpu:
#     os.system('export OMP_NUM_THREADS=1')

# for o, p, c, e, l, n, vw, pi in itertools.product(optimizer, plr, clr, envs, eplen, numtest, vweights, policy):
#     prefix = 'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
#     command = 'python examples/ppo_gym.py --opt {} --plr {} --clr {} --env-name {} --maxeplen {} --num-test {} --vwght \"{}\" --policy {}'.format(o, p, c, e, l, n, vw, pi)
#     command += ' --multitask'
#     command += ' --outputdir {}'.format(outputdir)
#     command += ' --printf'
#     command += ' &'
#     print(prefix + command)
#     # os.system(prefix + command)
#     i += 1
#     if i >= num_gpus:
#         i = 0


# """
# # 4/6/19

# Multitask for velocity, varying the primitives

# """
# optimizer = ['adam']
# plr = [1e-5]
# clr = [1e-4]
# envs = ['Ant-v3']
# policy = ['composite']
# eplen = [200]
# numtest = [100]
# vweights = ['0 0']
# nprims = [1, 2, 3, 4, 5, 6, 7, 8]

# outputdir = 'runs/ant_test_multitask_vel_nprims'

# gpu = True
# num_gpus = 8
# i = 0

# if gpu:
#     os.system('export OMP_NUM_THREADS=1')

# for o, p, c, e, l, n, vw, pi, np in itertools.product(optimizer, plr, clr, envs, eplen, numtest, vweights, policy, nprims):
#     prefix = 'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
#     command = 'python examples/ppo_gym.py --opt {} --plr {} --clr {} --env-name {} --maxeplen {} --num-test {} --vwght \"{}\" --policy {} --nprims {}'.format(o, p, c, e, l, n, vw, pi, np)
#     command += ' --multitask'
#     command += ' --outputdir {}'.format(outputdir)
#     command += ' --printf'
#     command += ' &'
#     print(prefix + command)
#     os.system(prefix + command)
#     i += 1
#     if i >= num_gpus:
#         i = 0


# """
# # 4/6/19

# Multitask for velocity, varying the primitives. Transfer

# """
# optimizer = ['adam']
# plr = [1e-5]
# clr = [1e-4]
# envs = ['Ant-v3']
# policy = ['composite']
# eplen = [200]
# numtest = [100]
# vweights = ['0 0']
# nprims = [2, 4, 6, 8]

# outputdir = 'runs/ant_test_multitask_vel_nprims_transfer_fp'

# gpu = True
# num_gpus = 2
# i = 0

# if gpu:
#     os.system('export OMP_NUM_THREADS=1')

# for o, p, c, e, l, n, vw, pi, np in itertools.product(optimizer, plr, clr, envs, eplen, numtest, vweights, policy, nprims):
#     prefix = 'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
#     command = 'python examples/ppo_gym.py --opt {} --plr {} --clr {} --env-name {} --maxeplen {} --num-test {} --vwght \"{}\" --policy {} --nprims {}'.format(o, p, c, e, l, n, vw, pi, np)
#     command += ' --for-transfer'
#     command += ' --multitask'
#     command += ' --outputdir {}'.format(outputdir)
#     command += ' --printf'
#     command += ' &'
#     print(prefix + command)
#     os.system(prefix + command)
#     i += 1
#     if i >= num_gpus:
#         i = 0


# """
# # 4/6/19

# Multitask for velocity, for primitive. Transfer

# """
# optimizer = ['adam']
# plr = [1e-5]
# clr = [1e-4]
# envs = ['Ant-v3']
# policy = ['primitive']
# eplen = [200]
# numtest = [100]
# vweights = ['0 0']
# # nprims = [2, 4, 6, 8]

# outputdir = 'runs/ant_test_multitask_vel_prim_transfer'

# gpu = True
# num_gpus = 2
# i = 0

# if gpu:
#     os.environ['OMP_NUM_THREADS'] = '1'

# for o, p, c, e, l, n, vw, pi in itertools.product(optimizer, plr, clr, envs, eplen, numtest, vweights, policy):
#     prefix = 'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
#     command = 'python examples/ppo_gym.py --opt {} --plr {} --clr {} --env-name {} --maxeplen {} --num-test {} --vwght \"{}\" --policy {}'.format(o, p, c, e, l, n, vw, pi)
#     command += ' --for-transfer'
#     command += ' --multitask'
#     command += ' --outputdir {}'.format(outputdir)
#     command += ' --printf'
#     command += ' &'
#     print(prefix + command)
#     os.system(prefix + command)
#     i += 1
#     if i >= num_gpus:
#         i = 0


# """
# # 4/10/19

# Multitask for velocity, for primitive. Transfer

# """
# optimizer = ['adam']
# plr = [1e-5]
# clr = [1e-4]
# envs = ['Ant-v3']
# policy = ['primitive']
# eplen = [200]
# numtest = [100]
# vweights = ['0 0']
# # nprims = [2, 4, 6, 8]
# tasks = ['1_234', '12_34', '13_24', '123_4']

# outputdir = 'runs/ant_test_multitask_vel_prim_transfer'

# gpu = True
# num_gpus = 4
# i = 0

# if gpu:
#     os.environ['OMP_NUM_THREADS'] = '1'

# for o, p, c, e, l, n, vw, pi, t in itertools.product(optimizer, plr, clr, envs, eplen, numtest, vweights, policy, tasks):
#     prefix = 'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
#     command = 'python examples/ppo_gym.py --opt {} --plr {} --clr {} --env-name {} --maxeplen {} --num-test {} --vwght \"{}\" --policy {} --tasks {}'.format(o, p, c, e, l, n, vw, pi, t)
#     command += ' --for-transfer'
#     command += ' --multitask'
#     command += ' --outputdir {}'.format(outputdir)
#     command += ' --printf'
#     command += ' &'
#     print(prefix + command)
#     # os.system(prefix + command)
#     i += 1
#     if i >= num_gpus:
#         i = 0



"""
# # 4/6/19

# Multitask for velocity, varying the primitives. Transfer Continuous

"""
# optimizer = ['adam']
# plr = [1e-5]
# clr = [1e-4]
# envs = ['Ant-v3']
# policy = ['composite']
# eplen = [200]
# numtest = [100]
# vweights = ['0 0']
# nprims = [4]
# tasks = ['1_234', '12_34', '13_24', '123_4']

# outputdir = 'runs/ant_test_multitask_vel_comp_cont_transfer'

# gpu = True
# num_gpus = 4
# i = 0

# if gpu:
#     os.system('export OMP_NUM_THREADS=1')

# for o, p, c, e, l, n, vw, pi, np, t in itertools.product(optimizer, plr, clr, envs, eplen, numtest, vweights, policy, nprims, tasks):
#     prefix = 'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
#     command = 'python examples/ppo_gym.py --opt {} --plr {} --clr {} --env-name {} --maxeplen {} --num-test {} --vwght \"{}\" --policy {} --nprims {} --tasks {}'.format(o, p, c, e, l, n, vw, pi, np, t)
#     command += ' --for-transfer'
#     command += ' --multitask'
#     command += ' --outputdir {}'.format(outputdir)
#     command += ' --printf'
#     command += ' &'
#     print(prefix + command)
#     # os.system(prefix + command)
#     i += 1
#     if i >= num_gpus:
#         i = 0


# """
# # # 4/15/19

# # CompositeTransferPolicy

# """
# optimizer = ['adam']
# plr = [1e-5]
# clr = [1e-4]
# envs = ['Ant-v3']
# policy = ['composite']
# eplen = [200]
# numtest = [100]
# vweights = ['0 0']
# nprims = [4]
# tasks = ['1_234', '12_34', '13_24', '123_4']
# nsamp = 5

# outputdir = 'runs/ant_test_multitask_vel_comptransfer_cont'

# gpu = True
# num_gpus = 2
# i = 0

# if gpu:
#     os.system('export OMP_NUM_THREADS=1')

# for o, p, c, e, l, n, vw, pi, np, t in itertools.product(optimizer, plr, clr, envs, eplen, numtest, vweights, policy, nprims, tasks):
#     prefix = 'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
#     command = 'python examples/ppo_gym.py --opt {} --plr {} --clr {} --env-name {} --maxeplen {} --num-test {} --vwght \"{}\" --policy {} --nprims {} --tasks {} --nsamp {}'.format(o, p, c, e, l, n, vw, pi, np, t, nsamp)
#     command += ' --for-transfer'
#     command += ' --multitask'
#     command += ' --outputdir {}'.format(outputdir)
#     command += ' --printf'
#     command += ' &'
#     print(prefix + command)
#     # os.system(prefix + command)
#     i += 1
#     if i >= num_gpus:
#         i = 0


# """
# # # 4/17/19

# # CompositeTransferPolicy with sampled weights and entropy loss
# # wef doesn't work for 1e-4 (weight std blows up)

# """
# optimizer = ['adam']
# plr = [1e-5]
# clr = [1e-4]
# envs = ['Ant-v3']
# policy = ['composite']
# eplen = [200]
# numtest = [100]
# vweights = ['0 0']
# nprims = [4]
# tasks = ['1_234', '12_34', '13_24', '123_4']
# nsamp = 2
# wef = [0.001, 0.005]

# outputdir = 'runs/ant_test_multitask_vel_comptransfer_cont_we'

# gpu = True
# num_gpus = 2
# i = 0

# if gpu:
#     os.system('export OMP_NUM_THREADS=1')

# for o, p, c, e, l, n, vw, pi, np, t, we in itertools.product(optimizer, plr, clr, envs, eplen, numtest, vweights, policy, nprims, tasks, wef):
#     prefix = 'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
#     command = 'python examples/ppo_gym.py --opt {} --plr {} --clr {} --env-name {} --maxeplen {} --num-test {} --vwght \"{}\" --policy {} --nprims {} --tasks {} --nsamp {} --weight-entropy-coeff {}'.format(o, p, c, e, l, n, vw, pi, np, t, nsamp, we)
#     command += ' --for-transfer'
#     command += ' --multitask'
#     command += ' --outputdir {}'.format(outputdir)
#     command += ' --printf'
#     command += ' &'
#     print(prefix + command)
#     # os.system(prefix + command)
#     i += 1
#     if i >= num_gpus:
#         i = 0


# """
# # # 4/18/19

# # CompositeTransferPolicy with sampled weights and entropy loss
# # try wef 0 and 1e-5

# """
# optimizer = ['adam']
# plr = [1e-5]
# clr = [1e-4]
# envs = ['Ant-v3']
# policy = ['composite']
# eplen = [200]
# numtest = [100]
# vweights = ['0 0']
# nprims = [4]
# tasks = ['1_234', '12_34', '13_24', '123_4']
# nsamp = 2
# wef = [0, 1e-5]

# outputdir = 'runs/ant_test_multitask_vel_comptransfer_cont_we2'

# gpu = True
# num_gpus = 2
# i = 0

# if gpu:
#     os.system('export OMP_NUM_THREADS=1')

# for o, p, c, e, l, n, vw, pi, np, t, we in itertools.product(optimizer, plr, clr, envs, eplen, numtest, vweights, policy, nprims, tasks, wef):
#     prefix = 'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
#     command = 'python examples/ppo_gym.py --opt {} --plr {} --clr {} --env-name {} --maxeplen {} --num-test {} --vwght \"{}\" --policy {} --nprims {} --tasks {} --nsamp {} --weight-entropy-coeff {}'.format(o, p, c, e, l, n, vw, pi, np, t, nsamp, we)
#     command += ' --for-transfer'
#     command += ' --multitask'
#     command += ' --outputdir {}'.format(outputdir)
#     command += ' --printf'
#     command += ' &'
#     print(prefix + command)
#     # os.system(prefix + command)
#     i += 1
#     if i >= num_gpus:
#         i = 0


# """
# # # 4/21/19

# # Latent Policy Transfer

# """
# optimizer = ['adam']
# plr = [1e-5]
# clr = [1e-4]
# envs = ['Ant-v3']
# policy = ['latent']
# eplen = [200]
# numtest = [100]
# vweights = ['0 0']
# nprims = [2, 4]
# tasks = ['123_4']
# nsamp = 2
# klps = [0.0002, 0.0001]

# outputdir = 'runs/ant_test_multitask_vel_latenttransfer_cont'

# gpu = True
# num_gpus = 2
# i = 1

# if gpu:
#     os.system('export OMP_NUM_THREADS=1')

# for o, p, c, e, l, n, vw, pi, np, t, klp in itertools.product(optimizer, plr, clr, envs, eplen, numtest, vweights, policy, nprims, tasks, klps):
#     prefix = 'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
#     command = 'python examples/ppo_gym.py --opt {} --plr {} --clr {} --env-name {} --maxeplen {} --num-test {} --vwght \"{}\" --policy {} --nprims {} --tasks {} --nsamp {} --klp {}'.format(o, p, c, e, l, n, vw, pi, np, t, nsamp, klp)
#     command += ' --for-transfer'
#     command += ' --multitask'
#     command += ' --outputdir {}'.format(outputdir)
#     command += ' --printf'
#     command += ' &'
#     print(prefix + command)
#     os.system(prefix + command)
#     # i += 1
#     # if i >= num_gpus:
#     #     i = 0



"""
# # 4/18/19

# CompositeTransferPolicy with sampled weights and entropy loss
# 5e-4

"""
# optimizer = ['adam']
# plr = [1e-5]
# clr = [1e-4]
# envs = ['Ant-v3']
# policy = ['composite']
# eplen = [200]
# numtest = [100]
# vweights = ['0 0']
# nprims = [2]
# tasks = ['123_4']
# nsamp = 2
# wef = [1e-3]
# # action_penalty = [1e-3, 1e-4]

# outputdir = 'runs/ant_test_multitask_vel_comptransfer_cont_we3'

# gpu = True
# num_gpus = 2
# i = 0

# if gpu:
#     os.system('export OMP_NUM_THREADS=1')

# for o, p, c, e, l, n, vw, pi, np, t, we in itertools.product(optimizer, plr, clr, envs, eplen, numtest, vweights, policy, nprims, tasks, wef):
#     prefix = 'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
#     command = 'python examples/ppo_gym.py --opt {} --plr {} --clr {} --env-name {} --maxeplen {} --num-test {} --vwght \"{}\" --policy {} --nprims {} --tasks {} --nsamp {} --weight-entropy-coeff {}'.format(o, p, c, e, l, n, vw, pi, np, t, nsamp, we)
#     command += ' --for-transfer'
#     command += ' --multitask'
#     command += ' --outputdir {}'.format(outputdir)
#     command += ' --printf'
#     command += ' &'
#     print(prefix + command)
#     # os.system(prefix + command)
#     # i += 1
#     # if i >= num_gpus:
#     #     i = 0



# # """
# # # # 4/23/19

# # # Simplify args; same functionality as before.

# # """
# policy = ['composite', 'latent', 'primitive']
# nprims = [4]
# tasks = ['123_4']
# wef = [0.0004]
# seeds = [0, 1, 2]
# outputdir = 'runs/ant_workshop'

# gpu = True
# num_gpus = 1
# i = 0

# if gpu:
#     os.system('export OMP_NUM_THREADS=1')

# for pi, np, t, we, s in itertools.product(policy, nprims, tasks, wef, seeds):
#     prefix = 'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
#     command = 'python examples/ppo_gym.py --policy {} --nprims {} --tasks {} --weight-entropy-coeff {} --seed {}'.format(pi, np, t, we, s)
#     command += ' --for-transfer'
#     command += ' --outputdir {}'.format(outputdir)
#     command += ' --printf'
#     command += ' &'
#     print(prefix + command)
#     # os.system(prefix + command)
#     # i += 1
#     # if i >= num_gpus:
#     #     i = 0




# # """
# # # # 4/23/19

# # Testing different wef

# # """
# policy = ['composite']
# nprims = [4]
# tasks = ['123_4']
# wef = [0.0001, 0.0002]#[0.0005, 0.0003]
# seeds = [0, 1, 2]
# outputdir = 'runs/ant_workshop'

# gpu = True
# num_gpus = 1
# i = 0

# if gpu:
#     os.system('export OMP_NUM_THREADS=1')

# for pi, np, t, we, s in itertools.product(policy, nprims, tasks, wef, seeds):
#     prefix = 'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
#     command = 'python examples/ppo_gym.py --policy {} --nprims {} --tasks {} --weight-entropy-coeff {} --seed {}'.format(pi, np, t, we, s)
#     command += ' --for-transfer'
#     command += ' --outputdir {}'.format(outputdir)
#     command += ' --printf'
#     command += ' &'
#     print(prefix + command)
#     # os.system(prefix + command)
#     # i += 1
#     # if i >= num_gpus:
#     #     i = 0


# """
# # # 4/23/19

# Testing different wef for primitive

# """
# policy = ['primitive']
# nprims = [4]
# tasks = ['4_4']
# wef = [0.0005]
# # seeds = [0, 1, 2]
# seeds = [1, 3]
# outputdir = 'runs/ant_workshop'

# gpu = True
# num_gpus = 1
# i = 0

# if gpu:
#     os.system('export OMP_NUM_THREADS=1')

# for pi, np, t, we, s in itertools.product(policy, nprims, tasks, wef, seeds):
#     prefix = ''#'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
#     command = 'python examples/ppo_gym.py --policy {} --nprims {} --tasks {} --weight-entropy-coeff {} --seed {}'.format(pi, np, t, we, s)
#     command += ' --for-transfer'
#     command += ' --outputdir {}'.format(outputdir)
#     command += ' --printf'
#     command += ' &'
#     print(prefix + command)
#     # os.system(prefix + command)
#     # i += 1
#     # if i >= num_gpus:
#     #     i = 0



# """
# 4/24/19
# 2 prims: compare differnt models
# """
# policy = ['composite', 'latent', 'primitive']
# nprims = [2]
# tasks = ['123_4']
# wef = [0.0003]
# seeds = [0, 1, 2]
# outputdir = 'runs/ant_workshop/p2'

# gpu = False
# num_gpus = 1
# i = 0

# os.system('export OMP_NUM_THREADS=1')

# for pi, np, t, we, s in itertools.product(policy, nprims, tasks, wef, seeds):
#     prefix = 'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
#     command = 'python examples/ppo_gym.py --policy {} --nprims {} --tasks {} --weight-entropy-coeff {} --seed {}'.format(pi, np, t, we, s)
#     command += ' --for-transfer'
#     command += ' --outputdir {}'.format(outputdir)
#     command += ' --printf'
#     command += ' &'
#     print(prefix + command)
#     # os.system(prefix + command)
#     # i += 1
#     # if i >= num_gpus:
#     #     i = 0


# """
# 4/24/19
# 2 prims: different wef
# """
# policy = ['composite']
# nprims = [2]
# tasks = ['123_4']
# wef = [0.0001, 0.0002]
# seeds = [0, 1, 2]
# outputdir = 'runs/ant_workshop/p2'

# gpu = False
# num_gpus = 1
# i = 0

# os.system('export OMP_NUM_THREADS=1')

# for pi, np, t, we, s in itertools.product(policy, nprims, tasks, wef, seeds):
#     prefix = 'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
#     command = 'python examples/ppo_gym.py --policy {} --nprims {} --tasks {} --weight-entropy-coeff {} --seed {}'.format(pi, np, t, we, s)
#     command += ' --for-transfer'
#     command += ' --outputdir {}'.format(outputdir)
#     command += ' --printf'
#     command += ' &'
#     print(prefix + command)
#     # os.system(prefix + command)
#     # i += 1
#     # if i >= num_gpus:
#     #     i = 0


# """
# 4/25/19

# Testing different wef
# hide state from weight network

# """
# policy = ['composite']
# nprims = [4]
# tasks = ['123_4']
# wef = [0.0002]
# seeds = [0, 1, 2]
# outputdir = 'runs/ant_workshop'

# gpu = False
# num_gpus = 1
# i = 0

# if gpu:
#     os.system('export OMP_NUM_THREADS=1')

# for pi, np, t, we, s in itertools.product(policy, nprims, tasks, wef, seeds):
#     prefix = 'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
#     command = 'python examples/ppo_gym.py --policy {} --nprims {} --tasks {} --weight-entropy-coeff {} --seed {}'.format(pi, np, t, we, s)
#     command += ' --hide-state'
#     command += ' --for-transfer'
#     command += ' --outputdir {}'.format(outputdir)
#     command += ' --printf'
#     command += ' &'
#     print(prefix + command)
#     # os.system(prefix + command)
#     # i += 1
#     # if i >= num_gpus:
#     #     i = 0


# """
# 4/27/19

# Have noise as part of the state
# Latent, Composite, 4 primitives

# """
# policy = ['latent', 'composite']
# nprims = [4]
# tasks = ['123_4']
# wef = [0]
# seeds = [0, 1, 2]
# outputdir = 'runs/ant_workshop_noise/p4'

# gpu = False
# num_gpus = 1
# i = 0

# if gpu:
#     os.system('export OMP_NUM_THREADS=1')

# for pi, np, t, we, s in itertools.product(policy, nprims, tasks, wef, seeds):
#     prefix = 'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
#     command = 'python examples/ppo_gym.py --policy {} --nprims {} --tasks {} --weight-entropy-coeff {} --seed {}'.format(pi, np, t, we, s)
#     if pi == 'composite':
#         command += ' --hide-state'
#     command += ' --for-transfer'
#     command += ' --outputdir {}'.format(outputdir)
#     command += ' --printf'
#     command += ' &'
#     print(prefix + command)
#     # os.system(prefix + command)
#     # i += 1
#     # if i >= num_gpus:
#     #     i = 0




# """
# 4/29/19

# Have noise as part of the state
# Latent, Composite, 4 primitives
# train for a long time

# """
# policy = ['primitive', 'latent', 'composite']
# nprims = [4]
# tasks = ['123_4']
# wef = [0]
# seeds = [0, 1, 2]
# outputdir = 'runs/ant_workshop_long'
# max_iter_num = 10000

# gpu = False
# num_gpus = 1
# i = 0

# if gpu:
#     os.system('export OMP_NUM_THREADS=1')

# for pi, np, t, we, s in itertools.product(policy, nprims, tasks, wef, seeds):
#     prefix = 'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
#     command = 'python examples/ppo_gym.py --policy {} --nprims {} --tasks {} --weight-entropy-coeff {} --seed {} --max-iter-num {}'.format(pi, np, t, we, s, max_iter_num)
#     if pi == 'composite':
#         command += ' --hide-state'
#     command += ' --for-transfer'
#     command += ' --outputdir {}'.format(outputdir)
#     command += ' --printf'
#     command += ' &'
#     print(prefix + command)
#     # os.system(prefix + command)
#     # i += 1
#     # if i >= num_gpus:
#     #     i = 0


# """
# # # 4/29/19

# From scratch
# Train for a long time

# """
# policy = ['primitive']
# nprims = [4]
# tasks = ['4_4']
# wef = [0]
# seeds = [0, 1, 2]
# outputdir = 'runs/ant_workshop_long'
# max_iter_num = 10000

# gpu = False
# num_gpus = 1
# i = 0

# if gpu:
#     os.system('export OMP_NUM_THREADS=1')

# for pi, np, t, we, s in itertools.product(policy, nprims, tasks, wef, seeds):
#     prefix = ''#'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
#     command = 'python examples/ppo_gym.py --policy {} --nprims {} --tasks {} --weight-entropy-coeff {} --seed {} --max-iter-num {}'.format(pi, np, t, we, s, max_iter_num)
#     command += ' --for-transfer'
#     command += ' --outputdir {}'.format(outputdir)
#     command += ' --printf'
#     command += ' &'
#     print(prefix + command)
#     # os.system(prefix + command)
#     # i += 1
#     # if i >= num_gpus:
#     #     i = 0





# """
# 4/29/19

# Have noise as part of the state
# Latent, Composite, 4 primitives
# train for a long time
# interpolation extrapolation

# """
# policy = ['primitive', 'latent', 'composite']
# nprims = [4]
# tasks = ['12_3', '12_4']
# wef = [0]
# seeds = [0, 1, 2]
# outputdir = 'runs/ant_workshop_long_pg'
# max_iter_num = 10000

# gpu = False
# num_gpus = 1
# i = 0

# if gpu:
#     os.system('export OMP_NUM_THREADS=1')

# for pi, np, t, we, s in itertools.product(policy, nprims, tasks, wef, seeds):
#     prefix = 'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
#     command = 'python examples/ppo_gym.py --policy {} --nprims {} --tasks {} --weight-entropy-coeff {} --seed {} --max-iter-num {}'.format(pi, np, t, we, s, max_iter_num)
#     if pi == 'composite':
#         command += ' --hide-state'
#     command += ' --for-transfer'
#     command += ' --outputdir {}'.format(outputdir)
#     command += ' --printf'
#     command += ' &'
#     print(prefix + command)
#     # os.system(prefix + command)
#     # i += 1
#     # if i >= num_gpus:
#     #     i = 0


# """
# # # 4/29/19

# From scratch
# Train for a long time
# interpolation extrapolation

# """
# policy = ['primitive']
# nprims = [4]
# tasks = ['4_4', '3_3']
# wef = [0]
# seeds = [0, 1, 2]
# outputdir = 'runs/ant_workshop_long_pg'
# max_iter_num = 10000

# gpu = False
# num_gpus = 1
# i = 0

# if gpu:
#     os.system('export OMP_NUM_THREADS=1')

# for pi, np, t, we, s in itertools.product(policy, nprims, tasks, wef, seeds):
#     prefix = ''#'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
#     command = 'python examples/ppo_gym.py --policy {} --nprims {} --tasks {} --weight-entropy-coeff {} --seed {} --max-iter-num {}'.format(pi, np, t, we, s, max_iter_num)
#     command += ' --for-transfer'
#     command += ' --outputdir {}'.format(outputdir)
#     command += ' --printf'
#     command += ' &'
#     print(prefix + command)
#     # os.system(prefix + command)
#     # i += 1
#     # if i >= num_gpus:
#     #     i = 0




"""
5/5/19

Have noise as part of the state
Latent, Composite, 4 primitives
train for a long time

"""
policy = ['primitive', 'latent', 'composite']
nprims = [4]
tasks = ['123_4']
wef = [0]
seeds = [0, 1, 2]
outputdir = 'runs/ant_workshop_xlong'
max_iter_num = 20000

gpu = False
num_gpus = 1
i = 0

if gpu:
    os.system('export OMP_NUM_THREADS=1')

for pi, np, t, we, s in itertools.product(policy, nprims, tasks, wef, seeds):
    prefix = 'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
    command = 'python examples/ppo_gym.py --policy {} --nprims {} --tasks {} --weight-entropy-coeff {} --seed {} --max-iter-num {}'.format(pi, np, t, we, s, max_iter_num)
    if pi == 'composite':
        command += ' --hide-state'
    command += ' --for-transfer'
    command += ' --outputdir {}'.format(outputdir)
    command += ' --printf'
    command += ' &'
    print(prefix + command)
    # os.system(prefix + command)
    # i += 1
    # if i >= num_gpus:
    #     i = 0


"""
5/5/19

From scratch
Train for a long time

"""
policy = ['primitive']
nprims = [4]
tasks = ['4_4']
wef = [0]
seeds = [0, 1, 2]
outputdir = 'runs/ant_workshop_xlong'
max_iter_num = 20000

gpu = False
num_gpus = 1
i = 0

if gpu:
    os.system('export OMP_NUM_THREADS=1')

for pi, np, t, we, s in itertools.product(policy, nprims, tasks, wef, seeds):
    prefix = ''#'CUDA_VISIBLE_DEVICES={} '.format(i) if gpu else ''
    command = 'python examples/ppo_gym.py --policy {} --nprims {} --tasks {} --weight-entropy-coeff {} --seed {} --max-iter-num {}'.format(pi, np, t, we, s, max_iter_num)
    command += ' --for-transfer'
    command += ' --outputdir {}'.format(outputdir)
    command += ' --printf'
    command += ' &'
    print(prefix + command)
    # os.system(prefix + command)
    # i += 1
    # if i >= num_gpus:
    #     i = 0




# TODO: you need to set for_transfer to be true!