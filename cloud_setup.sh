#!/bin/bash
mkdir -p runs assets/learned_models
rsync -avxzr custom_envs/ant_v3*.py custom_envs/mujoco_env.py /home/michaelchang/.miniconda3/envs/pt10/lib/python3.6/site-packages/gym/envs/mujoco/
rsync -avxzr custom_envs/assets/ant.xml /home/michaelchang/.miniconda3/envs/pt10/lib/python3.6/site-packages/gym/envs/mujoco/assets/
