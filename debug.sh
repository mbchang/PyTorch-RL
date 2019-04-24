#!/bin/bash
python examples/ppo_gym.py --env-name Ant-v3 --debug --policy composite --vwght "0 0" --multitask --for-transfer --tasks 12_34