import gym
import numpy as np
import stable_baselines3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
# import mujoco_py
import torch
from stable_baselines3.common.evaluation import evaluate_policy
import imageio
import os
from stable_baselines3.common.callbacks import EvalCallback

from stable_baselines3.common.env_util import make_vec_env
from gym_minigrid.wrappers import FlatObsWrapper, ImgObsWrapper, RGBImgObsWrapper, FullyObsWrapper
from envs.obs_wrapper import ImgFlatObsWrapper
import ipdb
import utils.data_sepsis as data_sepsis

# --------------
# args
# --------------

left_interval = 5
right_interval = 1
batch_size = 32
proto_num = 4
load_test_goal = 'sepsis_val.csv'
load_goal = 'sepsis_train.csv'

data_path = '/data/sepsis/'
ipdb.set_trace()

# --------------
# load demons
# --------------
train_batched_demon = data_sepsis.load_batched_sepsis(load_goal, data_path,batch_size=batch_size,
                                           left_interval=left_interval, right_interval=right_interval, cluster_num=proto_num)

test_batched_demon = data_sepsis.load_batched_sepsis(load_test_goal, data_path,batch_size=batch_size,
                                           left_interval=left_interval, right_interval=right_interval, cluster_num=proto_num)



print("finished")
