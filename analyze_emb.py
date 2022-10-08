#

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
from gym_minigrid.wrappers import FlatObsWrapper, ImgObsWrapper, RGBImgObsWrapper
import matplotlib.pyplot as plt
from gym.wrappers import FrameStack
from copy import deepcopy
from imitation.algorithms import bc
from imitation.algorithms import adversarial
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
import ipdb
import utils.expert as exp
import json
import random

from plot.plot_dist import plot_dist1D
from plot.plot_minigrid import plot_grid
from algorithms import skill_bc_new as skill_bc

from envs.new_fourrooms import *

from algorithms.skill_bc import tensor_distance_cos, tensor_distance_euc


env_name = 'MiniGrid-FourRooms-v0'
load_goals = [[13, 16], [13, 5], [13, 12]]
left_interval = 5
right_interval = 1
batch_size = 32
distance_func = tensor_distance_euc

# analyze learned parameters
load_goal = "best_model"
load_path = "logs/{}.pth".format(load_goal)
best_model = torch.load(load_path)
memory = best_model['model4']['proto_memory']
print(distance_func(memory, memory))