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
from gym.wrappers import FrameStack
import matplotlib.pyplot as plt

from imitation.algorithms import bc
from imitation.algorithms.adversarial import gail, airl
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
import ipdb
import utils.expert as exp
import json
from algorithms import skill_bc as skill_bc
from torch.utils.tensorboard import SummaryWriter
from envs.obs_wrapper import ImgFlatObsWrapper
from plot.plot_dist import plot_dist1D
import random
from envs.new_fourrooms import *
# --------------
# args
# --------------

env_name = 'MiniGrid-FourRooms-v0'

seed = 42
method = 'bi-modeling'
goal = [13, 16] # pos_x(width), pos_y(height)

load_goals = [[13, 16], [13, 5], [13, 12]]
# load_goals = [[13, 16]]
load_test_goals = [[13, 16]]
train_epoch = 30
left_interval = 5
right_interval = 1
batch_size = 32
lr = 0.0005
lambda_mi = 0
lambda_clus = 0
lambda_consist = 0
lambda_pseudo = 10
proto_num = 4
load_best = False # if finetune is based on the last model or the best model
use_log = True
save_model = True # if save the learned policy model for further analysis
batch_mi = False
action_in_mi = True
distance='euc'
checkpoint_name = 'best_pure_pseu.pth'

if goal is not None:
    env_kwargs = {'goal_pos': goal}
else:
    env_kwargs = None

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# analyze lambda_mi
lrs = [0.0005]
batched_performance = []
# analyze batch size
for lr in lrs:
    env = gym.make(env_name, **env_kwargs)
    env = Monitor(env)
    env = ImgFlatObsWrapper(env)

    tensorboard_log = 'logs_new/{}_{}/load{}train{}/lr{}_mi{}_clus{}consis{}_bs{}_loadbest{}'.format(env_name, method, load_goals,
                                                                                         train_epoch,
                                                                                         lr, lambda_mi, lambda_clus,
                                                                                         lambda_consist,
                                                                                         batch_size, load_best)
    os.makedirs(tensorboard_log, exist_ok=True)
    if use_log:
        logger = SummaryWriter(tensorboard_log)
    else:
        logger = None

    # ---------------------
    # collect trajectories
    # ---------------------
    traj_dir = 'traj/{}'.format(env_name)
    batched_demon = exp.load_batched_demon(load_goals=load_goals, traj_dir=traj_dir, batch_size=batch_size,
                                           left_interval=left_interval, right_interval=right_interval, cluster_num=proto_num)
    test_batched_demons = []
    for load_test_goal in load_test_goals:
        test_batched_demon = exp.load_batched_demon(load_goals=[load_test_goal], traj_dir=traj_dir, batch_size=batch_size,
                                                left_interval=left_interval, right_interval=right_interval)
        test_batched_demons.append(test_batched_demon)

    exp_batched_demon = exp.load_batched_demon(load_goals=[load_goals[0]], traj_dir=traj_dir, batch_size=batch_size,
                                           left_interval=left_interval, right_interval=right_interval)


    # ----------------------
    # construct policy model
    # ----------------------
    if method == 'bi-modeling':
        alg_args = {'batch_size': 32, 'lr': lr, 'interval': left_interval, 'future_interval': right_interval,
                    'dropout': 0,
                    'hid_dim': 32, 'proto_dim': 16, 'proto_num': proto_num, 'hard_sel': True, 'distance': distance,
                    'task': 'cls',
                    'lambda_mi': lambda_mi, 'lambda_clus': lambda_clus,'lambda_consist': lambda_consist,
                    'batch_mi': batch_mi, 'lambda_pseudo': lambda_pseudo,}
        trainer = skill_bc.SkillBC(observation_space=env.observation_space, action_space=env.action_space,
                                   demonstrations=batched_demon, alg_args=alg_args, load_best=load_best,
                                   test_demonstrations=test_batched_demons, logger=logger, action_in_mi=action_in_mi)
        learner = trainer

    # ----------------------
    # learn model
    # ----------------------

    trainer.train_all(epochs=train_epoch, use_bi=True,
                      eval_env=FrameStack(env, num_stack=left_interval), prefix='pretrain')
    trainer.pre_trained = True
    best_reward = trainer.train_all(epochs=train_epoch, use_bi=False, opt_all=False,
                                     eval_env=FrameStack(env, num_stack=left_interval), prefix='adapt',)
    print('best reward: {}'.format(best_reward))
    best_model = trainer.best_content

    save_path = 'logs/{}'.format(checkpoint_name)
    torch.save(best_model, save_path)

    '''
    if True:
        trainer.demonstrations = exp_batched_demon

    best_reward_finetuned = trainer.train_all(epochs=train_epoch, use_bi=False,
                                              eval_env =FrameStack(env, num_stack=left_interval), prefix='finetune')
    print('best reward after finetune: {}'.format(best_reward))

    rewards = [best_reward, best_reward_finetuned]
    batched_performance.append(rewards)

    if save_model:
        if best_reward_finetuned > best_reward:
            best_model = trainer.best_content
        save_path = 'logs/{}'.format(checkpoint_name)
        torch.save(best_model, save_path)
        ipdb.set_trace()


for i, rewards in enumerate(batched_performance):
    print('performance of setting {}: {}'.format(i, rewards))

    '''


