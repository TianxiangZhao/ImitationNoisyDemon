# visualize trajectories in MiniGrid

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
from copy import deepcopy
from imitation.algorithms import bc
from imitation.algorithms import adversarial
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
import ipdb
import utils.expert as exp
import json

from plot.plot_dist import plot_dist1D
from plot.plot_minigrid import plot_grid

# --------------
# args
# --------------

env_name = 'MiniGrid-FourRooms-v0'
#env_name = 'MiniGrid-DoorKey-8x8-v0'
FIX_MAP = True
FIX_TASK = True
goal = [13, 12]

load_goal = [13, 12]
seed = 42

# ---------------------
# env configuration
# ---------------------
os.environ["SDL_VIDEODRIVER"] = "dummy"

img_ct = 0

def vis_env(env):
    global img_ct
    img = env.render(mode='rgb_array')
    plt.imsave('imgs/env_set{}.jpg'.format(img_ct), img)
    img_ct += 1

# env = gym.make(env_name)
# env = FlatObsWrapper(env)
if FIX_TASK:
    env_kwargs = {'goal_pos': goal}
else:
    env_kwargs = None
env = make_vec_env(env_name,n_envs=1,wrapper_class=FlatObsWrapper, seed=seed,env_kwargs=env_kwargs)

env.reset()
vis_env(env)
if FIX_MAP:
    state_ini = exp.get_env_state(env)
else:
    state_ini = None
exp.set_env_state(env, state_ini)
vis_env(env)

# ---------------------
# collect episodes
# ---------------------
# plt.imsave('imgs/vis.jpg',env.gen_obs()['image'])
load_path = 'data/{}'.format(env_name)

if load_goal is not None:
    expert = PPO.load(load_path+'_goal{}/'.format(load_goal)+'model_'+str(seed)+'.pth', custom_objects={"learning_rate":0.0, "lr_schedule": lambda _:0.0, "clip_range": lambda _:0.0})
else:
    expert = PPO.load(load_path+'/model_'+str(seed)+'.pth', custom_objects={"learning_rate":0.0, "lr_schedule": lambda _:0.0, "clip_range": lambda _:0.0})

mean_reward, std_reward = evaluate_policy(expert, env, n_eval_episodes=100)
print(f"expert mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# noisy_expert = exp.model_add_uniform_noise(expert, noise_level=noise)

# collect visualization trajectory
trajPos, trajDirect, trajR, trajAct = exp.collect_Grid_trajectories(expert, env, max_length=128, traj_num=100, state_ini=state_ini)

fig = plot_dist1D(np.array(trajR), 'Rewards distribution')
plt.savefig('imgs/rewardDist{}.jpg'.format(env_name))

# group episodes
idx_lists = [(np.array(trajR)>0.7).nonzero()[0], (np.array(trajR)<0.2).nonzero()[0]]

# ---------------------
# visualization
# ---------------------
if 'VecEnv' in type(env).__name__:
    height, width = env.envs[0].grid.height, env.envs[0].grid.width
    vis_grid = env.envs[0].grid.grid
else:
    height, width = env.env.grid.height, env.env.grid.width
    vis_grid = env.env.grid.grid

assert 'Grid' in env_name, "only support MiniGrid environment for now"
for group_id, idx_list in enumerate(idx_lists):
    move_array = np.zeros((height, width, 4)) #
    for idx in idx_list:
        pos_traj = trajPos[idx]
        dir_traj = trajDirect[idx]
        act_traj = trajAct[idx]

        for move in (act_traj==2).nonzero()[0]:
            pos = pos_traj[move] # (column, row)
            dir = dir_traj[move]
            move_array[pos[1], pos[0], int(dir)] += 1

    if state_ini is not None:
        map_grid = state_ini.grid.grid # list of grid items, with length (height * width)
    else:
        map_grid = vis_grid #

    fig = plot_grid(height, width, map_grid, move_array)

    plt.imsave('imgs/ActionDist{}goal{}_load{}_{}.jpg'.format(env_name,goal,load_goal, group_id),fig)


# plot episode