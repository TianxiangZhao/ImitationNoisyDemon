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
from gym_minigrid.wrappers import FlatObsWrapper, FullyObsWrapper, ImgObsWrapper
import matplotlib.pyplot as plt
from copy import deepcopy
from imitation.algorithms import bc
from imitation.algorithms import adversarial
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
import ipdb
import utils.expert as exp
import json
from envs.obs_wrapper import ImgFlatObsWrapper, IndexObsWrapper
from plot.plot_dist import plot_dist1D
from plot.plot_minigrid import plot_grid
from envs.new_fourrooms import *
import random
# --------------
# args
# --------------

#env_name = 'MiniGrid-FourRooms-v0'
#env_name = 'MiniGrid-FourRooms-v1'
env_name = 'MiniGrid-DistShift1-v0'
#env_name = 'MiniGrid-LavaGapS5-v0'
#env_name = 'MiniGrid-DoorKey-5x5-v0'
print(env_name)

FIX_MAP = False
#FIX_TASK = True
FIX_TASK = False
goal = [13, 5]

load_goal = [13, 5]
#load_goal = [13, 12]
#load_goal = None
seed = 4

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

if FIX_TASK:
    env_kwargs = {'goal_pos': goal}
    env = gym.make(env_name, **env_kwargs)
else:
    env_kwargs = None
    env = gym.make(env_name)
    load_goal = None
    goal = None


# env = FullyObsWrapper(env)
# env = FlatObsWrapper(env)
if 'FourRoom' in env_name:
    env = ImgFlatObsWrapper(env)
else:
    env=ImgFlatObsWrapper(env)

# env = make_vec_env(env_name,n_envs=1,wrapper_class=FlatObsWrapper, seed=seed,env_kwargs=env_kwargs)
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
    expert = PPO.load(load_path+'_goal{}_ImgFull/'.format(load_goal)+'model_'+str(seed)+'.pth', custom_objects={"learning_rate":0.0, "lr_schedule": lambda _:0.0, "clip_range": lambda _:0.0})
else:
    expert = PPO.load(load_path+'_Index/best_model', custom_objects={"learning_rate":0.0, "lr_schedule": lambda _:0.0, "clip_range": lambda _:0.0})

#mean_reward, std_reward = evaluate_policy(expert, env, n_eval_episodes=100)
#print(f"expert mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# noisy_expert = exp.model_add_uniform_noise(expert, noise_level=noise)

# collect visualization trajectory
# trajPos, trajDirect, trajR, trajAct = exp.collect_Grid_trajectories(expert, env, max_length=128, traj_num=100, state_ini=state_ini)
trajS, trajA, trajR, trajDone, trajDirect = exp.collect_trajectories(expert, env, max_length=128, traj_num=10000)

fig = plot_dist1D(np.array(trajR), 'Rewards distribution')
plt.savefig('imgs/rewardDist{}.jpg'.format(env_name))

# group episodes
idx_lists = [(np.array(trajR)>0.5).nonzero()[0], (np.logical_and(np.array(trajR)>0.3, np.array(trajR)<0.6)).nonzero()[0]]
ipdb.set_trace()
# ---------------------
# save trajectories
# ---------------------
assert len(idx_lists[0])>2000, "expert model too low quality!"
idx_lists[0] = idx_lists[0][:2000]

exp_trajS = [trajS[i] for i in idx_lists[0]]
exp_trajA = [trajA[i] for i in idx_lists[0]]
exp_trajDone = [trajDone[i] for i in idx_lists[0]]
exp_trajDirect = [trajDirect[i] for i in idx_lists[0]]

exp_trajS = np.concatenate(exp_trajS, axis=0)
exp_trajA = np.concatenate(exp_trajA, axis=0)
exp_trajDone = np.concatenate(exp_trajDone, axis=0)
exp_trajDirect = np.concatenate(exp_trajDirect, axis=0)

traj_path = 'traj/{}'.format(env_name)
os.makedirs(traj_path, exist_ok=True)
if goal is not None:
    np.savez(traj_path+'/goal{}.npz'.format(goal), S=exp_trajS, A=exp_trajA, Done=exp_trajDone, Direct=exp_trajDirect)
else: #use generated different quality trajs
    np.savez(traj_path + '/goal_opt.npz', S=exp_trajS, A=exp_trajA, Done=exp_trajDone, Direct=exp_trajDirect)

    random.shuffle(idx_lists[1])
    idx_lists[0] = idx_lists[1][:2000]
    exp_trajS = [trajS[i] for i in idx_lists[0]]
    exp_trajA = [trajA[i] for i in idx_lists[0]]
    exp_trajDone = [trajDone[i] for i in idx_lists[0]]
    exp_trajDirect = [trajDirect[i] for i in idx_lists[0]]

    exp_trajS = np.concatenate(exp_trajS, axis=0)
    exp_trajA = np.concatenate(exp_trajA, axis=0)
    exp_trajDone = np.concatenate(exp_trajDone, axis=0)
    exp_trajDirect = np.concatenate(exp_trajDirect, axis=0)

    traj_path = 'traj/{}'.format(env_name)
    os.makedirs(traj_path, exist_ok=True)
    np.savez(traj_path+'/goal_noisy.npz', S=exp_trajS, A=exp_trajA, Done=exp_trajDone, Direct=exp_trajDirect)

print("trajectory saved")
