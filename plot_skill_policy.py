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
from envs.obs_wrapper import ImgFlatObsWrapper
from envs.obs_wrapper import DirectImgFlatObsWrapper
from plot.plot_dist import plot_dist1D
from plot.plot_minigrid import plot_grid
from algorithms import skill_bc as skill_bc
from plot.plot_utils import draw_act_dist, draw_skill_traj
from envs.new_fourrooms import *
# --------------
# args
# --------------
env_name = 'MiniGrid-FourRooms-v1'
#env_name = 'MiniGrid-DoorKey-8x8-v0'
FIX_MAP = False
FIX_TASK = True
goal = [13, 16]
load_goal = "noP_nopseu_onehot"#best_all_pre. best_noPseuHard,Img_noPseu
load_path = "logs/{}.pth".format(load_goal)
#load_path = "logs/{}.pth_finetuned".format(load_goal)
seed = 42
action_in_mi = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
distance='euc'
hard_sel = True
index_mi = True
proto_num = 8
deterministic = True

# ---------------------
# env configuration
# ---------------------
os.environ["SDL_VIDEODRIVER"] = "dummy"

img_ct = 0

def vis_env(env):
    global img_ct
    img = env.render(mode='rgb_array')
    plt.imsave('imgs_new/env_set{}.jpg'.format(img_ct), img)
    img_ct += 1

if FIX_TASK:
    env_kwargs = {'goal_pos': goal}
else:
    env_kwargs = None

env = gym.make(env_name, **env_kwargs)
env = Monitor(env)
#env = ImgFlatObsWrapper(env)
env = DirectImgFlatObsWrapper(env)

env.reset()
vis_env(env)


# ---------------------
# collect episodes
# ---------------------

alg_args = {'batch_size': 32, 'lr': 5e-4, 'interval': 5, 'future_interval': 1,
            'dropout': 0,
            'hid_dim': 32, 'proto_dim': 16, 'proto_num': proto_num, 'hard_sel': hard_sel, 'distance': distance,
            'task': 'cls', 'index_mi': index_mi,
            'lambda_mi': 0.5, 'lambda_clus': 0.5, 'batch_mi': True, }
trainer = skill_bc.SkillBC(observation_space=env.observation_space, action_space=env.action_space,
                           demonstrations=None, alg_args=alg_args, load_best=False, action_in_mi=action_in_mi,
                           test_demonstrations=None, logger=None)
best_model = torch.load(load_path)
trainer.load_models(best_model)
env = FrameStack(env, num_stack=5)
reward, std = trainer.evaluate_env(env, n_eval_episodes=100, use_bi=False)
print("reward of loaded model, reward: {}, std: {}".format(reward, std))

# collect visualization trajectory
if FIX_MAP:
    state_ini = exp.get_env_state(env)
else:
    state_ini = None
exp.set_env_state(env, state_ini)
vis_env(env)

trajPos, trajDirect, trajR, trajAct, traj_skill = trainer.collect_Grid_trajectories(
    env, max_length=128, traj_num=100, state_ini=state_ini, fix_skill=None, deterministic=deterministic)
#trajPos, trajDirect, trajR, trajAct, traj_skill = exp.collect_clust_traj(load_goals=[[13, 16], [13, 5],],
#                                                                         traj_dir='traj/{}'.format(env_name))

#fig = plot_dist1D(np.array(trajR), 'Rewards distribution')
#plt.savefig('imgs_new/{}rewardDist{}.jpg'.format(load_goal,env_name))

# ---------------------
# visualization
# ---------------------
assert 'Grid' in env_name, "only support MiniGrid environment for now"

# group episodes
traj_info_dict = {'pos': trajPos, 'dir': trajDirect, 'act': trajAct, 'skill': traj_skill}
idx_lists = [(np.array(trajR)>-1).nonzero()[0], (np.array(trajR)<0.5).nonzero()[0]]
# idx_lists = [(np.array(traj_skill)>3).nonzero()[0], (np.array(traj_skill)<4).nonzero()[0]]
#ipdb.set_trace()
draw_act_dist(env, traj_info_dict, idx_lists, state_ini=state_ini, name='{}FourRoom_skilled'.format(load_goal))

ipdb.set_trace()
for list_id,idx_list in enumerate(idx_lists):
    for id, idx in enumerate(idx_list):
        if id <=5 and list_id ==0:
            traj_info_dict = {'pos': trajPos[idx], 'dir': trajDirect[idx], 'act': trajAct[idx], 'skill': traj_skill[idx]}
            draw_skill_traj(env, traj_info_dict, state_ini=state_ini,
                          name='{}FourRoom_{}traj{}'.format(load_goal, list_id,id))


# plot skill-wise
#'''
for skill in range(proto_num):
    trajPos, trajDirect, trajR, trajAct, traj_skill = trainer.collect_Grid_trajectories(
        env, max_length=20, traj_num=100, state_ini=state_ini, fix_skill=skill, show_max_part=False)

    #fig = plot_dist1D(np.array(trajR), 'Rewards distribution')
    #plt.savefig('imgs_new/{}rewardDist{}_skill{}.jpg'.format(load_goal,env_name, skill))

    traj_info_dict = {'pos': trajPos, 'dir': trajDirect, 'act': trajAct, 'skill': traj_skill}
    idx_lists = [(np.array(trajR) > -1).nonzero()[0], (np.array(trajR) < 0.5).nonzero()[0]]
    draw_act_dist(env, traj_info_dict, idx_lists, state_ini=state_ini, name='{}FourRoom_skill_{}'.format(load_goal,skill))

    '''
    for list_id,idx_list in enumerate(idx_lists):
        for id, idx in enumerate(idx_list):
            if id <=5 and list_id==0:
                traj_info_dict = {'pos': trajPos[idx], 'dir': trajDirect[idx], 'act': trajAct[idx], 'skill': traj_skill[idx]}
                draw_skill_traj(env, traj_info_dict, state_ini=state_ini,
                              name='{}FourRoom_{}skill{}traj{}'.format(load_goal, list_id,skill,id))
    '''

#'''
