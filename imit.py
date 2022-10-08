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
import matplotlib.pyplot as plt

from imitation.algorithms import bc
import density
from imitation.algorithms.adversarial import gail, airl
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
import ipdb
import utils.expert as exp
import json
from envs.obs_wrapper import ImgFlatObsWrapper
from plot.plot_dist import plot_dist1D
from envs.new_fourrooms import *

# --------------
# args
# --------------

env_name = 'MiniGrid-FourRooms-v1'

n_env = 1
seed = 42
noise = 0.15
traj_size = 128
traj_num = 1000
batch_size = 32
methods = ['bc','airl'] # airl, bc
goal = [13, 16]
load_goals = [[13, 16]]
#load_goals = [[13, 16], [13, 5], [13, 12]]
train_epoch = 50
if goal is not None:
    env_kwargs = {'goal_pos': goal}
else:
    env_kwargs = None

env = make_vec_env(env_name,n_envs=2,wrapper_class=ImgFlatObsWrapper, seed=seed, env_kwargs=env_kwargs)
#env = gym.make(env_name, **env_kwargs)
#env = Monitor(env)
#env = FlatObsWrapper(env)


# ---------------------
# collect trajectories
# ---------------------
traj_dir = 'traj/{}'.format(env_name)

state_list = []
action_list = []
Done_list = []

for load_goal in load_goals:
    np_file = np.load(traj_dir+'/goal{}.npz'.format(load_goal))
    # np_file.files
    state_list.append(np_file['S'])
    action_list.append(np_file['A'])
    Done_list.append(np_file['Done'])

states = np.concatenate(state_list,axis=0)
actions = np.concatenate(action_list, axis=0)#.reshape(states.shape[0],-1)
Dones = np.concatenate(Done_list, axis=0)#.reshape(states.shape[0],-1)

demonstration = exp.traj_array2demon(states,actions,Dones)
batched_demon = exp.batch_demon(demonstration, batch_size)

# ---------------------
# train imitated agent
# ---------------------
result_lists = []
std_lists = []
for method in methods:
    tensorboard_log = 'logs/{}_{}/'.format(env_name, method)
    os.makedirs(tensorboard_log, exist_ok=True)

    if method == 'bc':
        from stable_baselines3.common.utils import configure_logger
        logger = configure_logger(tensorboard_log=tensorboard_log, tb_log_name='Load{}'.format(load_goals),
                                  reset_num_timesteps=False)
        learner = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=batched_demon,
            batch_size=batch_size,
            #custom_logger=logger,
        )
        trainer = learner
    elif method == 'density':
        from imitation.util import logger as imit_logger
        logger = imit_logger.configure(folder=tensorboard_log+'Load{}'.format(load_goals),format_strs=["stdout","csv","tensorboard"])

        learner = PPO(env=env, policy=MlpPolicy, batch_size=batch_size, ent_coef=0.0, learning_rate=3e-4, n_epochs=10,
                      n_steps=32)
        learner = density.DensityAlgorithm(
                venv=env, demonstrations=batched_demon, rl_algo=learner,custom_logger=logger
        )
        trainer = learner

    else:
        from imitation.util import logger as imit_logger
        logger = imit_logger.configure(folder=tensorboard_log+'Load{}'.format(load_goals),format_strs=["stdout","csv","tensorboard"])

        learner = PPO(env=env, policy=MlpPolicy, batch_size=batch_size, ent_coef=0.0, learning_rate=3e-4, n_epochs=10, n_steps=32)
        reward_net = BasicRewardNet(env.observation_space, env.action_space, normalize_input_layer=RunningNorm)

        if method == 'gail':
            trainer = gail.GAIL(demonstrations=batched_demon, demo_batch_size=batch_size, gen_replay_buffer_capacity=128,
                                        n_disc_updates_per_round=4, venv=env, custom_logger=logger,
                                        gen_algo=learner, reward_net=reward_net,allow_variable_horizon=True)
        elif method == 'airl':
            trainer = airl.AIRL(demonstrations=batched_demon, demo_batch_size=batch_size, gen_replay_buffer_capacity=128,
                                        n_disc_updates_per_round=4, venv=env,
                                        gen_algo=learner, reward_net=reward_net,allow_variable_horizon=True)

    result_list = []
    std_list = []
    for trained_epoch in range(int(train_epoch/5)):
        if method == 'bc':
            trainer.train(n_epochs=5)
        elif method == 'density':
            trainer.train()
        else:
            eval_env = FlatObsWrapper(Monitor(gym.make(env_name)))
            #eval_call
            trainer.train(5*5000)

        reward, result_std = evaluate_policy(learner.policy, env, n_eval_episodes=100, render=False)
        print(f"Reward after training: {reward}")
        result_list.append(reward)
        std_list.append(result_std)

    result_lists.append(result_list)
    std_lists.append(std_list)

for i, (result_list, std_list) in enumerate(zip(result_lists, std_lists)):
    print('method {}, rewards: {}'.format(i, result_list))
    print('method {}, stds: {}'.format(i, std_list))

ipdb.set_trace()

print('finished')
