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
from imitation.algorithms.adversarial import gail, airl
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
import ipdb
import utils.expert as exp
import json
from gym import spaces
import utils.data_sepsis as data_sepsis
from plot.plot_dist import plot_dist1D
from utils.evaluate import evaluate_model
# --------------
# args
# --------------

env_name = 'sepsis'

load_test_goal = 'sepsis_val.csv'
load_goal = 'sepsis_train.csv'
data_path = './data/sepsis/'
n_env = 1
seed = 42
noise = 0.15
traj_size = 128
traj_num = 1000
batch_size = 32
methods = ['bc',] # gail, airl, bc
train_epoch = 40

left_interval = 1
right_interval = 1
batch_size = 128
proto_num = 4
reduce=5
# ---------------------
# collect trajectories
# ---------------------
traj_dir = 'traj/{}'.format(env_name)

state_list = []
action_list = []
Done_list = []
#train_batched_demon = data_sepsis.load_batched_sepsis(load_goal, data_path,batch_size=batch_size,
#                                               left_interval=left_interval, right_interval=right_interval, cluster_num=proto_num)
all_batched_demon, train_batched_demon = data_sepsis.load_batched_sepsis(load_goal, data_path,batch_size=batch_size, split='all_pos', left_interval=left_interval, right_interval=right_interval, cluster_num=proto_num, reduce=reduce)
#train_batched_demon = all_batched_demon


valid_batched_demons, test_batched_demons = data_sepsis.load_batched_sepsis(load_test_goal, data_path,batch_size=-1, split='pos_neg',
                                               left_interval=left_interval, right_interval=right_interval, cluster_num=proto_num, val_ratio=0.2, reduce=reduce)
valid_batched_demon = valid_batched_demons[0]
test_batched_demon = test_batched_demons[0]
# ---------------------
# train imitated agent
# ---------------------
result_lists = []
std_lists = []
auroc_lists = []
macroF_lists = []
for method in methods:
    tensorboard_log = 'logs/{}_{}/'.format(env_name, method)
    os.makedirs(tensorboard_log, exist_ok=True)

    from imitation.algorithms import bc

    if method == 'bc':
        from stable_baselines3.common.utils import configure_logger
        logger = configure_logger(tensorboard_log=tensorboard_log, tb_log_name='Load',
                                  reset_num_timesteps=False)
        learner = bc.BC(
            observation_space=spaces.Box(
            low=0,
            high=255,
            shape=(43,),
            dtype='uint8'
        ),
            action_space=spaces.Discrete(25//reduce),
            demonstrations=train_batched_demon,
            batch_size=batch_size,
            custom_logger=logger,
        )
        trainer = learner

    else:
        from imitation.util import logger as imit_logger
        logger = imit_logger.configure(folder=tensorboard_log+'Load',format_strs=["stdout","csv","tensorboard"])

        learner = PPO(env=None, policy=MlpPolicy, batch_size=batch_size, ent_coef=0.0, learning_rate=3e-4, n_epochs=10, n_steps=32)
        reward_net = BasicRewardNet(np.zeros((43,)), spaces.Discrete(25), normalize_input_layer=RunningNorm)

        if method == 'gail':
            trainer = gail.GAIL(demonstrations=train_batched_demon, demo_batch_size=batch_size, gen_replay_buffer_capacity=128,
                                        n_disc_updates_per_round=4, venv=None, custom_logger=logger,
                                        gen_algo=learner, reward_net=reward_net,allow_variable_horizon=True)
        elif method == 'airl':
            trainer = airl.AIRL(demonstrations=train_batched_demon, demo_batch_size=batch_size, gen_replay_buffer_capacity=128,
                                        n_disc_updates_per_round=4, venv=None,
                                        gen_algo=learner, reward_net=reward_net,allow_variable_horizon=True)

    result_list = []
    std_list = []
    auroc_list = []
    macroF_list = []
    for trained_epoch in range(int(train_epoch/5)):
        if method == 'bc':
            trainer.train(n_epochs=5)
        else:
            #eval_call
            trainer.train(5*5000)

        reward, result_std, auroc, macroF = evaluate_model(learner.policy, test_batched_demon)
        print(f"Reward after training: {reward}")
        result_list.append(reward)
        std_list.append(result_std)
        auroc_list.append(auroc)
        macroF_list.append(macroF)

    result_lists.append(result_list)
    std_lists.append(std_list)
    auroc_lists.append(auroc_list)
    macroF_lists.append(macroF_list)

for i, (result_list, std_list) in enumerate(zip(result_lists, std_lists)):
    print('method {}, rewards: {}'.format(i, result_list))
    print('method {}, stds: {}'.format(i, std_list))
    print('method {}, aurocs: {}'.format(i, auroc_lists[i]))
    print('method {}, macroF s: {}'.format(i, macroF_lists[i]))

ipdb.set_trace()

print('finished')
