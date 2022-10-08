from stable_baselines3.common.monitor import Monitor
import torch
import os
from gym.wrappers import FrameStack
import ipdb
import utils.expert as exp
from algorithms import skill_bc as skill_bc
from torch.utils.tensorboard import SummaryWriter
from envs.obs_wrapper import DirectImgFlatObsWrapper
import random
from envs.new_fourrooms import *
# --------------
# args
# --------------

env_name = 'MiniGrid-FourRooms-v1'

seed = 42
method = 'bi-modeling'
goal = [13, 16]

load_goals = [[13, 16], [13, 5],] #[13, 12]
# load_goals = [[13, 16]]
load_test_goals = [[13, 16], [13, 5] ]
train_epoch = 50
left_interval = 5
right_interval = 1
batch_size = 32
lr = 0.0005
lambda_mi = 0.5
lambda_clus = 0.5
lambda_consist = 0 #not hold for this dataset, because collected time steps are shuffled
lambda_pseudo = 0
proto_num = 8
batch_mi = True # whether encourage maximizing batch_wise entropy
action_in_mi = True
hard_sel=False
pretrain_pseudo = False
PU = True
PU_thresh = 0.2
use_DEC = True
index_mi = True # whether compute mi based on index as states or index+local img as states
distance='euc'
train_ratio = 1

use_log = True
save_model = True # if save the learned policy model for further analysis
load_best = False # if finetune is based on the last model or the best model
if goal is not None:
    env_kwargs = {'goal_pos': goal}
else:
    env_kwargs = None

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# analyze lambda_mi
train_ratios = [0.05, 0.1, 0.2, 0.5, 1]
batched_performance = []
# analyze batch size
for train_ratio in train_ratios:
    checkpoint_name = 'pseu{}DEC{}mi{}clust{}ratio{}.pth'.format(lambda_pseudo, use_DEC, lambda_mi, lambda_clus,train_ratio)
    env = gym.make(env_name, **env_kwargs)
    env = Monitor(env)
    env = DirectImgFlatObsWrapper(env)

    tensorboard_log = 'logs_Fourroom/{}_{}/load{}train{}_ratio{}_PU/lr{}_mi{}_clus{}PU{}thresh{}DEC{}consis{}_pseudo{}pre{}_hardsel{}_bs{}_loadbest{}'.format(env_name, method, load_goals,
                                                                                         train_epoch, train_ratio,
                                                                                         lr, lambda_mi, lambda_clus,
                                                                                         PU, PU_thresh, use_DEC,
                                                                                        lambda_consist, lambda_pseudo,
                                                                                         pretrain_pseudo,hard_sel,
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
                                           left_interval=left_interval, right_interval=right_interval, cluster_num=proto_num,
                                           revise_goal=goal, pos_ratio=train_ratio,)
    test_batched_demons = []
    for load_test_goal in load_test_goals:
        test_batched_demon = exp.load_batched_demon(load_goals=[load_test_goal], traj_dir=traj_dir, batch_size=batch_size,
                                                left_interval=left_interval, right_interval=right_interval,cluster_num=proto_num,
                                                revise_goal=goal)
        test_batched_demons.append(test_batched_demon)

    exp_batched_demon = exp.load_batched_demon(load_goals=[load_goals[0]], traj_dir=traj_dir, batch_size=batch_size,
                                           left_interval=left_interval, right_interval=right_interval, cluster_num=proto_num,
                                           revise_goal=goal, pos_ratio=train_ratio,)
    if len(load_goals)==2:
        unknown_batched_demon = exp.load_batched_demon(load_goals=[load_goals[1]], traj_dir=traj_dir, batch_size=batch_size,
                                           left_interval=left_interval, right_interval=right_interval, cluster_num=proto_num,
                                           revise_goal=goal)
    else:
        unknown_batched_demon = exp.load_batched_demon(load_goals=load_goals[1:], traj_dir=traj_dir,
                                                       batch_size=batch_size,
                                                       left_interval=left_interval, right_interval=right_interval,
                                                       cluster_num=proto_num,
                                                       revise_goal=goal)


    # ----------------------
    # construct policy model
    # ----------------------
    if method == 'bi-modeling':
        alg_args = {'batch_size': 32, 'lr': lr, 'interval': left_interval, 'future_interval': right_interval,
                    'dropout': 0,
                    'hid_dim': 32, 'proto_dim': 16, 'proto_num': proto_num, 'hard_sel': hard_sel, 'distance': distance,
                    'task': 'cls', 'pretrain_pseudo': pretrain_pseudo,
                    'lambda_mi': lambda_mi, 'lambda_clus': lambda_clus,'lambda_consist': lambda_consist,
                    'batch_mi': batch_mi, 'lambda_pseudo': lambda_pseudo, 'index_mi': index_mi,
                    'PU': PU, 'use_DEC': use_DEC}
        trainer = skill_bc.SkillBC(observation_space=env.observation_space, action_space=env.action_space,
                                   demonstrations=batched_demon, alg_args=alg_args, load_best=load_best,
                                   test_demonstrations=test_batched_demons, logger=logger, action_in_mi=action_in_mi,
                                   env=env)
        learner = trainer

    # ----------------------
    # learn model
    # ----------------------
    #best_model = torch.load("logs/{}.pth".format("noP_nopseu_onehot"))
    best_model = torch.load("logs_Fourroom/pretrained_{}.pth".format(checkpoint_name))
    trainer.load_models(best_model)


    trainer.pre_trained = True

    unknown_optimality_score = trainer.propagate_score(expert_batches=exp_batched_demon, unknown_batches=unknown_batched_demon, threshold=PU_thresh)
    exp_optimality_score = [np.ones(batch['acts'].shape[0]) for batch in exp_batched_demon]

    if True:
        # trainer.demonstrations = exp_batched_demon
        all_demonstration = exp_batched_demon + unknown_batched_demon
        all_optimality = exp_optimality_score + unknown_optimality_score

        index = list(range(len(all_demonstration)))
        random.shuffle(index)
        all_demonstration = [all_demonstration[ind] for ind in index]
        all_optimality = [all_optimality[ind] for ind in index]

        trainer.demonstrations = all_demonstration

    #best_reward = 0
    #'''
    best_reward = trainer.train_all(epochs=train_epoch, use_bi=False, opt_all=False,
                                     eval_env=FrameStack(env, num_stack=left_interval), prefix='adapt',
                                    weights=all_optimality)
    print('best reward: {}'.format(best_reward))

    # finetune the whole framework, without Knowledge distillation
    best_reward_finetuned = trainer.train_all(epochs=train_epoch, use_bi=False,
                                              eval_env =FrameStack(env, num_stack=left_interval), prefix='finetune',
                                              weights=all_optimality)
    print('best reward after finetune: {}'.format(best_reward_finetuned))
    #'''

    rewards = [best_reward, best_reward_finetuned]
    batched_performance.append(rewards)


for i, rewards in enumerate(batched_performance):
    print('performance of setting {}: {}'.format(i, rewards))


