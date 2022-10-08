import os
import numpy as np
import torch
import ipdb
import utils.data_sepsis as data_sepsis
from algorithms import skill_bc
import random
from torch.utils.tensorboard import SummaryWriter
from gym import spaces

# --------------
# args
# --------------
env_name = 'sepsis'
left_interval = 5
right_interval = 1
batch_size = 128
proto_num = 8
load_test_goal = 'sepsis_val.csv'
load_goal = 'sepsis_train.csv'

data_path = './data/sepsis/'
method = 'bi-modeling'
lr = 0.001
lambda_mi = 1
lambda_clus = 0
lambda_consist = 0
lambda_pseudo = 0
PU = True
PU_thresh = 0.5
use_DEC = False
batch_mi = False
action_in_mi = True
distance = 'euc'
seed = 42
train_ratio = 1

use_log = True
train_epoch = 30

# --------------
# load demons
# --------------

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# analyze lambda_mi
lambda_mis = [0, 0.5, 1, 2]
batched_performance = []
# analyze batch size
for lambda_mi in lambda_mis:
    checkpoint_name = 'pseu{}DEC{}mi{}clust{}ratio{}nobi.pth'.format(lambda_pseudo, use_DEC, lambda_mi, lambda_clus,
                                                                     train_ratio)
    tensorboard_log = 'logs_sepsis/{}_{}_seed{}ratio{}nobi/lr{}_mi{}_clus{}PU{}thresh{}DEC{}consis{}pseudo{}_bs{}'.format(
        env_name, method, seed,
        train_ratio, lr, lambda_mi, lambda_clus,
        PU, PU_thresh, use_DEC,
        lambda_consist, lambda_pseudo,
        batch_size, )
    os.makedirs(tensorboard_log, exist_ok=True)
    if use_log:
        logger = SummaryWriter(tensorboard_log)
    else:
        logger = None

    (train_batched_demon, exp_batched_demon), scaler = data_sepsis.load_batched_sepsis(load_goal, data_path,
                                                                                       batch_size=batch_size,
                                                                                       left_interval=left_interval,
                                                                                       right_interval=right_interval,
                                                                                       cluster_num=proto_num,
                                                                                       split='all_pos', shuffle=True,
                                                                                       pos_ratio=train_ratio,
                                                                                       return_scaler=True)
    _, unknown_batched_demon = data_sepsis.load_batched_sepsis(load_goal, data_path, batch_size=batch_size,
                                                               left_interval=left_interval,
                                                               right_interval=right_interval,
                                                               cluster_num=proto_num,
                                                               split='pos_neg', shuffle=True, scaler=scaler)

    test_batched_demons = data_sepsis.load_batched_sepsis(load_test_goal, data_path, batch_size=-1,
                                                          left_interval=left_interval, right_interval=right_interval,
                                                          cluster_num=proto_num,
                                                          split='pos_neg', scaler=scaler)
    if method == 'bi-modeling':
        alg_args = {'batch_size': 32, 'lr': lr, 'interval': left_interval, 'future_interval': right_interval,
                    'dropout': 0,
                    'hid_dim': 32, 'proto_dim': 16, 'proto_num': proto_num, 'hard_sel': True, 'distance': distance,
                    'task': 'cls', 'pretrain_pseudo': False,
                    'lambda_mi': lambda_mi, 'lambda_clus': lambda_clus, 'lambda_consist': lambda_consist,
                    'batch_mi': batch_mi, 'lambda_pseudo': lambda_pseudo, 'index_mi': False,
                    'PU': PU, 'use_DEC': use_DEC}
        trainer = skill_bc.SkillBC(observation_space=np.zeros((45,)), action_space=spaces.Discrete(25),
                                   demonstrations=train_batched_demon, alg_args=alg_args, load_best=False,
                                   test_demonstrations=test_batched_demons, logger=logger, action_in_mi=action_in_mi)
        learner = trainer

    # ----------------------
    # learn model
    # ----------------------

    trainer.train_all(epochs=train_epoch, use_bi=False,
                      eval_env=None, prefix='pretrain')
    trainer.pre_trained = True
    save_path = 'logs_sepsis/pretrained_{}'.format(checkpoint_name)
    best_model = trainer.get_models()
    torch.save(best_model, save_path)

    unknown_optimality_score = trainer.propagate_score(expert_batches=exp_batched_demon,
                                                       unknown_batches=unknown_batched_demon, threshold=PU_thresh)
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

    best_reward_finetuned = trainer.train_all(epochs=train_epoch, use_bi=False,
                                              eval_env=None, prefix='finetune', weights=all_optimality)
    print('best reward after finetune: {}'.format(best_reward_finetuned))

    '''
    best_model = trainer.best_content
    save_path = 'logs/finetuned_{}'.format(checkpoint_name)
    torch.save(best_model, save_path)
    '''

    rewards = [best_reward_finetuned]
    batched_performance.append(rewards)

for i, rewards in enumerate(batched_performance):
    print('performance of setting {}: {}'.format(i, rewards))

print("finished")
