# implemented with reference to Standford-ILIAD/ILEED and stable_baselines3
import copy
from types import MethodType
from typing import Optional
import ipdb
import math
from torch import Tensor
from numpy.random import uniform
import numpy as np
from copy import deepcopy
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
import random
from sklearn.cluster import SpectralClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA

# ------------------------------------------------------------------------------------#
# PART 0: Policy configuration
# ------------------------------------------------------------------------------------#

def get_env_state(env):
    if 'VecEnv' in type(env).__name__:
        assert len(env.envs) == 1, 'only support 1 env for VecEnv'
        return deepcopy(env.envs[0])
    else:
        return deepcopy(env.env)

def set_env_state(env, state):
    if state is None:
        print('no state initialization used')
        return env.reset()

    if 'VecEnv' in type(env).__name__:
        assert len(env.envs) == 1, 'only support 1 env for VecEnv'
        env.envs[0] = deepcopy(state)
        obs = [env.envs[0].observation(env.envs[0].gen_obs())] # for MiniGrid Env with FlatObsWrapper
    elif 'FrameStack' in type(env).__name__:
        env.env = deepcopy(state)
        obs = env.observation()
    else:
        env.env = deepcopy(state)

        try:
            obs = env.observation(env.gen_obs())
        except:
            obs = np.array(list(env.env.unwrapped.state))

    return obs

# ------------------------------------------------------------------------------------#
# PART I: Noise functions
# ------------------------------------------------------------------------------------#
def model_add_uniform_noise(expert_model, noise_level):
    '''
        expert acts based on p(a|s), r of the time
        expert acts randomly, (1-r) of the time
        noise_level=0 keeps the original expert
        noise_level=1 makes expert completely random
    '''

    def _get_action_dist_from_latent(self, latent_pi: Tensor, latent_sde: Optional[Tensor] = None) -> Distribution:
        """
        Retrieve action distribution given the latent codes.
        :param latent_pi: Latent code for the actor
        :param latent_sde: Latent code for the gSDE exploration function
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        use_unif = uniform() < noise_level

        """
        if isinstance(self.action_dist, DiagGaussianDistribution):
            std = self.log_std
            if use_unif: std /= max(r, 1e-5)
            return self.action_dist.proba_distribution(mean_actions, std)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            std = self.log_std
            if use_unif: std /= max(r, 1e-5)
            return self.action_dist.proba_distribution(mean_actions, std, latent_sde)
        """

        if use_unif:
            mean_actions *= 0

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_sde)
        else:
            raise ValueError("Invalid action distribution")

    expert_model.policy._get_action_dist_from_latent = MethodType(_get_action_dist_from_latent, expert_model.policy)

    return expert_model

#------------------------------------------------------------------------------------#
# PART II: Collecting Trajectories
#------------------------------------------------------------------------------------#
def collect_trajectories(model, env, max_length=128, deterministic=False, traj_num=100, state_ini=None):
    '''
    Calculates the trajectory of our expert within some env, which are for policy prediction
    Inputs:
        env           - env class to evaluate on. If the env is wrapped with vec_env, it must be the top wrapper.
        deterministic - BOOL: wether the policy we use is deterministic
        max_length        - int: maximum timesteps we want the collected traj to be
    Returns:
        S_list, A_list, Done_list: list of arrays
        R_list: list of float
    '''
    S_list, A_list, R_list, Done_list, Direction_list = [], [], [], [], []
    Direction_dict = [[1,0],[0,1],[-1,0],[0,-1]]

    for traj in range(traj_num):
        S, A, R, Done, Direction = [], [], [], [], []

        obs = env.reset()
        if state_ini is not None:
            obs = set_env_state(env, state_ini)
        done = False
        length = 0
        while not done and length<max_length:
            if 'VecEnv' in type(env).__name__:
                Direction.append(np.array(Direction_dict[env.envs[0].agent_dir]))
            else:
                Direction.append(np.array(Direction_dict[env.agent_dir]))

            # on policy
            act, _ = model.policy.predict(obs, deterministic=deterministic)
            obs_new, rew, done, _ = env.step(act) # return np.array
            # store the corresponding trajectory

            S.append(obs)
            A.append(act)
            R.append(rew)
            Done.append(done)

            # on to the next step
            obs = obs_new
            length = length+1

        if 'VecEnv' in type(env).__name__:
            S_list.append(np.concatenate(S, axis=0))
            A_list.append(np.concatenate(A, axis=0))
            R_list.append(np.concatenate(R, axis=0).sum())
            Done_list.append(np.concatenate(Done, axis=0))
            Direction_list.append(np.concatenate(Direction, axis=0))
        else:
            S_list.append(np.stack(S)) # shape: [step, 2]
            A_list.append(np.array(A)) # shape: [step,]
            R_list.append(np.array(R).sum()) # shape: integer
            Done_list.append(np.array(Done)) # shape: [step, ]
            Direction_list.append(np.stack(Direction))

    return S_list, A_list, R_list, Done_list, Direction_list

def collect_Grid_trajectories(model, env, max_length=128, deterministic=False, traj_num=100, state_ini=None):
    '''
    Calculates the trajectory of our expert within some env, for visualization of MiniGrid env
    Inputs:
        env           - env class to evaluate on
        deterministic - BOOL: wether the policy we use is deterministic
        max_length        - int: maximum timesteps we want the collected traj to be
    Returns:
        S_list, A_list, Done_list: list of arrays
        R_list: list of float
    '''
    pos_list, direct_list, R_list, A_list = [], [], [], []
    for traj in range(traj_num):
        pos, direct, R, A = [], [], [], []

        obs = env.reset()
        if state_ini is not None:
            obs = set_env_state(env, state_ini)
        done = False
        length = 0
        while not done and length<max_length:
            if 'VecEnv' in type(env).__name__:
                pos.append(env.envs[0].agent_pos)
                direct.append(env.envs[0].agent_dir)

            else:
                pos.append(env.agent_pos)
                direct.append(env.agent_dir)

            # on policy
            act, _ = model.policy.predict(obs, deterministic=deterministic)
            obs_new, rew, done, _ = env.step(act) # return np.array
            # store the corresponding trajectory
            A.append(act)
            R.append(rew)

            # on to the next step
            obs = obs_new
            length = length+1

        if 'VecEnv' in type(env).__name__:
            pos_list.append(np.stack(pos)) # shape: [step, 2]
            A_list.append(np.concatenate(A, axis=0)) # shape: [step,]
            R_list.append(np.concatenate(R, axis=0).sum()) # shape: integer
            direct_list.append(np.array(direct)) # shape: [step, ]
        else:
            pos_list.append(np.stack(pos)) # shape: [step, 2]
            A_list.append(np.array(A)) # shape: [step,]
            R_list.append(np.array(R).sum()) # shape: integer
            direct_list.append(np.array(direct)) # shape: [step, ]

    return pos_list, direct_list, R_list, A_list


#------------------------------------------------------------------------------------#
# PART III: data formulation
#------------------------------------------------------------------------------------#

def traj_array2demon(states, actions, dones, interval=1, future_interval=None, cluster_label=None):
    '''
    change trajectory arrays into list of time-step demonstrations

    :param states: numpy array of [all_steps, state_dim]
    :param actions: numpy array of [all_steps, action_dim] or [all_steps,]
    :param dones:  numpy array of [all_steps,]
    :param interval: step of states recorded in each time step
    :return:
        list of dictionaries. Length: all_steps. Each element: {'obs'=, 'acts'=, 'next_obs'=, 'dones'=}
    '''
    if future_interval is None:
        future_interval = interval
    demon_list = []

    finished_steps = dones.nonzero()[0]
    cur_traj = 0 # denote idx of current trajectory

    for step, [state, action, done] in enumerate(zip(states, actions, dones)):
        if cur_traj >= finished_steps.shape[0]: # uncompleted trajectory in the end
            break

        start_idx = 0 if cur_traj ==0 else finished_steps[cur_traj-1]+1 # start point of current trajectory

        step_dict = {}
        step_dict['acts'] = action
        step_dict['dones'] = done
        if cluster_label is not None:
            step_dict['cluster_label'] = cluster_label[step]

        obs = []
        for i in range(interval):
            if step-interval+1+i < start_idx:
                obs.append(np.zeros(state.shape))
            else:
                obs.append(states[step-interval+i+1])
        step_dict['obs'] = np.concatenate(obs, axis=-1)

        next_obs = []
        for i in range(future_interval):
            if step+i > finished_steps[cur_traj]:
                next_obs.append(np.zeros(state.shape))
            else:
                next_obs.append(states[step+i])
        step_dict['next_obs'] = np.concatenate(next_obs, axis=-1)

        demon_list.append(step_dict)

        if step >= finished_steps[cur_traj]:
            cur_traj += 1

    return demon_list

def batch_demon(demonstrations, bs):
    '''
    make batched demonstration from individual demonstration sets
    :param demonstrations: list of dictionaries. Length: all_steps. Each element: {'obs'=, 'acts'=, 'next_obs'=, 'dones'=}
    :param bs: batch_size

    :return:
        list of dictionaries. Length: all_steps//batch_size. Each element: {'obs'=[bs, ], 'acts'=[bs,], 'next_obs'=[], 'dones'=[]}
    '''

    full_length = len(demonstrations)

    batched_list = []
    for b in range(int(full_length/bs)):
        batched_dict = {}

        for key in demonstrations[0].keys():
            content = []
            for i in range(bs):
                idx = b*bs+i
                content.append(demonstrations[idx][key])

            content = np.array(content)
            batched_dict[key] = content

        batched_list.append(batched_dict)

    return batched_list

def trajarray2list(states, actions, Dones):
    '''
    change trajectory arrays to list of trajectories, each trajectory is a dictionary

    :param states:
    :param actions:
    :param Dones:
    :return:
    '''
    list_traj = []

    idx_start = 0

    for idx_end in Dones.nonzero()[0]:
        traj = {}

        traj['obs'] = states[idx_start:idx_end+1]
        traj['acts'] = actions[idx_start:idx_end+1]
        traj['dones'] = Dones[idx_start:idx_end+1]

        idx_start = idx_end+1
        list_traj.append(traj)

    return list_traj

def emb_cluster(obs_np, cluster_num=4):
    if obs_np.shape[-1] >= 10:
        pca = PCA(n_components=4)
        pca.fit(obs_np)
        trans_obs = pca.transform(obs_np)
    else:
        trans_obs = obs_np
    '''
    if False:
        clustering = DBSCAN(eps=1, min_samples=2).fit(trans_obs)
        cluster_label = clustering.labels_
        (clustering.labels_ == 2).sum()

    elif True:
        clustering = SpectralClustering(n_clusters=4,assign_labels='discretize',random_state=0).fit(trans_obs)
        cluster_label = clustering.labels_
    '''
    if True:
        clustering = KMeans(n_clusters=4, random_state=0, ).fit(trans_obs)
        cluster_label = clustering.labels_

    return cluster_label

def region_cluster(obs_np, cluster_num=4): # obs_np: (pos_x, pos_y)
    assert math.sqrt(cluster_num)%1 ==0, "cluster_num have to be square in region_cluster"

    region_label = np.zeros(obs_np.shape[0])
    region_label[obs_np[:,0]>=9] += 2
    region_label[obs_np[:,1]>=9] += 1

    return region_label

def load_batched_demon(load_goals, traj_dir, batch_size=32, left_interval=1, right_interval=1,cluster_num=4,
                       revise_goal=None, pos_ratio=1, label=None):
    """

    :param load_goals:
    :param traj_dir:
    :param batch_size:
    :param left_interval:
    :param right_interval:
    :return:
    """
    state_list = []
    action_list = []
    Done_list = []
    direction_list = []
    group_list = []

    for i,load_goal in enumerate(load_goals):
        np_file = np.load(traj_dir + '/goal{}.npz'.format(load_goal))

        data_size = np_file['Done'].shape[0]
        if label == 'train':
            data_size = int(data_size*0.9)
            if i==0 and pos_ratio !=1:
                data_size = int(data_size*pos_ratio)
        elif label == 'test':
            data_size = int(data_size* 0.1)

        if label == 'test':
            # np_file.files
            state_list.append(np_file['S'][-data_size:])
            action_list.append(np_file['A'][-data_size:])
            Done_list.append(np_file['Done'][-data_size:])
            direction_list.append(np_file['Direct'][-data_size:])
        else:
            # np_file.files
            state_list.append(np_file['S'][:data_size])
            action_list.append(np_file['A'][:data_size])
            Done_list.append(np_file['Done'][:data_size])
            direction_list.append(np_file['Direct'][:data_size])


        if i==0:
            group_array = np.zeros(data_size, dtype=np.int64)
        else:
            group_array = np.ones(data_size, dtype=np.int64)
        group_list.append(group_array)

    states = np.concatenate(state_list, axis=0)
    actions = np.concatenate(action_list, axis=0)  # .reshape(states.shape[0],-1)
    Dones = np.concatenate(Done_list, axis=0)  # .reshape(states.shape[0],-1)
    directions = np.concatenate(direction_list,axis=0)
    groups = np.concatenate(group_list, axis=0)

    if revise_goal is not None:
        states[:,-3] = revise_goal[0]
        states[:,-4] = revise_goal[1]

    filted_idx = actions <= 3
    states = states[filted_idx]
    actions = actions[filted_idx]
    Dones = Dones[filted_idx]
    directions = directions[filted_idx]
    groups = groups[filted_idx]

    filted_idx = actions <= 3
    i = 0
    while i < filted_idx.shape[0]-1:
        if actions[i] != 2 and actions[i+1] != 2:
            if actions[i] != actions[i+1]:
                filted_idx[i] = False
                filted_idx[i+1] = False
                i += 2
            else:
                i += 1
        else:
            i += 1
    states = states[filted_idx]
    actions = actions[filted_idx]
    Dones = Dones[filted_idx]
    directions = directions[filted_idx]
    groups = groups[filted_idx]

    states = np.concatenate((states, directions), axis=-1)
    obs_np = states[:,-4:] #pos_x, pos_y; direct_x, direct_y
    obs_np = np.concatenate((obs_np, actions.reshape((actions.shape[0],-1))), axis=-1)

    clust_label = emb_cluster(obs_np, cluster_num=cluster_num//2)
    # assign clust labels for negative trajectories
    clust_label[groups!=0] += cluster_num//2

   # print("use region label as cluster label")
   # region_label = region_cluster(obs_np, cluster_num=cluster_num)
    Bi_demonstration = traj_array2demon(states, actions, Dones, interval=left_interval,
                                            future_interval=right_interval, cluster_label=clust_label)
    # shuffle the collected data
    random.shuffle(Bi_demonstration)
    if batch_size==-1:
        batch_size = len(Bi_demonstration)
    batched_demon = batch_demon(Bi_demonstration, batch_size)

    return batched_demon


def collect_clust_traj(load_goals, traj_dir, batch_size=32, left_interval=1, right_interval=1, cluster_num=8,
                       revise_goal=None):
    """

    :param load_goals:
    :param traj_dir:
    :param batch_size:
    :param left_interval:
    :param right_interval:
    :return:
    """
    state_list = []
    action_list = []
    Done_list = []
    direction_list = []
    encoded_dir_list = []
    group_list = []

    pos_list = []
    clust_list = []

    for i, load_goal in enumerate(load_goals):
        np_file = np.load(traj_dir + '/goal{}.npz'.format(load_goal))
        # np_file.files
        state_list.append(np_file['S'])
        action_list.append(np_file['A'])
        Done_list.append(np_file['Done'])
        directs = np_file['Direct']
        direct_array = np.zeros(directs.shape[0])
        for j, direct in enumerate(directs):
            if direct[0]==1:
                direct_array[j]=0
            elif direct[0] == -1:
                direct_array[j] = 2
            elif direct[0]==0:
                if direct[1]==1:
                    direct_array[j]=1
                elif direct[1]==-1:
                    direct_array[j]=3
            else:
                ipdb.set_trace()

        direction_list.append(direct_array)
        encoded_dir_list.append(directs)
        if i == 0:
            group_array = np.zeros(np_file['Done'].shape, dtype=np.int64)
        else:
            group_array = np.ones(np_file['Done'].shape, dtype=np.int64)
        group_list.append(group_array)

        pos_list.append(np_file['S'][:, -2:])

    states = np.concatenate(state_list, axis=0)
    actions = np.concatenate(action_list, axis=0)  # .reshape(states.shape[0],-1)
    Dones = np.concatenate(Done_list, axis=0)  # .reshape(states.shape[0],-1)
    directions = np.concatenate(direction_list, axis=0)
    encoded_dir = np.concatenate(encoded_dir_list, axis=0)

    groups = np.concatenate(group_list, axis=0)
    poss = np.concatenate(pos_list, axis=0)

    if revise_goal is not None:
        states[:, -3] = revise_goal[0]
        states[:, -4] = revise_goal[1]

    filted_idx = actions <= 3
    states = states[filted_idx]
    actions = actions[filted_idx]
    Dones = Dones[filted_idx]
    directions = directions[filted_idx]
    groups = groups[filted_idx]
    poss = poss[filted_idx]
    encoded_dir = encoded_dir[filted_idx]

    i = 0

    states = np.concatenate((states, encoded_dir), axis=-1)
    obs_np = states[:, -4:]  # pos_x, pos_y; direct_x, direct_y
    obs_np = np.concatenate((obs_np, actions.reshape((actions.shape[0], -1))), axis=-1)

    clust_label = emb_cluster(obs_np, cluster_num=cluster_num // 2)

    # assign clust labels for negative trajectories
    clust_label[groups != 0] += cluster_num // 2


    return [pos for pos in poss], [direction for direction in directions], [Done for Done in Dones], [action for action in actions], [clust for clust in clust_label]