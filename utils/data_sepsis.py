import numpy as np
import ipdb
import utils.expert as exp
import random
import copy
from sklearn.preprocessing import MinMaxScaler

def load_batched_sepsis(load_goals, traj_dir, batch_size=32, left_interval=1, right_interval=1,cluster_num=4,
                        split=None, shuffle=False, pos_ratio=1, scaler=None, return_scaler=False, val_ratio=None, reduce=1):
    """
    :param load_goals:
    :param traj_dir:
    :param batch_size:
    :param left_interval:
    :param right_interval:
    :param cluster_num:
    :param split: {None, 'pos_neg', 'all_pos'}. if None: return all batches as a single list.
        if 'pos_neg': return two lists, of pos/neg batches
        if 'all_pos': return two lists, of all batches and only expert batches
    :param shuffle: if True: batches will be shuffled
    :return:
    """
    assert (not return_scaler) or val_ratio is None, "cannot return scaler when val_ratio is given, for val/test split"

    given_bs = batch_size
    data = np.genfromtxt(traj_dir+load_goals, delimiter=',')[1:,:]


    states = data[:,4:-2] #132358
    actions = data[:,-1].astype(int)  # .reshape(states.shape[0],-1)
    if reduce !=1:
        actions = (actions/reduce).astype(int)
    Dones = data[:,-2]!=0  # .reshape(states.shape[0],-1). sum: 10203
    pre_actions = copy.deepcopy(actions)
    pre_actions[Dones] = -1
    pre_actions[1:] = pre_actions[:-1]
    pre_actions[0] = -1
    states=np.concatenate((states, pre_actions.reshape(-1, 1)), axis=-1)

    # normalize features
    if scaler is None:
        scaler = MinMaxScaler().fit(states)
        states = scaler.transform(states)
    else:
        states = scaler.transform(states)

    optimality = data[:, -2]
    for i in range(optimality.shape[0], 0, -1):
        if optimality[i - 1] == 0:
            optimality[i - 1] = optimality[i]
    assert (optimality != 0).any(), "wrong optimality score"

    if pos_ratio!=1:
        sel_ind = (optimality==15).nonzero()[0]
        sel_ind = sel_ind[:int(sel_ind.shape[0]*pos_ratio)]
        optimality[sel_ind] = 0 # not gonna use it

    obs_np = states
    clust_label = exp.emb_cluster(obs_np, cluster_num=cluster_num//2)
    clust_label[optimality==-15] += cluster_num//2


    if split is not None:
        batched_demons = []
        val_demons = []

        if split == 'all_pos':
            index_list = [optimality != 0, optimality == 15]
            for i,index in enumerate(index_list):
                Bi_demonstration = exp.traj_array2demon(states[index], actions[index], Dones[index], interval=left_interval,future_interval=right_interval, cluster_label=clust_label[index])
                if shuffle:
                    random.shuffle(Bi_demonstration)

                if val_ratio is not None:
                    demon_size = len(Bi_demonstration)
                    val_demonstration = Bi_demonstration[:int(demon_size*val_ratio)]
                    Bi_demonstration = Bi_demonstration[int(demon_size*val_ratio):]
                    if given_bs == -1:
                        batch_size = len(val_demonstration)
                    batched_val_demon = exp.batch_demon(val_demonstration, batch_size)
                    val_demons.append(batched_val_demon)

                if given_bs == -1:
                    batch_size = len(Bi_demonstration)
                batched_demon = exp.batch_demon(Bi_demonstration, batch_size)
                batched_demons.append(batched_demon)
            if return_scaler:
                return batched_demons, scaler
            return batched_demons
        elif split == 'pos_neg':
            index_list = [optimality == 15, optimality == -15]
            for i,index in enumerate(index_list):
                Bi_demonstration = exp.traj_array2demon(states[index], actions[index], Dones[index], interval=left_interval,
                                                    future_interval=right_interval, cluster_label=clust_label[index])
                if shuffle:
                    random.shuffle(Bi_demonstration)

                if val_ratio is not None:
                    demon_size = len(Bi_demonstration)
                    val_demonstration = Bi_demonstration[:int(demon_size*val_ratio)]
                    Bi_demonstration = Bi_demonstration[int(demon_size*val_ratio):]
                    if given_bs == -1:
                        batch_size = len(val_demonstration)
                    batched_val_demon = exp.batch_demon(val_demonstration, batch_size)
                    val_demons.append(batched_val_demon)

                if given_bs == -1:
                    batch_size = len(Bi_demonstration)
                batched_demon = exp.batch_demon(Bi_demonstration, batch_size)
                batched_demons.append(batched_demon)

            if val_ratio is not None:
                return val_demons, batched_demons
            elif return_scaler:
                return batched_demons, scaler
            return batched_demons

        else:
            print("not recognized split")
            ipdb.set_trace()

    else:
        index = optimality!=0
        Bi_demonstration =exp.traj_array2demon(states[index], actions[index], Dones[index], interval=left_interval,
                                                    future_interval=right_interval, cluster_label=clust_label[index])
        if shuffle:
            random.shuffle(Bi_demonstration)

        if val_ratio is not None:
            demon_size = len(Bi_demonstration)
            val_demonstration = Bi_demonstration[:int(demon_size * val_ratio)]
            Bi_demonstration = Bi_demonstration[int(demon_size * val_ratio):]
            if given_bs == -1:
                batch_size = len(val_demonstration)
            batched_val_demon = exp.batch_demon(val_demonstration, batch_size)


        if given_bs == -1:
            batch_size = len(Bi_demonstration)
        batched_demon = exp.batch_demon(Bi_demonstration, batch_size)

        if val_ratio is not None:
            return batched_val_demon, batched_demon
        elif return_scaler:
            return batched_demon, scaler
        return batched_demon

def load_split_sepsis(load_goals, traj_dir, batch_size=32, left_interval=1, right_interval=1,cluster_num=4,
                        shuffle=True, pos_ratio=1, reduce=1):
    """
    :param load_goals:
    :param traj_dir:
    :param batch_size:
    :param left_interval:
    :param right_interval:
    :param cluster_num:
    :param split: {None, 'pos_neg', 'all_pos'}. if None: return all batches as a single list.
        if 'pos_neg': return two lists, of pos/neg batches
        if 'all_pos': return two lists, of all batches and only expert batches
    :param shuffle: if True: batches will be shuffled
    :return: train; val; test data, in form of batched demonstrations
    """
    datas = []
    for goal in load_goals:
        data = np.genfromtxt(traj_dir+goal, delimiter=',')[1:,:]
        datas.append(data)
    data = np.concatenate(datas, axis=0)
    ipdb.set_trace()

    states = data[:,2:-2] #132358
    actions = data[:,-1].astype(int)  # .reshape(states.shape[0],-1)
    if reduce !=1:
        actions = (actions/reduce).astype(int)
    Dones = data[:,-2]!=0  # .reshape(states.shape[0],-1). sum: 10203
    pre_actions = copy.deepcopy(actions)
    pre_actions[Dones] = -1
    pre_actions[1:] = pre_actions[:-1]
    pre_actions[0] = -1
    states=np.concatenate((states, pre_actions.reshape(-1, 1)), axis=-1)

    # normalize features
    scaler = MinMaxScaler().fit(states)
    states = scaler.transform(states)

    optimality = data[:, -2]
    for i in range(optimality.shape[0], 0, -1):
        if optimality[i - 1] == 0:
            optimality[i - 1] = optimality[i]
    assert (optimality != 0).any(), "wrong optimality score"

    obs_np = states
    clust_label = exp.emb_cluster(obs_np, cluster_num=cluster_num//2)
    clust_label[optimality==-15] += cluster_num//2

    if pos_ratio!=1:
        sel_ind = (optimality==15).nonzero()[0]
        sel_ind = sel_ind[int(sel_ind.shape[0]*pos_ratio):]
        optimality[sel_ind] = 0

    index = optimality !=0
    Bi_demonstration = exp.traj_array2demon(states[index], actions[index], Dones[index], interval=left_interval,future_interval=right_interval, cluster_label=clust_label[index])
    if shuffle:
        random.shuffle(Bi_demonstration)
    demon_size = len(Bi_demonstration)

    train_demonstration = Bi_demonstration[:int(demon_size*0.5)]
    val_demonstration = Bi_demonstration[int(demon_size*0.5):int(demon_size*0.7)]
    test_demonstration = Bi_demonstration[int(demon_size*0.7):]

    batched_train_demon = exp.batch_demon(train_demonstration, batch_size)
    batched_val_demon = exp.batch_demon(val_demonstration, len(val_demonstration))
    batched_test_demon = exp.batch_demon(test_demonstration, len(test_demonstration))

    return batched_train_demon, batched_val_demon, batched_test_demon
