# change to iteratively PU + disentanglement
# enable only uni-directional pretraining+finetuning

import random

import gym
import numpy as np
import torch
import os
from torch.optim import Adam
import ipdb
import models
import utils.log_utils as log_utils
from algorithms import losses
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import copy
import utils.expert as exp
from plot.plot_minigrid import plot_grid
from plot.plot_tsne import *
from sklearn.metrics import roc_auc_score, f1_score
import time

def tensor_distance_cos(tensor1, tensor2):
    distance = [F.cosine_similarity(emb, tensor2) for emb in tensor1]
    distance = torch.stack(distance)

    return distance

def tensor_distance_euc(tensor1, tensor2):
    distance = [1/F.pairwise_distance(emb, tensor2)+1 for emb in tensor1]
    distance = torch.stack(distance)

    return distance

class SkillBC(object):
    def __init__(self, observation_space, action_space, demonstrations,
                 alg_args, load_best, action_in_mi=False,
                 test_demonstrations=None, logger=None, env=None, valid_demon=None, correct_softmax=False, extra_dim=0):
        """
        :param observation_space:
        :param action_space:
        :param demonstrations: batched dataset
        :param alg_args:
        :param load_best:
        :param action_in_mi:
        :param test_demonstrations: list of batched datasets. To support dataset-wise examination
        :param logger:
        """
        # set up models
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.args = alg_args
        self.demonstrations = demonstrations
        self.logger = logger
        self.test_demonstrations = test_demonstrations
        self.valid_demon = valid_demon
        self.load_best = load_best
        self.env=env

        if 'neg_K' in alg_args.keys():
            self.neg_K = alg_args['neg_K']
        else:
            self.neg_K = 10

        if alg_args['distance'] == 'cos':
            self.distance_func = tensor_distance_cos
            self.d_min = 0.2
        elif alg_args['distance'] == 'euc':
            self.distance_func = tensor_distance_euc
            self.d_min = 1.0
        else:
            raise NotImplementedError("not implemented distance type")
        self.use_mi_batch = alg_args['batch_mi']
        self.action_n = action_space.n

        self.feature_ext = models.MLPFeature(in_dim=alg_args['interval']*(np.prod(observation_space.shape)+extra_dim),
                                                         hid_dim=alg_args['hid_dim'], out_dim=alg_args['hid_dim'],
                                                         dropout=alg_args['dropout']).to(self.device)
        self.future_feature_ext = models.MLPFeature(in_dim=alg_args['future_interval'] * (np.prod(observation_space.shape)+extra_dim),
                                                         hid_dim=alg_args['hid_dim'], out_dim=alg_args['hid_dim'],
                                                         dropout=alg_args['dropout']).to(self.device)
        self.bi_net = models.BiPolicy(left_dim=alg_args['hid_dim'], right_dim=alg_args['hid_dim'],
                                                hid_dim=alg_args['hid_dim'], out_dim=alg_args['proto_dim'],
                                                dropout=alg_args['dropout']).to(self.device)
        self.uni_net = models.UniPolicy(in_dim=alg_args['hid_dim'],
                                                hid_dim=alg_args['hid_dim'], out_dim=alg_args['proto_dim'],
                                                dropout=alg_args['dropout']).to(self.device)
        self.match_net = models.MatchNet(proto_dim=alg_args['proto_dim'], proto_num=alg_args['proto_num'],
                                         hard_sel=alg_args['hard_sel'], temper=alg_args['temper'],
                                         tensor_dis_func=self.distance_func).to(self.device)

        self.skill_feature_ext = models.MLPFeature(in_dim=np.prod(observation_space.shape)+extra_dim,
                                                         hid_dim=alg_args['hid_dim'], out_dim=alg_args['hid_dim'],
                                                         dropout=alg_args['dropout']).to(self.device)
        self.skill_net = models.SkillPolicy(skill_dim=alg_args['proto_dim'], state_dim=alg_args['hid_dim'],
                                                         hid_dim=alg_args['hid_dim'], out_dim=action_space.n).to(self.device)
        self.action_in_mi = action_in_mi

        if self.args['index_mi']:
            state_size = 6
        else:
            state_size = int(np.prod(observation_space.shape))
        if action_in_mi:
            self.condE_head = models.CondENet(in_dim=state_size+action_space.n+alg_args['proto_dim'],
                                                         hid_dim=alg_args['hid_dim'], out_dim=1,
                                                         dropout=alg_args['dropout']).to(self.device) # give the score
        else:
            self.condE_head = models.CondENet(in_dim=state_size+alg_args['proto_dim'],
                                                         hid_dim=alg_args['hid_dim'], out_dim=1,
                                                         dropout=alg_args['dropout']).to(self.device) # give the score
        #'''

        self.models = [self.feature_ext, self.future_feature_ext, self.bi_net, self.uni_net, self.match_net,
                       self.skill_feature_ext, self.skill_net, self.condE_head]

        # set up optimizers
        self.optim_high = Adam(list(self.bi_net.parameters())+list(self.future_feature_ext.parameters())
                                +list(self.feature_ext.parameters())+list(self.uni_net.parameters()),
                            lr=self.args['lr'])
        self.optim_low = Adam(list(self.skill_net.parameters())+list(self.skill_feature_ext.parameters()),
                              lr=self.args['lr'])
        self.optim_proto = Adam(self.match_net.parameters(), lr=self.args['lr'])
        self.optim_condE = Adam(self.condE_head.parameters(), lr=self.args['lr'])
        self.DEC = 0
        self.PU_thresh = alg_args['PU_thresh']

        # losses
        self.task = alg_args['task']
        if self.task == 'cls':
            #self.loss_fn = torch.nn.CrossEntropyLoss()
            self.loss_fn = F.cross_entropy
        else:
            self.loss_fn = torch.nn.MSELoss()

        self.pre_trained = False
        self.best_content = None
        self.trained_bi = False

        self.skill_optimality = None
        self.KD_weight=1
        self.correct_softmax = correct_softmax

    def get_models(self):
        content = {}
        for i, model in enumerate(self.models):
            content['model{}'.format(i)] = copy.deepcopy(model.state_dict())

        return content

    def load_models(self, state_dicts):
        for i, model in enumerate(self.models):
            model.load_state_dict(state_dicts['model{}'.format(i)])

        return

    def infer(self, observation_tensor, future_tensor=None, use_bi=False, return_info=False, fixed_skill=None,
              deterministic=False):
        """

        :param observation_tensor:
        :param future_tensor:
        :param use_bi:
        :param return_info:
        :param fixed_skill:
        :param deterministic: whether sample skills in a deterministic manner
        :return:
        """
        high_state_emb = self.feature_ext(observation_tensor)
        if use_bi:
            assert future_tensor is not None, "future should not be None when use bi-direction"
            future_emb = self.future_feature_ext(future_tensor)
            pre_skill_emb = self.bi_net(high_state_emb, future_emb)
        else:
            pre_skill_emb = self.uni_net(high_state_emb)
        skill_emb, match_prob, selected_skill = self.match_net(
            pre_skill_emb, return_s=True, return_selected=True, fixed_skill=fixed_skill, deterministic=deterministic)

        cur_state_length = int(observation_tensor.shape[-1]//self.args['interval'])
        low_state_emb = self.skill_feature_ext(observation_tensor[:,-cur_state_length:])
        action_prob = self.skill_net(skill_embedding=skill_emb, state_embedding=low_state_emb,)

        if self.env is not None:
            if 'FourRoom' in type(self.env.unwrapped).__name__:
                action_mask = action_prob.new(action_prob.shape).fill_(1)
                action_mask[:,3:] = 0
                action_prob = action_prob*action_mask

        if return_info:
            info = {}
            info['pre_skill_emb'] = pre_skill_emb # inputs for the match_net, which should follow a Mixed_gaussian dist
            info['skill_emb'] = skill_emb # updated skill embedding of each instance after prototype-matching
            info['match_prob'] = match_prob # matching probability of each instance to each skill embeb prototype
            info['selected_skill'] = selected_skill
            info['low_emb'] = low_state_emb
            return action_prob, info

        return action_prob

    def draw_emb_skill(self, embs, skills, PCA=False):

        state_embs = np.concatenate(embs, axis=0)
        skills = np.concatenate(skills, axis=0)

        if PCA:
            fig = plot_pca(state_embs, skills)
        else:
            fig = plot_tsne(state_embs, skills)

        return fig

    def draw_skill_anchor(self, embs, actions, proto_emb=None, PCA=False, action_reduce=1):
        state_embs = np.concatenate(embs, axis=0)
        actions = np.concatenate(actions, axis=0)
        if proto_emb is None:
            idx = np.random.randint(state_embs.shape[0], size=8)
            anchor_emb = state_embs[idx,:]
        else:
            anchor_emb = proto_emb
        #
        actions = (actions/action_reduce).astype(int)

        if PCA:
            fig = plot_pca_anchored(state_embs, actions, anchor_emb)
        else:
            fig = plot_tsne_anchored(state_embs, actions, anchor_emb)

        return fig

    def draw_trajectories_skill(self, demonstrations, skills):
        height, width = self.env.env.grid.height, self.env.env.grid.width
        map_grid = self.env.grid.grid

        move_array = np.zeros((height, width, 4)) #
        skill_array = np.zeros((height, width, 8))
        for i, (demonstration, skill) in enumerate(zip(demonstrations, skills)):
            pos = demonstration['obs'][:,-4:-2]
            dir = demonstration['obs'][:,-2]*2+demonstration['obs'][:,-1]
            dir[dir==2] = 0
            dir[dir==-2] = 2
            dir[dir==-1] = 3
            act = demonstration['acts']

            for move in (act==2).nonzero()[0]:
                move_array[int(pos[move,1]), int(pos[move,0]), int(dir[move])] += 1
                skill_array[int(pos[move,1]), int(pos[move,0]), int(skill[move])] += 1

            if i > 50:
                break

        fig = plot_grid(height, width, map_grid, move_array, skill_array,skill_confidence=True)

        return fig

    def clear_grad(self):
        self.optim_proto.zero_grad()
        self.optim_low.zero_grad()
        self.optim_high.zero_grad()
        self.optim_condE.zero_grad()

        return

    def compute_mi(self, states, S_em, actions, cluster_label =None, DEC=0.0, random_prob=0.2, op_label=None):
        """
        optimize parameters related to conditional_entropy estimation
        :param demon_batch:
        :param use_bi:
        :param DEC: probability to use DEC instead of pre-defined cluster label
        :param random_prob: if use DEC, prob of use random negative pair
        :return:
        """
        if self.args['index_mi']:
            state_start = -6
        else:
            state_start = 0

        if op_label is None:# optimality label of each instance is not given
            if cluster_label is not None: # first k/2 clusters are positive, last are negative
                op_label = (cluster_label< self.args['proto_num']/2).float()
                op_label[op_label==0] = -1

        if self.action_in_mi:
            # condE_loss ver2:
            action_one_hot = actions.new(actions.shape[0],self.action_n).fill_(0).float()
            action_one_hot.scatter_add_(dim=-1, index = actions.reshape(actions.shape[0],-1),
                                        src=torch.ones((actions.shape[0],self.action_n), device=action_one_hot.device))
            pos_MI_score = -F.softplus(-self.condE_head([states.reshape(states.shape[0], self.args['interval'],-1)[:,-1,state_start:],
                                           action_one_hot], S_em)).mean()

            if True: # compute distance for sampling positive pairs
                if cluster_label is not None and random.random() > DEC:
                    distance = (cluster_label.reshape(-1,1) == cluster_label.reshape(1,-1)).float()
                    #for i in range(S_em.shape[0]):
                    #    candidate_idxs = (cluster_label==cluster_label[i]).nonzero().reshape(-1)
                    #    target_idx[i] = candidate_idxs[random.randint(0, len(candidate_idxs)-1)]
                else:
                    distance = self.distance_func(S_em.detach(), S_em.detach())
                    dist_mask = distance.new(distance.shape).fill_(0)
                    for idx in range(distance.shape[0]):
                        top_k = torch.topk(distance[idx], int(S_em.shape[0]/16)+1, largest=False)
                        dist_mask[idx].scatter_(0, top_k.indices, 1)
                    distance = dist_mask

                distance = torch.divide(distance, distance.sum(dim=-1).reshape(-1, 1))

            sampling_time = 3
            for sam in range(sampling_time):
                target_idx = torch.multinomial(distance, num_samples=1).reshape(-1)
                pos_MI_score += -F.softplus(
                    -self.condE_head([states.reshape(states.shape[0], self.args['interval'], -1)[:, -1, state_start:],
                                      action_one_hot], S_em[target_idx])).mean()
            pos_MI_score = pos_MI_score / (sampling_time + 1)

            neg_MI_score = 0
            if True: # sampling negative pairs
                if cluster_label is not None and random.random()>DEC:
                    # sample negative from those in different pseudo cluster
                    distance = (cluster_label.reshape(-1,1) != cluster_label.reshape(1,-1)).float()
                else:
                    distance = self.distance_func(S_em.detach(), S_em.detach())
                    if op_label is not None:
                        dist_mask = (op_label.reshape(-1,1) == -op_label.reshape(1,-1)).float()
                    else:
                        dist_mask = distance.new(distance.shape).fill_(0)
                    for idx in range(distance.shape[0]):
                        top_k = torch.topk(distance[idx], int(S_em.shape[0]/4*3), largest=True)
                        dist_mask[idx].scatter_(0, top_k.indices, 1)
                    distance = dist_mask

                distance = torch.divide(distance, distance.sum(dim=-1).reshape(-1, 1))

            sampling_time = 6
            for i in range(sampling_time):
                target_idx = torch.multinomial(distance, num_samples=1).reshape(-1)
                candidate_S_em = S_em[target_idx]
                neg_MI_score += F.softplus(self.condE_head([states.reshape(states.shape[0], self.args['interval'],-1)[:,-1,state_start:],
                                           action_one_hot], candidate_S_em)).mean()

        else:
            pos_MI_score = -F.softplus(
                -self.condE_head([states.reshape(states.shape[0], self.args['interval'], -1)[:, -1, state_start:],
                                  ], S_em)).mean()

            if True:  # compute distance for sampling positive pairs
                if cluster_label is not None and random.random() > DEC:
                    distance = (cluster_label.reshape(-1, 1) == cluster_label.reshape(1, -1)).float()
                    # for i in range(S_em.shape[0]):
                    #    candidate_idxs = (cluster_label==cluster_label[i]).nonzero().reshape(-1)
                    #    target_idx[i] = candidate_idxs[random.randint(0, len(candidate_idxs)-1)]
                else:
                    distance = self.distance_func(S_em.detach(), S_em.detach())
                    dist_mask = distance.new(distance.shape).fill_(0)
                    for idx in range(distance.shape[0]):
                        top_k = torch.topk(distance[idx], 4, largest=False)
                        dist_mask[idx].scatter_(0, top_k.indices, 1)
                    distance = dist_mask
                distance = torch.divide(distance, distance.sum(dim=-1).reshape(-1, 1))

            sampling_time = 2
            for sam in range(sampling_time):
                target_idx = torch.multinomial(distance, num_samples=1).reshape(-1)
                pos_MI_score += -F.softplus(
                    -self.condE_head([states.reshape(states.shape[0], self.args['interval'], -1)[:, -1, state_start:],
                                      ], S_em[target_idx])).mean()
            pos_MI_score = pos_MI_score / (sampling_time + 1)

            neg_MI_score = 0
            if True:  # sampling negative pairs
                if cluster_label is not None and random.random() > DEC:
                    # sample negative from those in different pseudo cluster
                    distance = (cluster_label.reshape(-1, 1) != cluster_label.reshape(1, -1)).float()
                else:
                    distance = self.distance_func(S_em.detach(), S_em.detach())
                    if cluster_label is not None:

                        dist_mask = (op_label.reshape(-1, 1) == -op_label.reshape(1, -1)).float()
                    else:
                        dist_mask = distance.new(distance.shape).fill_(0)
                    for idx in range(distance.shape[0]):
                        top_k = torch.topk(distance[idx], self.neg_K, largest=True)
                        dist_mask[idx].scatter_(0, top_k.indices, 1)
                    distance = dist_mask

                distance = torch.divide(distance, distance.sum(dim=-1).reshape(-1, 1))

            sampling_time = 6
            for i in range(sampling_time):
                target_idx = torch.multinomial(distance, num_samples=1).reshape(-1)
                candidate_S_em = S_em[target_idx]
                neg_MI_score += F.softplus(
                    self.condE_head([states.reshape(states.shape[0], self.args['interval'], -1)[:, -1, state_start:],
                                     ], candidate_S_em)).mean()

        MI_est = pos_MI_score - neg_MI_score/sampling_time
        condE_loss = -MI_est

        return condE_loss

    def train_batch(self, demon_batch, use_bi=False, opt_all=True, weight=None):
        """

        :param demon_batch:
        :param use_bi:
        :return:
        """
        for model in self.models:
            model.train()

        # wrap observation as torch.tensor
        states = torch.tensor(demon_batch['obs'], dtype=torch.float, device=self.device)# batch, interval*state_dim
        if self.task == 'cls':
            actions = torch.tensor(demon_batch['acts'], dtype=torch.long, device=self.device)
        else:
            ipdb.set_trace()
            actions = torch.tensor(demon_batch['acts'], dtype=torch.float, device=self.device)
        future_states = torch.tensor(demon_batch['next_obs'], dtype=torch.float, device=self.device)
        clust_label = torch.tensor(demon_batch['cluster_label'], dtype=torch.long, device=self.device)

        # infer action prediction
        action_dist, info = self.infer(states, future_tensor=future_states, use_bi=use_bi, return_info=True)
        if (action_dist!=action_dist).any():
            print("Irregular output")
            ipdb.set_trace()

        # -------------
        # calculate loss
        # -------------
        if use_bi is False and self.trained_bi: # to adapt the high-level policy with knowledge distill
            with torch.no_grad():
                _, teacher_info = self.infer(states, future_tensor=future_states, use_bi=True, return_info=True)
            teacher_z_label = teacher_info['match_prob'].detach().argmax(dim=-1).long()
            S_mat = info['match_prob']
            if self.correct_softmax:
                skill_distill_loss = F.nll_loss(torch.log(S_mat+0.0000001), teacher_z_label)
            else:
                skill_distill_loss = F.cross_entropy(S_mat, teacher_z_label)
        else:
            if self.correct_softmax:
                skill_distill_loss = F.nll_loss(torch.log(action_dist+0.0000001), actions).detach().fill_(0)
            else:
                skill_distill_loss = self.loss_fn(action_dist, actions).detach().fill_(0)

        # prob true action:
        avg_true_prob = torch.gather(action_dist, -1, actions.reshape(actions.shape[0],1)).detach().mean()
        prob_true_action = (action_dist.argmax(-1) == actions).sum()/action_dist.shape[0]

        # imitation learning loss: bc
        if weight is not None and self.args['PU']:
            weight_tensor = torch.tensor(weight, dtype=torch.float, device=self.device)

            pos_ind = weight_tensor>0.25
            neg_ind = weight_tensor<-0.25
            if pos_ind.sum()>0:
                if self.correct_softmax:
                    loss_bc_pos_ins = F.nll_loss(torch.log(action_dist[pos_ind]+0.0000001), actions[pos_ind], reduction='none')
                else:
                    loss_bc_pos_ins = self.loss_fn(action_dist[pos_ind], actions[pos_ind], reduction='none')
                loss_bc_pos = torch.mul(loss_bc_pos_ins, weight_tensor[pos_ind]).sum()
            else:
                loss_bc_pos = 0
            if neg_ind.sum()>0:
                loss_bc_neg = torch.gather(action_dist[neg_ind],index = actions[neg_ind].reshape(-1,1), dim=-1).sum()/neg_ind.sum()
            else:
                loss_bc_neg = 0

            loss_bc = loss_bc_pos + loss_bc_neg
            if pos_ind.sum()==0 and neg_ind.sum()==0:
                print("non-informative batch")
                loss_bc = self.loss_fn(action_dist, actions)*0

        else:
            if self.correct_softmax:
                loss_bc = F.nll_loss(torch.log(action_dist+0.0000001), actions)
            else:
                loss_bc = self.loss_fn(action_dist, actions)

        if not self.pre_trained:
            # MI(Z, (s,a))
            S_mat = info['match_prob']
            S_entropy_batch = -torch.mul(S_mat.mean(dim=0), torch.log(S_mat.mean(dim=0))).sum()

            # get pseudo group label
            if self.skill_optimality is not None and self.args['use_DEC']:
                selected_skill = info['selected_skill'].argmax(dim=-1)
                true_prob = torch.gather(action_dist, -1, actions.reshape(actions.shape[0], 1)).detach().reshape(-1)
                skill_optimality = torch.FloatTensor(self.skill_optimality).to(true_prob.device)
                op_score = skill_optimality[selected_skill] * true_prob
                op_score[clust_label<self.args['proto_num']/2] = 1
                op_score[torch.logical_and(op_score < 0.1, op_score > -0.1)] = 0
                op_score[op_score > 0.1] = 1
                op_score[op_score < -0.1] = -1
                op_label = op_score.float()
            else:
                op_label = None

            condE_loss = self.compute_mi(states=states, S_em=info['skill_emb'], actions=actions,
                                         cluster_label=clust_label, DEC=self.DEC, op_label=op_label)

            # clustering loss
            proto_sets = self.match_net.proto_memory
            proto_dis = self.distance_func(proto_sets, proto_sets)
            proto_div_loss = torch.triu(torch.clamp(self.d_min-proto_dis, 0), diagonal=1).sum()
            S_entropy_ins = -torch.mul(S_mat, torch.log(S_mat+0.0000001)).mean(dim=0).sum()

            # cluster pseudo label
            if self.correct_softmax:
                cluster_loss = F.nll_loss(torch.log(S_mat+0.0000001), clust_label)
            else:
                cluster_loss = F.cross_entropy(S_mat,clust_label)

            # aggregate loss
            if not self.use_mi_batch:
                S_entropy_batch = S_entropy_batch.detach()
            losses = loss_bc + self.args['lambda_mi']*(condE_loss-S_entropy_batch)+self.args['lambda_clus']*S_entropy_ins + \
                     self.args['lambda_pseudo'] * cluster_loss
                     # self.args['lambda_clus']*(proto_div_loss+S_entropy_ins) + \

        else:
            losses = loss_bc

        if use_bi is False and self.trained_bi:
            losses += skill_distill_loss*self.KD_weight

        # -------------
        # optimize
        # -------------
        #
        self.clear_grad()
        losses.backward()

        if opt_all:
            self.optim_proto.step()
            self.optim_low.step()
        self.optim_condE.step()
        self.optim_high.step()

        # -------------
        # log
        # -------------
        #
        '''
        print("bc loss: {}, condE loss: {}, S_entropy_batch: {}, proto_div_loss: {}, S_entropy_ins: {}".format(
            loss_bc.item(), condE_loss.item(), S_entropy_batch.item(), proto_div_loss.item(), S_entropy_ins.item()
        ))
        '''
        if not self.pre_trained:
            logs = {'bc loss':loss_bc.item(), 'condE loss': condE_loss.item(), 'S_entropy_batch': S_entropy_batch.item(),
                'proto_div_loss': proto_div_loss.item(), 'S_entropy_ins': S_entropy_ins.item(),
                'prob_true_act': prob_true_action.item(),
                'pseudo cluster loss': cluster_loss.item(),'avg_true_prob': avg_true_prob.item(), }
        else:
            logs = {'bc loss': loss_bc.item(), 'prob_true_act': prob_true_action.item(),
                    'avg_true_prob': avg_true_prob.item(), }
            logs['skill_distill_loss'] = skill_distill_loss.item()

        S_dist = info['match_prob'].mean(dim=0)
        for i, s_value in enumerate(S_dist):
            logs['prob_P{}'.format(i)] = s_value.item()

        return logs

    def train_pseudo(self, demon_batch, use_bi=False,):
        """

        :param demon_batch:
        :param use_bi:
        :return:
        """
        for model in self.models:
            model.train()

        # wrap observation as torch.tensor
        states = torch.tensor(demon_batch['obs'], dtype=torch.float, device=self.device)# batch, interval*state_dim
        future_states = torch.tensor(demon_batch['next_obs'], dtype=torch.float, device=self.device)
        clust_label = torch.tensor(demon_batch['cluster_label'], dtype=torch.long, device=self.device)
        actions = torch.tensor(demon_batch['acts'], dtype=torch.long, device=self.device)

        # infer action prediction
        action_dist, info = self.infer(states, future_tensor=future_states, use_bi=use_bi, return_info=True)

        # -------------
        # calculate loss
        # -------------
        # pseudo cluster loss
        S_mat = info['match_prob']
        if self.correct_softmax:
            cluster_loss = F.nll_loss(torch.log(S_mat+0.0000001), clust_label)
        else:
            cluster_loss = F.cross_entropy(S_mat,clust_label)

        # diversity inside batch
        S_entropy_batch = -torch.mul(S_mat.mean(dim=0), torch.log(S_mat.mean(dim=0))).sum()

        # MI loss
        condE_loss = self.compute_mi(states=states, S_em=info['skill_emb'], actions=actions,
                                    cluster_label = clust_label)

        # clustering loss
        proto_sets = self.match_net.proto_memory
        proto_dis = self.distance_func(proto_sets, proto_sets)
        proto_div_loss = torch.triu(torch.clamp(self.d_min-proto_dis, 0), diagonal=1).sum()
        S_entropy_ins = -torch.mul(S_mat, torch.log(S_mat)).mean(dim=0).sum()

        # consistency in skill loss
        S_consist_loss = (S_mat[1:, :].detach() - S_mat[:-1, :]).abs().mean()


        if use_bi is False and self.trained_bi: # to adapt the high-level policy with knowledge distill
            with torch.no_grad():
                _, teacher_info = self.infer(states, future_tensor=future_states, use_bi=True, return_info=True)
            teacher_z_label = teacher_info['match_prob'].detach().argmax(dim=-1).long()
            S_mat = info['match_prob']
            if self.correct_softmax:
                skill_distill_loss = F.nll_loss(torch.log(S_mat+0.0000001), teacher_z_label)
            else:
                skill_distill_loss = F.cross_entropy(S_mat, teacher_z_label)
        else:
            skill_distill_loss = None

        # aggregate loss

        if not self.use_mi_batch:
            S_entropy_batch = S_entropy_batch.detach()
        losses = self.args['lambda_mi'] * (condE_loss - S_entropy_batch) + \
                 self.args['lambda_clus'] * (proto_div_loss + S_entropy_ins) + \
                 self.args['lambda_consist'] * S_consist_loss + self.args['lambda_pseudo'] * cluster_loss
        if use_bi is False and self.trained_bi:
            losses += skill_distill_loss*self.KD_weight

        # -------------
        # optimize
        # -------------
        #
        self.clear_grad()
        losses.backward()

        self.optim_proto.step()
        self.optim_low.step()
        self.optim_high.step()
        self.optim_condE.step()

        # -------------
        # log
        # -------------
        #
        '''
        print("pseudo loss: {}, mi loss {}, cluster loss{} ".format(cluster_loss.item(), condE_loss.item(),
                                                                    (proto_div_loss + S_entropy_ins).item(),)
              )
        '''
        logs = {
                'pseudo cluster loss': cluster_loss.item(),}
        return logs

    def test_batch(self, demon_batch, use_bi=False, return_info = False, deterministic=False):
        for model in self.models:
            model.eval()

        # wrap observation as torch.tensor
        states = torch.tensor(demon_batch['obs'], dtype=torch.float, device=self.device)# batch, interval*state_dim
        if self.task == 'cls':
            actions = torch.tensor(demon_batch['acts'], dtype=torch.long, device=self.device)
        else:
            ipdb.set_trace()
            actions = torch.tensor(demon_batch['acts'], dtype=torch.float, device=self.device)
        clust_label = torch.tensor(demon_batch['cluster_label'], dtype=torch.long, device=self.device)
        future_states = torch.tensor(demon_batch['next_obs'], dtype=torch.float, device=self.device)

        # infer action prediction
        action_dist, info = self.infer(states, future_tensor=future_states, use_bi=use_bi, return_info=True,
                                       deterministic=deterministic)

        info['true_action_prob'] = torch.gather(action_dist, -1, actions.reshape(actions.shape[0],1)).detach().reshape(-1)

        # -------------
        # calculate loss
        # -------------
        # prob true action:
        avg_true_prob = torch.gather(action_dist, -1, actions.reshape(actions.shape[0],1)).detach().mean()
        prob_true_action = (action_dist.argmax(-1) == actions).sum()/action_dist.shape[0]

        # imitation learning loss: bc
        if self.correct_softmax:
            loss_bc = F.nll_loss(torch.log(action_dist+0.0000001), actions)
        else:
            loss_bc = self.loss_fn(action_dist, actions)

        # MI(Z, (s,a))
        S_mat = info['match_prob']
        S_entropy_batch = -torch.mul(S_mat.mean(dim=0), torch.log(S_mat.mean(dim=0))).sum()

        # clustering loss
        proto_sets = self.match_net.proto_memory
        proto_dis = self.distance_func(proto_sets, proto_sets)
        proto_div_loss = torch.triu(torch.clamp(self.d_min-proto_dis, 0), diagonal=1).sum()
        S_entropy_ins = -torch.mul(S_mat, torch.log(S_mat)).mean(dim=0).sum()

        # -------------
        # log
        # -------------
        #
        #'''
        print("test bc loss: {}, prob_true_act loss: {}, S_entropy_batch: {}, proto_div_loss: {}, S_entropy_ins: {}".format(
            loss_bc.item(), prob_true_action.item(), S_entropy_batch.item(), proto_div_loss.item(), S_entropy_ins.item()
        ))
        #'''
        logs = {'bc loss':loss_bc.item(), 'S_entropy_batch': S_entropy_batch.item(),
                'proto_div_loss': proto_div_loss.item(), 'S_entropy_ins': S_entropy_ins.item(),
                'prob_true_act': prob_true_action.item(), 'avg_true_prob':avg_true_prob.item(), }

        #'''
        S_dist = S_mat.mean(dim=0)
        for i, s_value in enumerate(S_dist):
            logs['prob_P{}'.format(i)] = s_value.item()
        #'''

        # add auxiliary metrics to logs
        if actions.shape[0]>200:
            if action_dist.shape[-1] != 7: #not for Fourroom

                auc_score = roc_auc_score(actions.detach().cpu(), action_dist.detach().cpu(), average='macro', multi_class='ovr')
                logs['acroc'] = auc_score

        macro_F = f1_score(actions.detach().cpu(), torch.argmax(action_dist,dim=-1).detach().cpu(), average='macro')
        logs['macro_F'] = macro_F

        if return_info:
            return logs, info

        return logs


    def train_all(self, epochs=10, use_bi=False, eval_env = None, prefix='all', opt_all=True, weights=None, PU_interval=-1,
                  expert_batches=None, unknown_batches=None):
        """
        :param epochs:
        :param use_bi:
        :param eval_env:
        :param prefix:
        :param opt_all: if true: optimize all parameters; else, only optimize the high-level policy
        :return:
        """
        batch_num = len(self.demonstrations)
        best_reward = -1000
        best_content = {}

        for k in range(epochs):
            # test
            time_start = time.time()
            reward = None
            if self.test_demonstrations is not None and k%1==0:
                if self.valid_demon is not None:
                    log_valid = {}
                    print("valid after epoch {}".format(k))
                    for data in self.valid_demon:
                        with torch.no_grad():
                            logs = self.test_batch(data, use_bi=use_bi, return_info=False)
                            for key in logs:
                                if key not in log_valid:
                                    log_valid[key] = log_utils.meters(orders=1)
                                log_valid[key].update(logs[key], 1)

                    for key in logs:
                        if self.logger is not None:
                            self.logger.add_scalar('valid_{}/'.format(prefix) + key, log_valid[key].avg(), k * batch_num)

                print("test after epoch {}".format(k))
                selected_skill_lists = []
                low_emb_lists = []
                for i, test_demonstration in enumerate(self.test_demonstrations):
                    log_test = {}
                    selected_skills = []
                    low_embs = []
                    gt_actions = []
                    pre_skill_embs = []

                    for data in test_demonstration:
                        with torch.no_grad():
                            time_test_batch_start = time.time()
                            logs, info = self.test_batch(data, use_bi=use_bi, return_info=True)
                            #print("test batch time: {}".format(time.time()-time_test_batch_start))
                        for key in logs:
                            if key not in log_test:
                                log_test[key] = log_utils.meters(orders=1)
                            log_test[key].update(logs[key], 1)
                        selected_skills.append(info['selected_skill'].argmax(dim=-1).cpu().numpy())
                        gt_actions.append(data['acts'])
                        low_embs.append(info['low_emb'].cpu().numpy())
                        pre_skill_embs.append(info['pre_skill_emb'].cpu().numpy())

                    if reward is None: # log the reward as performance on 1st test set,
                        if 'acroc' in log_test.keys():
                            reward = log_test['acroc'].avg()
                        else:
                            reward = -log_test['bc loss'].avg()

                    for key in logs:
                        if self.logger is not None:
                            self.logger.add_scalar('test_{}_set{}/'.format(prefix, i) + key, log_test[key].avg(), k * batch_num)

                    if k%10 ==0: # draw learned skills
                        if self.env is not None: # can directly draw traj-skill distribution by vis envs
                            #collect skill selection information
                            fig = self.draw_trajectories_skill(test_demonstration,selected_skills)
                            if self.logger is not None:
                                self.logger.add_image('test_{}_set{}/skill_selection'.format(prefix, i),
                                                      fig.transpose((2, 0, 1)), k * batch_num)
                        if True:
                            #selected_skill_lists.append(selected_skills) #for visualizing together. not used for now
                            #low_emb_lists.append(low_embs)
                            vis_batch = 30
                            match_prob = info['match_prob']
                            sel_idx = match_prob.detach().argmax(dim=0)
                            proto_emb = info['pre_skill_emb'][sel_idx].detach().cpu().numpy()

                            # visualize skill selection in embeddings space
                            if self.env is None: # reducing number of actions for visualization
                                action_reduce = 1
                            else:
                                action_reduce = 1
                            fig = self.draw_skill_anchor(pre_skill_embs[:vis_batch], gt_actions[:vis_batch], proto_emb=proto_emb, PCA=False, action_reduce=action_reduce)
                            if self.logger is not None:
                                self.logger.add_figure('test_{}_set{}_TSNE/skill_anchor'.format(prefix, i),
                                                fig, k * batch_num)
                            fig = self.draw_skill_anchor(pre_skill_embs[:vis_batch], gt_actions[:vis_batch],  proto_emb=proto_emb, PCA=True, action_reduce=action_reduce)
                            if self.logger is not None:
                                self.logger.add_figure('test_{}_set{}_PCA/skill_anchor'.format(prefix, i),
                                                       fig, k * batch_num)

                            fig = self.draw_skill_anchor(low_embs[:vis_batch], gt_actions[:vis_batch],
                                                         proto_emb=None, PCA=False, action_reduce=action_reduce)
                            if self.logger is not None:
                                self.logger.add_figure('test_{}_set{}_TSNE/policy_anchor'.format(prefix, i),
                                                       fig, k * batch_num)
                            fig = self.draw_skill_anchor(low_embs[:vis_batch], gt_actions[:vis_batch],
                                                         proto_emb=None, PCA=True, action_reduce=action_reduce)
                            if self.logger is not None:
                                self.logger.add_figure('test_{}_set{}_PCA/policy_anchor'.format(prefix, i),
                                                       fig, k * batch_num)

                            """
                            fig = self.draw_emb_skill(low_embs[:vis_batch], selected_skills[:vis_batch], PCA=False)
                            if self.logger is not None:
                                self.logger.add_figure('test_{}_set{}/skill_selection_Tsne'.format(prefix, i),
                                                fig, k * batch_num)
                            """
                            fig = self.draw_emb_skill(low_embs[:vis_batch], selected_skills[:vis_batch], PCA=True)
                            if self.logger is not None:
                                self.logger.add_figure('test_{}_set{}/skill_selection_PCA'.format(prefix, i),
                                                fig, k * batch_num)

                # overwrite reward
                if eval_env is not None and use_bi is not True:
                    reward, std = self.evaluate_env(eval_env, n_eval_episodes=100, use_bi=False)
                    if self.logger is not None:
                        self.logger.add_scalar('test_{}/reward'.format(prefix), reward, k*batch_num)
                        self.logger.add_scalar('test_{}/reward_std'.format(prefix), std, k*batch_num)

                if reward >= best_reward:
                    best_reward = reward
                    best_content = self.get_models()

            time_test = time.time()
            #print("test time: {}".format(time_test-time_start))
            # train
            for b, batched_demon in enumerate(self.demonstrations):
                if weights is not None:
                    time_train_batch_start = time.time()
                    logs = self.train_batch(batched_demon, use_bi=use_bi, opt_all=opt_all, weight=weights[b])
                    #print("train batch time: {}".format(time.time() - time_train_batch_start))
                else:
                    print("start batch {}".format(b))
                    time_train_batch_start = time.time()
                    logs = self.train_batch(batched_demon, use_bi=use_bi, opt_all=opt_all)
                    #print("train batch time: {}".format(time.time() - time_train_batch_start))

                # log batch-wise results
                for key in logs:
                    if self.logger is not None:
                        self.logger.add_scalar('{}/'.format(prefix) + key, logs[key], k * batch_num + b)
                if self.logger is not None:
                    self.logger.add_scalar('{}/'.format(prefix)+'DEC', self.DEC, k * batch_num + b)
            if self.DEC < 0.7 and self.args['use_DEC']:
                #self.DEC += 0.03
                self.DEC += 0.02
            else:
                self.DEC = 0

            # update PU pseudo
            if PU_interval != -1:
                if k!=0 and k%PU_interval ==0:
                    self.propagate_score(expert_batches=expert_batches, unknown_batches=unknown_batches, use_bi=use_bi,
                                         threshold=self.PU_thresh)

        if self.load_best:
            self.load_models(best_content)
        self.best_content = best_content

        if use_bi:
            self.trained_bi = True

        return best_reward

    def get_skill_optimality(self, expert_batches, unknown_batches, use_bi=True):
        # use bi-directional skill discovery to give scores to skills
        #assert self.pre_trained is True

        skill_selection_count = np.zeros((self.args['proto_num'],2))
        skill_acc_prob = np.zeros((self.args['proto_num'],2))

        for i, batches in enumerate([expert_batches, unknown_batches]):
            for data in batches:
                with torch.no_grad():
                    logs, info = self.test_batch(data, use_bi=use_bi, return_info=True)
                selected_skill = info['selected_skill'].argmax(dim=-1).cpu().numpy()
                skill_act_prob = info['true_action_prob'].cpu().numpy()
                np.add.at(skill_selection_count[:,i], selected_skill.ravel(), 1)
                np.add.at(skill_acc_prob[:,i], selected_skill.ravel(), skill_act_prob)

        skill_acc_prob = np.divide(skill_acc_prob, skill_selection_count+0.0001)

        expert_preference = np.divide(skill_selection_count[:,0]/skill_selection_count[:,0].sum()-skill_selection_count[:,1]/skill_selection_count[:,1].sum(),
                                      skill_selection_count[:,0]/skill_selection_count[:,0].sum()+1)
        optimality_score = expert_preference*skill_acc_prob[:,0]
        normalized_optimality_score = optimality_score/np.linalg.norm(optimality_score)

        self.skill_optimality = normalized_optimality_score

        return

    def propagate_score(self, expert_batches, unknown_batches, threshold=0.1, use_bi=True):
        # use bi-directional skill discovery to give scores to unknown batches
        # assert self.pre_trained is True

        # propagate score from expert batches to skills
        self.get_skill_optimality(expert_batches=expert_batches, unknown_batches=unknown_batches, use_bi=use_bi)

        # update socres for unknown batches based on selected skills and action prediction
        assert self.skill_optimality is not None

        optimality_scores = []
        for data in unknown_batches:
            with torch.no_grad():
                logs, info = self.test_batch(data, use_bi=use_bi, return_info=True)
            selected_skill = info['selected_skill'].argmax(dim=-1).cpu().numpy()
            skill_act_prob = info['true_action_prob'].cpu().numpy()
            op_score = self.skill_optimality[selected_skill]*skill_act_prob
            op_score[np.logical_and(op_score<threshold, op_score>-threshold)] = 0

            optimality_scores.append(op_score)
        return optimality_scores

    def evaluate_env(self, env, n_eval_episodes=50, use_bi=False):
        """
        Evaluate learned action prediction on real environments
        :param env:
        :param n_eval_episodes:
        :return:
        """
        def predict_bi(observation, deterministic=False):
            # not implemented yet, no
            ipdb.set_trace()

            obs = observation.reshape(observation.size(0), -1)
            out = self.infer(obs.float(), future_tensor=None, use_bi=True, return_info=False).argmax(dim=-1)

        def predict_uni(observation, deterministic=False):
            obs = observation.reshape(observation.size(0), -1) # if FrameStack is 5, VecEnv is 3, will be (3, 5, xxx)
            out = self.infer(obs.float(), future_tensor=None, use_bi=False, return_info=False).argmax(dim=-1)
            return out

        model = PPO("MlpPolicy", env, device=self.device)
        if use_bi:
            model.policy._predict = predict_bi
        else:
            model.policy._predict = predict_uni
        #model.policy._predict = MethodType(predict_bi, model.policy)
        mean_reward, std_reward = evaluate_policy(model=model, env=env, n_eval_episodes=n_eval_episodes)
        return mean_reward, std_reward

    def collect_Grid_trajectories(self, env, max_length=128, deterministic=False, traj_num=100,
                                  state_ini=None, fix_skill=None, show_max_part=False):
        """
        collect trajectories for visualization of minigrid env
        :param env: cannot be a dummy env
        :param max_length:
        :param deterministic:
        :param traj_num:
        :param state_ini:
        :param fix_skill: index of the fixed skill to use. if do not pre-set the skills, leave it as None.
        :param show_max_part: only used with fix_skill set. Only draw the part fix_skill will be selected by model
        :return: trajPos, trajDirect, trajR, trajAct, traj_skill
        """

        pos_list, direct_list, R_list, A_list, Z_list = [], [], [], [], []
        times = 0
        while len(pos_list) < traj_num:
            times +=1
            pos, direct, R, A, Z = [], [], [], [], []

            obs = env.reset()
            if state_ini is not None:
                obs = exp.set_env_state(env, state_ini)
            done = False
            length = 0
            while not done and length < max_length:
                if 'LazyFrames' in type(obs).__name__:
                    obs = np.array(obs)
                obs = obs.reshape((1,-1)) # only one environment
                obs = torch.tensor(obs, dtype=torch.float, device=self.device)

                # on policy
                with torch.no_grad():
                    action, info = self.infer(obs, use_bi=False, return_info=True, fixed_skill=fix_skill,
                                              deterministic=deterministic)
                act = action.argmax(dim=-1).item()
                skill_selected = info['selected_skill']
                if show_max_part and fix_skill is not None: #stop if selected skill is not fix_skill
                    if info['match_prob'].argmax(-1).item() != skill_selected.argmax(-1).item():
                        break
                pos.append(env.agent_pos)
                direct.append(env.agent_dir)
                obs_new, rew, done, _ = env.step(act)  # return np.array
                # store the corresponding trajectory
                A.append(act)
                R.append(rew)
                Z.append(skill_selected.argmax(dim=-1).item())

                # on to the next step
                obs = obs_new
                length = length + 1

            if len(pos)!=0:
                pos_list.append(np.stack(pos))  # shape: [step, 2]
                A_list.append(np.array(A))  # shape: [step,]
                R_list.append(np.array(R).sum())  # shape: integer
                direct_list.append(np.array(direct))  # shape: [step, ]
                Z_list.append(np.array(Z))

            if times > 10000:
                if len(pos) !=0:
                    break
                else:
                    show_max_part = False # give up, cannot collect sufficient number of trajectories

        return pos_list, direct_list, R_list, A_list, Z_list

