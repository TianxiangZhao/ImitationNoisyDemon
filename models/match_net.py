import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
import torch.nn as nn
import math
import ipdb
import models.model_utils as m_utils


class MatchNet(torch.nn.Module):
    def __init__(self, proto_dim, proto_num=8, hard_sel=False, tensor_dis_func=None, temper=1):
        '''

        :param proto_dim:
        :param proto_num:
        :param hard_sel:  whether use hard gumble softmax
        :return:
        '''
        super().__init__()
        self.proto_memory = nn.Parameter(torch.zeros((proto_num, proto_dim)))
        self.hard_sel = hard_sel
        self.reset_parameters()
        self.tensor_dis_func = tensor_dis_func
        self.temper = temper

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.proto_memory, a=math.sqrt(5))

        return

    def forward(self, state, return_s=False, return_selected=False, fixed_skill=None, deterministic=False):
        """

        :param state:
        :param return_s:
        :param return_selected:
        :param fixed_skill: whether fix using the given skill variable
        :param deterministic: whether use gumbel softmax to sample skill, or use argmax
        :return:
        """
        if self.tensor_dis_func is None:
            s_matrix = torch.mm(state, self.proto_memory.T)
        else:
            s_matrix = self.tensor_dis_func(state, self.proto_memory)
        s_prob = torch.softmax(s_matrix/self.temper, dim=-1)

        # use gumbel softmax
        s_sampled = m_utils.gumbel_softmax(torch.log(s_prob+0.000001), hard=self.hard_sel)
        if deterministic:
            s_sampled.fill_(0)
            index = s_prob.argmax(dim=-1)
            for i, ind in enumerate(index):
                s_sampled[i,ind] +=1

        if fixed_skill is not None:
            s_sampled.fill_(0)
            s_sampled[:,fixed_skill] = 1
        output = torch.mm(s_sampled, self.proto_memory)

        if return_s:
            if return_selected:
                return output, s_prob, s_sampled
            return output, s_prob

        if return_selected:
            return output, s_sampled

        return output

