import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
import torch.nn as nn
import ipdb

class CondENet(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, layers=3, dropout=0):
        '''
        net for estimating conditional entropy, like I(z|s,a)
        :param in_dim:
        :param hid_dim:
        :param out_dim:
        :param layers:
        :param dropout:
        :return:
        '''
        super().__init__()

        modules = []
        in_size = in_dim
        for layer in range(layers - 1):
            modules.append(Linear(in_size, hid_dim))
            in_size = hid_dim
            modules.append(nn.LeakyReLU(0.1))
            modules.append(nn.Dropout(dropout))
        modules.append(Linear(in_size, out_dim))

        self.net = nn.Sequential(*modules)

    def forward(self, sources, targets=None):
        if type(sources) == list:
            sources = torch.cat(sources, dim=-1)

        if targets is not None:
            sources = torch.cat([sources, targets], dim=-1)

        return self.net(sources)
