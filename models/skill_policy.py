import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
import torch.nn as nn


class SkillPolicy(torch.nn.Module):
    def __init__(self, skill_dim, state_dim, hid_dim, out_dim, layers=2, dropout=0):
        super().__init__()

        modules = []
        in_size = skill_dim + state_dim
        for layer in range(layers-1):
            modules.append(Linear(in_size, hid_dim))
            in_size = hid_dim
            modules.append(nn.LeakyReLU(0.1))
            modules.append(nn.Dropout(dropout))
        modules.append(Linear(in_size, out_dim))

        self.net = nn.Sequential(*modules)

    def forward(self, skill_embedding, state_embedding):
        feature = torch.cat((skill_embedding, state_embedding), dim=-1)

        return torch.softmax(self.net(feature), dim=-1)

