import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
import torch.nn as nn


class MLPFeature(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, layers=2, dropout=0):
        super().__init__()

        modules = []
        in_size = in_dim
        for layer in range(layers-1):
            modules.append(Linear(in_size,hid_dim))
            in_size = hid_dim
            modules.append(nn.LeakyReLU(0.1))
            modules.append(nn.Dropout(dropout))
        modules.append(Linear(in_size, out_dim))

        self.net = nn.Sequential(*modules)

    def forward(self, observations):
        return self.net(observations)
