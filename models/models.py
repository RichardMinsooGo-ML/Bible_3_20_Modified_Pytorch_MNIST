import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class Feed_Forward_Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_dim):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.a1 = nn.Sigmoid()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.a2 = nn.Sigmoid()
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.a3 = nn.Sigmoid()
        self.l4 = nn.Linear(hidden_size, output_dim)

        self.layers = [self.l1, self.a1,
                       self.l2, self.a2,
                       self.l3, self.a3,
                       self.l4]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
