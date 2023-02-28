import torch
from torch import nn
import numpy as np
import copy

class PlayerNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.online = nn.Sequential(
            nn.Linear(input_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, output_dim)
        )
        self.target = copy.deepcopy(self.online)
        # Q_target parameters are frozen
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)
