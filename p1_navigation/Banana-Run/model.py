import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import *

class DQN(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, fc_layers=[], seed=None):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc_layers (list of int): Dimension of each fully connected layer
            seed (int): Random seed
        """

        super(DQN, self).__init__()
        if seed != None: self.seed = torch.manual_seed(seed)
        combined_layers = [state_size] + fc_layers + [action_size]
        self.fcls = nn.ModuleList([nn.Linear(combined_layers[i], combined_layers[i+1]) for i in range(len(combined_layers)-1)])

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = reduce(lambda a,b: F.relu(b(a)), self.fcls[:-1], state)
        return self.fcls[-1](x)

