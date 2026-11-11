"""Neural network used by DQN.

Input 1: the 8x31x28 grid observation (walls, pellets, Pac-Man, ghosts).
Input 2: the 18 scalar features (direction, fright timer, phase, ...).

Three small convolutions turn the grid into 64x8x7 features, the scalar
vector is concatenated, and a 512-unit hidden layer produces four
Q-values (one per action).  This is deliberately small: Pac-Man's maze
is only 28x31, and a small net trains quickly on a laptop GPU.
"""

import torch
import torch.nn as nn

from .config import GRID_H, GRID_W, N_CHANNELS, NUM_ACTIONS, VEC_DIM


class QNetwork(nn.Module):
    def __init__(self, width=64):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(N_CHANNELS, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, width, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute the flattened convolution size from a dummy tensor so
        # the code stays correct if the maze size ever changes.
        with torch.no_grad():
            dummy = torch.zeros(1, N_CHANNELS, GRID_H, GRID_W)
            conv_out = self.conv(dummy).shape[1]

        self.head = nn.Sequential(
            nn.Linear(conv_out + VEC_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, NUM_ACTIONS),
        )

    def forward(self, grid, vec):
        features = self.conv(grid)
        features = torch.cat([features, vec], dim=1)
        return self.head(features)
