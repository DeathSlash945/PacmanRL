"""Neural network used by DQN.

Input 1: the 8x31x28 grid observation (walls, pellets, Pac-Man, ghosts).
Input 2: the 18 scalar features (direction, fright timer, phase, ...).
Input 3 (optional): the 16 ghost-direction features.

Three small convolutions turn the grid into a spatial feature map, the
scalar vector is concatenated, and a 512-unit trunk feeds a dueling head.
The head splits into one state-value output and four action-advantage
outputs; the Q-values are ``V + (A - mean(A))``.  Dueling helps in Pac-Man
because many states have nearly identical action values (a corridor with
no ghost nearby), while a few states need a very different value for one
escape action.  A small net keeps training fast on a laptop GPU.

The optional ghost-direction branch is kept separate so old checkpoints
(which have no such branch) still load with exactly the same parameter
names and shapes.
"""

import torch
import torch.nn as nn

from .config import (
    GHOST_DIR_DIM,
    GRID_H,
    GRID_W,
    N_CHANNELS,
    NUM_ACTIONS,
    VEC_DIM,
)


class QNetwork(nn.Module):
    def __init__(self, width=64, use_ghost_dir=False):
        super().__init__()
        self.use_ghost_dir = bool(use_ghost_dir)

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

        ghost_features = 64 if self.use_ghost_dir else 0
        if self.use_ghost_dir:
            self.ghost_mlp = nn.Sequential(
                nn.Linear(GHOST_DIR_DIM, 64),
                nn.ReLU(),
                nn.Linear(64, ghost_features),
                nn.ReLU(),
            )

        self.trunk = nn.Sequential(
            nn.Linear(conv_out + VEC_DIM + ghost_features, 512),
            nn.ReLU(),
        )
        self.value = nn.Linear(512, 1)
        self.advantage = nn.Linear(512, NUM_ACTIONS)

    def forward(self, grid, vec, ghost_dir=None):
        features = self.conv(grid)
        features = torch.cat([features, vec], dim=1)
        if self.use_ghost_dir:
            if ghost_dir is None:
                raise ValueError("ghost_dir is required when use_ghost_dir=True")
            features = torch.cat(
                [features, self.ghost_mlp(ghost_dir)], dim=1
            )
        hidden = self.trunk(features)
        value = self.value(hidden)
        advantage = self.advantage(hidden)
        # Subtracting the mean advantage keeps the decomposition identifiable
        # without changing the greedy action.
        return value + advantage - advantage.mean(dim=1, keepdim=True)
