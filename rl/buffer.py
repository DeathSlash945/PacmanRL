"""Fixed-size replay buffer for DQN.

The observation grid contains only these values: 0, 0.5, -1 and 1.
Multiplying by 2 and storing int8 keeps each frame at 8*31*28 = 6944
bytes instead of 27 KB as float32.  A 100k-transition buffer therefore
uses about 1.4 GB of RAM instead of 5.6 GB.
"""

import numpy as np
import torch

from .config import GRID_H, GRID_W, N_CHANNELS, VEC_DIM


def encode_grid(grid):
    """float observation (0, .5, -1, 1) -> int8 (-0, 1, -2, 2)."""
    return np.rint(grid * 2.0).astype(np.int8)


class ReplayBuffer:
    def __init__(self, capacity=100_000, seed=0):
        self.capacity = capacity
        self.grid = np.zeros((capacity, N_CHANNELS, GRID_H, GRID_W), dtype=np.int8)
        self.next_grid = np.zeros_like(self.grid)
        self.vec = np.zeros((capacity, VEC_DIM), dtype=np.float32)
        self.next_vec = np.zeros_like(self.vec)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.terminal = np.zeros(capacity, dtype=np.uint8)

        self.pos = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, next_obs, terminal):
        i = self.pos
        self.grid[i] = encode_grid(obs["grid"])
        self.vec[i] = obs["vec"]
        self.action[i] = action
        self.reward[i] = reward
        self.next_grid[i] = encode_grid(next_obs["grid"])
        self.next_vec[i] = next_obs["vec"]
        self.terminal[i] = float(terminal)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = self.rng.integers(0, self.size, size=batch_size)

        # Keep int8 on the CPU; the agent converts to float32 on the GPU.
        # This moves 4x fewer bytes over PCIe and avoids a CPU float cast.
        return {
            "grid": torch.from_numpy(self.grid[idx]),
            "vec": torch.from_numpy(self.vec[idx]),
            "action": torch.from_numpy(self.action[idx]),
            "reward": torch.from_numpy(self.reward[idx]),
            "next_grid": torch.from_numpy(self.next_grid[idx]),
            "next_vec": torch.from_numpy(self.next_vec[idx]),
            "terminal": torch.from_numpy(self.terminal[idx]).float(),
        }
