"""Deep Q-Network agent: replay buffer, target network, epsilon-greedy.

The algorithm is the classic DQN with two standard stabilisers:
  * a target network that is copied from the online network every
    `target_update` gradient steps, and
  * Double DQN action selection: the online net chooses the next action,
    the target net evaluates it.  This reduces over-estimation.
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..buffer import ReplayBuffer
from ..config import NUM_ACTIONS
from ..networks import QNetwork


class DQNAgent:
    def __init__(
        self,
        device=None,
        lr=1e-4,
        gamma=0.99,
        batch_size=64,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=200_000,
        target_update=2_000,
        replay_size=100_000,
        grad_clip=10.0,
        width=64,
        seed=0,
    ):
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.target_update = target_update
        self.grad_clip = grad_clip
        self.rng = random.Random(seed)

        self.online = QNetwork(width=width).to(self.device)
        self.target = QNetwork(width=width).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = ReplayBuffer(replay_size, seed=seed)

        self.decisions = 0
        self.updates = 0

    def act(self, obs, greedy=False):
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.randrange(NUM_ACTIONS)

        grid = torch.from_numpy(obs["grid"]).float().unsqueeze(0).to(self.device)
        vec = torch.from_numpy(obs["vec"]).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.online(grid, vec)
        return int(q.argmax(dim=1).item())

    def update_epsilon(self):
        self.decisions += 1
        frac = min(1.0, self.decisions / self.epsilon_decay)
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def remember(self, obs, action, reward, next_obs, terminal):
        self.buffer.add(obs, action, reward, next_obs, terminal)

    def learn(self):
        if self.buffer.size < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)
        grid = batch["grid"].to(self.device).float() / 2.0
        vec = batch["vec"].to(self.device)
        action = batch["action"].to(self.device)
        reward = batch["reward"].to(self.device)
        next_grid = batch["next_grid"].to(self.device).float() / 2.0
        next_vec = batch["next_vec"].to(self.device)
        terminal = batch["terminal"].to(self.device)

        q = self.online(grid, vec).gather(1, action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: online net picks the action, target net scores it.
            next_actions = self.online(next_grid, next_vec).argmax(dim=1, keepdim=True)
            next_q = self.target(next_grid, next_vec).gather(1, next_actions).squeeze(1)
            target = reward + self.gamma * next_q * (1.0 - terminal)

        loss = F.smooth_l1_loss(q, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optimizer.step()

        self.updates += 1
        if self.updates % self.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.item())

    def save(self, path):
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "decisions": self.decisions,
                "updates": self.updates,
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.online.load_state_dict(checkpoint["online"])
        self.target.load_state_dict(checkpoint["target"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.decisions = checkpoint.get("decisions", 0)
        self.updates = checkpoint.get("updates", 0)
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        return self
