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
        use_ghost_dir=False,
        use_amp=True,
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
        self.width = int(width)
        self.lr = float(lr)
        self.use_ghost_dir = bool(use_ghost_dir)

        # Mixed precision is a pure speed optimisation: weights and the
        # optimizer stay in float32, while the convolutions and matmuls run
        # in bf16/fp16.  bf16 needs no gradient scaler and has the same
        # exponent range as fp32, so it is the safer default on Ampere+.
        self.use_amp = bool(use_amp) and self.device.type == "cuda"
        self.amp_dtype = torch.float16
        if self.use_amp:
            self.amp_dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
            torch.backends.cudnn.benchmark = True
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp and self.amp_dtype == torch.float16,
        )

        self.buffer = ReplayBuffer(replay_size, seed=seed)
        self._build_networks()

        self.decisions = 0
        self.updates = 0
        self.loaded_partial = False

    def _build_networks(self):
        """(Re)create the online/target nets for the current architecture."""
        self.online = QNetwork(
            width=self.width, use_ghost_dir=self.use_ghost_dir
        ).to(self.device)
        self.target = QNetwork(
            width=self.width, use_ghost_dir=self.use_ghost_dir
        ).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.lr)

    def act(self, obs, greedy=False):
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.randrange(NUM_ACTIONS)

        grid = torch.from_numpy(obs["grid"]).float().unsqueeze(0).to(self.device)
        vec = torch.from_numpy(obs["vec"]).float().unsqueeze(0).to(self.device)
        ghost_dir = None
        if self.use_ghost_dir:
            ghost_dir = (
                torch.from_numpy(obs["ghost_dir"])
                .float()
                .unsqueeze(0)
                .to(self.device)
            )
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        ):
            q = self.online(grid, vec, ghost_dir)
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
        ghost_dir = None
        next_ghost_dir = None
        if self.use_ghost_dir:
            ghost_dir = batch["ghost_dir"].to(self.device).float()
            next_ghost_dir = batch["next_ghost_dir"].to(self.device).float()
        action = batch["action"].to(self.device)
        reward = batch["reward"].to(self.device)
        next_grid = batch["next_grid"].to(self.device).float() / 2.0
        next_vec = batch["next_vec"].to(self.device)
        terminal = batch["terminal"].to(self.device)

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        ):
            q = self.online(grid, vec, ghost_dir).gather(
                1, action.unsqueeze(1)
            ).squeeze(1)

            with torch.no_grad():
                # Double DQN: online net picks the action, target net scores it.
                next_actions = self.online(
                    next_grid, next_vec, next_ghost_dir
                ).argmax(dim=1, keepdim=True)
                next_q = self.target(
                    next_grid, next_vec, next_ghost_dir
                ).gather(1, next_actions).squeeze(1)
                target = reward + self.gamma * next_q * (1.0 - terminal)

            # The loss itself is computed in float32 so a rare large target
            # cannot overflow the reduced-precision forward pass.
            loss = F.smooth_l1_loss(q.float(), target.float())

        self.optimizer.zero_grad(set_to_none=True)
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
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
                "use_ghost_dir": self.use_ghost_dir,
                "width": self.width,
                "lr": self.lr,
                "architecture": 2,
            },
            path,
        )

    def _load_matching(self, module, state, checkpoint_width):
        """Load every compatible tensor, including a resized trunk input.

        When a checkpoint predates the ghost-direction branch, the only
        shape mismatch is ``trunk.0.weight``: it has VEC_DIM fewer input
        columns.  Copy the old columns and zero the new ones, so the old
        policy is initially preserved exactly.  The ghost branch can then
        learn on top of it.
        """
        own = module.state_dict()
        for key, value in state.items():
            if key not in own:
                continue
            if own[key].shape == value.shape:
                own[key] = value
                continue
            if (
                key.endswith("trunk.0.weight")
                and own[key].dim() == 2
                and value.dim() == 2
                and int(checkpoint_width) == self.width
            ):
                merged = own[key].clone()
                rows = min(merged.shape[0], value.shape[0])
                cols = min(merged.shape[1], value.shape[1])
                merged[:rows, :cols] = value[:rows, :cols]
                if cols < merged.shape[1]:
                    merged[:, cols:] = 0.0
                own[key] = merged
        module.load_state_dict(own)

    def load(self, path, allow_partial=False):
        checkpoint = torch.load(path, map_location=self.device)
        online_state = checkpoint["online"]
        target_state = checkpoint.get("target", online_state)

        checkpoint_use = bool(checkpoint.get("use_ghost_dir", False))
        checkpoint_width = checkpoint.get("width")
        if checkpoint_width is None:
            weight = online_state.get("conv.4.weight")
            checkpoint_width = (
                int(weight.shape[0]) if weight is not None else self.width
            )
        checkpoint_width = int(checkpoint_width)

        same_arch = (
            checkpoint_use == self.use_ghost_dir
            and checkpoint_width == self.width
        )

        if not same_arch and not allow_partial:
            # Evaluating an old checkpoint should just work, even though
            # the caller did not know which observation version it used.
            self.use_ghost_dir = checkpoint_use
            self.width = checkpoint_width
            self._build_networks()
            same_arch = True

        if same_arch:
            self.online.load_state_dict(online_state)
            self.target.load_state_dict(target_state)
            optimizer_state = checkpoint.get("optimizer")
            if optimizer_state is not None:
                try:
                    self.optimizer.load_state_dict(optimizer_state)
                except (ValueError, KeyError):
                    # A fresh optimizer is safer than a mismatched one.
                    pass
            self.loaded_partial = False
        else:
            # Transfer what is architecturally compatible (all convolutions,
            # both dueling heads, and the old part of the trunk input).
            self._load_matching(self.online, online_state, checkpoint_width)
            self._load_matching(self.target, target_state, checkpoint_width)
            self.loaded_partial = True

        self.decisions = checkpoint.get("decisions", 0)
        self.updates = checkpoint.get("updates", 0)
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        return self
