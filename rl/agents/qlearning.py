"""Tabular Q-learning with a hand-designed discrete state.

A truly tabular Q-table over the raw game is impossible: the maze has
244 pellets, so there are 2**244 pellet configurations.  This baseline
therefore compresses the game into a small tuple of features:

    (pacman_col, pacman_row,      # where Pac-Man is (868 possible tiles)
     pellet_bits,                 # which 4 neighbours still hold a pellet
     ghost_sector,                # 8-way direction to nearest dangerous ghost
     ghost_distance_bucket,       # 0..3 tiles, or 4 = no dangerous ghost
     fright_bucket,               # 0 none, 1 <=2s left, 2 >2s left
     pacman_direction)            # 0..3

Even this has millions of possible states, so we use a Python dict and
only allocate rows for states the agent actually visits.  That is still
"tabular" Q-learning: one independent value per discrete state.
"""

import math
import pickle
import random

import numpy as np

import map as maze

from ..config import ACTIONS, NUM_ACTIONS
from ..obs import wrap_tile

DIR_INDEX = {d: i for i, d in enumerate(ACTIONS)}

# Neighbour order is arbitrary but fixed: up, left, right, down.
PELLET_NEIGHBOURS = ((0, -1), (-1, 0), (1, 0), (0, 1))


def _wrapped_delta(a, b, width=28):
    """Shortest signed horizontal distance on a wrapping tunnel."""
    d = b - a
    if d > width // 2:
        d -= width
    elif d < -(width // 2):
        d += width
    return d


def _ghost_features(game):
    """Sector (0..8), distance bucket (0..4) for the nearest killer ghost."""
    pac_c, pac_r = wrap_tile(game.pacman.tile)
    best = None
    for ghost in game.ghosts:
        if ghost.mode not in ("scatter", "chase"):
            continue
        gc, gr = wrap_tile(ghost.tile)
        dx = _wrapped_delta(pac_c, gc)
        dy = gr - pac_r
        dist = abs(dx) + abs(dy)
        if best is None or dist < best[0]:
            best = (dist, dx, dy)

    if best is None:
        return 8, 4  # no dangerous ghost on the board

    dist, dx, dy = best
    sector = int((math.atan2(dy, dx) + math.pi) / (math.pi / 4)) % 8
    if dist <= 2:
        bucket = 0
    elif dist <= 5:
        bucket = 1
    elif dist <= 9:
        bucket = 2
    else:
        bucket = 3
    return sector, bucket


class QLearningAgent:
    def __init__(
        self,
        alpha=0.10,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=200_000,
        seed=0,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.rng = random.Random(seed)
        self.q = {}
        self.decisions = 0

    def state_from_game(self, game):
        pac = game.pacman
        c, r = wrap_tile(pac.tile)

        bits = 0
        for i, (dc, dr) in enumerate(PELLET_NEIGHBOURS):
            if (c + dc, r + dr) in maze.pellets:
                bits |= 1 << i

        sector, distance_bucket = _ghost_features(game)

        if game.fright == 0:
            fright_bucket = 0
        elif game.fright < 120:
            fright_bucket = 1
        else:
            fright_bucket = 2

        return (c, r, bits, sector, distance_bucket, fright_bucket,
                DIR_INDEX.get(pac.dir, 0))

    def q_values(self, state):
        row = self.q.get(state)
        if row is None:
            row = np.zeros(NUM_ACTIONS, dtype=np.float32)
            self.q[state] = row
        return row

    def act(self, state, greedy=False):
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.randrange(NUM_ACTIONS)

        row = self.q.get(state)
        if row is None:
            # Unseen state: any action is equally uninformed.  During
            # evaluation stay deterministic; during training explore.
            return 0 if greedy else self.rng.randrange(NUM_ACTIONS)

        if greedy:
            return int(np.argmax(row))

        best = np.flatnonzero(row == row.max())
        return int(self.rng.choice(best))

    def learn(self, state, action, reward, next_state, terminal):
        row = self.q_values(state)
        target = reward
        if not terminal:
            next_row = self.q.get(next_state)
            if next_row is not None:
                target += self.gamma * float(next_row.max())
        row[action] += self.alpha * (target - row[action])

        self.decisions += 1
        frac = min(1.0, self.decisions / self.epsilon_decay)
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(
                {"q": self.q, "decisions": self.decisions, "epsilon": self.epsilon},
                f,
                protocol=4,
            )

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = cls()
        agent.q = data["q"]
        agent.decisions = data.get("decisions", 0)
        agent.epsilon = data.get("epsilon", agent.epsilon_end)
        return agent
