"""Gym-style wrapper around the hand-written Pac-Man game.

API
---
    env = PacmanEnv(EnvConfig(frame_skip=4))
    obs = env.reset()
    obs, reward, done, info = env.step(action)

`obs` is the dict from ObservationBuilder: {"grid": (8,31,28), "vec": (19,)}.
`action` is an index into config.ACTIONS.
`done` is true on game over, level clear (when level_cap is set), or the
episode time limit.  `info["truncated"]` distinguishes the time limit
from a real terminal state.

The wrapper never modifies the game's source files.  It only calls
Game.update(), Pacman.set_dir(), and reads public attributes.
"""

import numpy as np

import map as maze
from env import Game

from .config import ACTIONS, NUM_ACTIONS, EnvConfig
from .obs import ObservationBuilder


class PacmanEnv:
    def __init__(self, config: EnvConfig | None = None):
        self.cfg = config or EnvConfig()

        # fx=False removes READY!, death and ghost-eaten pauses, so every
        # update() advances exactly one 60 Hz game frame.  That is what
        # makes frame_skip deterministic and training fast.
        self.game = Game(fx=False)

        self.obs_builder = ObservationBuilder()
        self.rng = np.random.default_rng(self.cfg.seed)

        self.frames = 0
        self.episode = 0
        self.episode_seed = 0
        self.episode_return = 0.0

        self.reset()

    def reset(self, seed: int | None = None):
        """Start a brand-new game at level 1 and return the first observation.

        If no seed is given, draw one from this environment's RNG.  The
        seed is passed to Game.reset(), which calls random.seed(seed),
        making frightened-ghost turns and fruit timing reproducible.
        """
        if seed is None:
            seed = int(self.rng.integers(0, 2**31 - 1))

        self.episode_seed = seed
        self.game.reset(seed=seed)
        self.frames = 0
        self.episode += 1
        self.episode_return = 0.0
        return self._obs()

    def step(self, action: int):
        """Apply one action for frame_skip game frames.

        The action is only a *requested* direction.  Pac-Man buffers it
        and applies it at the next tile centre, exactly like a human
        pressing an arrow key.
        """
        if not isinstance(action, (int, np.integer)) or not 0 <= int(action) < NUM_ACTIONS:
            raise ValueError(f"action must be an int in 0..{NUM_ACTIONS - 1}, got {action!r}")

        self.game.pacman.set_dir(ACTIONS[int(action)])

        reward = float(self.cfg.reward.step_penalty)
        died = False
        level_cleared = False
        truncated = False
        done = False

        for _ in range(self.cfg.frame_skip):
            lives_before = self.game.pacman.lives
            score_before = self.game.pacman.score

            self.game.update()
            self.frames += 1

            # The only reliable score signal: the game never subtracts
            # score on death, so pellets/ghosts/fruit are all positive
            # deltas and the death penalty is added separately.
            reward += (self.game.pacman.score - score_before) * self.cfg.reward.score_scale

            if self.game.pacman.lives < lives_before:
                died = True
                reward += self.cfg.reward.death_penalty

                # With fx=False, reset_round() already respawned Pac-Man
                # and forced dir/next_dir back to 'left'.  Continuing the
                # remaining skip frames would move him with a stale
                # action, so stop and let the agent choose again.
                if self.game.game_over:
                    done = True
                break

            if self.game.game_over:
                done = True
                break

            # Clearing a level calls next_level() inside update(), which
            # increments game.level and resets the round on the new level.
            if self.cfg.level_cap is not None and self.game.level > self.cfg.level_cap:
                level_cleared = True
                reward += self.cfg.reward.level_clear_bonus
                done = True
                break

            if self.frames >= self.cfg.max_frames:
                truncated = True
                done = True
                break

        self.episode_return += reward
        info = {
            "episode_seed": self.episode_seed,
            "frames": self.frames,
            "level": self.game.level,
            "score": self.game.pacman.score,
            "lives": self.game.pacman.lives,
            "dots_left": len(maze.pellets),
            "died": died,
            "level_cleared": level_cleared,
            "truncated": truncated,
            "episode_return": self.episode_return,
        }
        return self._obs(), reward, done, info

    def _obs(self):
        return self.obs_builder.build(self.game)

    def close(self):
        """Nothing to release: the game owns no OS resources."""
        return None
