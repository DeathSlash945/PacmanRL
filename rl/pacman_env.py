"""Gym-style wrapper around the hand-written Pac-Man game.

API
---
    env = PacmanEnv(EnvConfig(frame_skip=4))
    obs = env.reset()
    obs, reward, done, info = env.step(action)

`obs` is the dict from ObservationBuilder:
{"grid": (8,31,28), "vec": (18,), "ghost_dir": (16,)}.
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
        self.dots_removed = 0
        self.dots_eaten = 0
        self.dots_at_start = 0

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
        self.decisions_since_dot = 0
        self.dots_eaten = 0
        self.dots_removed = 0

        # Endgame curriculum: remove ordinary dots before the first
        # observation.  Power pellets are kept because they are part of the
        # escape strategy we want the agent to learn.
        self._apply_dot_curriculum(seed)
        # Game.__init__ sets total once; reset() does not update it.  Fruit
        # timing uses total - remaining, so it must match the actual level.
        self.game.total = len(maze.pellets)
        self.dots_at_start = len(maze.pellets)
        return self._obs()

    def _apply_dot_curriculum(self, seed: int) -> None:
        cfg = self.cfg
        low = max(0, int(cfg.dots_removed_min))
        high = max(0, int(cfg.dots_removed_max))
        if high < low:
            low, high = high, low
        if high <= 0:
            return

        dot_tiles = [pos for pos, kind in maze.pellets.items() if kind == "."]
        if not dot_tiles:
            return

        high = min(high, len(dot_tiles))
        low = min(low, high)
        # A private RNG makes the curriculum deterministic for a given
        # episode seed even though the game itself uses random.seed().
        rng = np.random.default_rng((int(seed) + 0x9E3779B9) & 0xFFFFFFFF)
        # Keep a slice of episodes on the full board.  A biased curriculum
        # otherwise stops showing the opening, and the opening is half of a
        # full-game evaluation.
        full_fraction = float(
            getattr(cfg, "dots_removed_full_fraction", 0.0)
        )
        if full_fraction > 0.0 and float(rng.random()) < full_fraction:
            return
        # A bias > 0 pushes the sampled count towards the high end.  The
        # uniform default is kept for backward compatibility, but endgame
        # training wants most episodes to start with only a few pellets
        # left, which the bias provides.
        u = float(rng.random())
        bias = max(0.0, float(getattr(cfg, "dots_removed_bias", 0.0)))
        if bias > 0.0:
            u = u ** (1.0 / (1.0 + bias))
        count = low + int(round(u * (high - low)))
        count = max(low, min(high, count))
        if count <= 0:
            return

        chosen = rng.choice(len(dot_tiles), size=count, replace=False)
        for index in chosen:
            maze.pellets.pop(dot_tiles[int(index)], None)
        self.dots_removed = count

    def _death_reward(self) -> float:
        """Death penalty, optionally scaled by progress on the current board.

        Dying with 240/244 pellets eaten should hurt much more than dying at
        the start of the episode; otherwise there is no gradient towards
        protecting a nearly-finished board.
        """
        penalty = self.cfg.reward.death_penalty
        weight = self.cfg.reward.death_penalty_progress
        if weight:
            progress = self.dots_eaten / max(1, self.dots_at_start)
            penalty *= 1.0 + weight * min(1.0, progress)
        return penalty

    def _danger_level(self) -> float:
        """Largest danger from a scatter/chase ghost within danger_radius.

        1.0 means a dangerous ghost is on an adjacent tile; the value falls
        linearly to 0 at the edge of the radius.  Wall geometry is ignored
        (Manhattan distance), so this is deliberately a cheap shaping hint,
        not a replacement for the death penalty.
        """
        radius = int(self.cfg.reward.danger_radius)
        if radius <= 0 or self.cfg.reward.danger_penalty == 0.0:
            return 0.0

        pc, pr = self.game.pacman.tile
        danger = 0.0
        for ghost in self.game.ghosts:
            if ghost.mode not in ("scatter", "chase"):
                continue
            gc, gr = ghost.tile
            dx = abs(gc - pc)
            dx = min(dx, 28 - dx)  # tunnel wrap
            distance = dx + abs(gr - pr)
            if distance <= radius:
                closeness = (radius - distance + 1) / radius
                danger = max(danger, closeness)
        return min(1.0, danger)

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
        dot_eaten = False
        ghost_score_total = 0
        levels_cleared_any = 0

        # Track actual displacement, not just Pac-Man's `moving` flag.  The
        # flag is only refreshed at tile centres, and a short pellet pause can
        # leave it stale.  Position is the ground truth for "did not move".
        moved = False
        last_pos = (self.game.pacman.x, self.game.pacman.y)

        for _ in range(self.cfg.frame_skip):
            lives_before = self.game.pacman.lives
            score_before = self.game.pacman.score
            combo_before = self.game.combo
            pellets_before = len(maze.pellets)
            level_before = self.game.level

            self.game.update()
            self.frames += 1

            if (self.game.pacman.x, self.game.pacman.y) != last_pos:
                moved = True
                last_pos = (self.game.pacman.x, self.game.pacman.y)

            # A shrinking pellet set means a dot or power pellet was eaten.
            # A level change also means the last pellet was eaten, even though
            # next_level() immediately refills the maze for the new level.
            if self.game.level > level_before:
                eaten_now = max(0, pellets_before)
                # next_level() refills the maze but Game.next_level() does
                # not update the fruit-timing total.
                self.game.total = len(maze.pellets)
                # Reward *every* level clear, not only the clear that ends a
                # capped episode.  With an uncapped full-game episode the
                # agent previously never received the clear bonus at all,
                # which removed the main signal for finishing a board.
                reward += self.cfg.reward.level_clear_bonus
                lives_bonus = self.cfg.reward.level_clear_lives_bonus
                if lives_bonus:
                    reward += lives_bonus * max(0, self.game.pacman.lives - 1)
                levels_cleared_any += 1
            else:
                eaten_now = max(0, pellets_before - len(maze.pellets))
            if eaten_now:
                self.dots_eaten += eaten_now
                dot_eaten = True

            # The only reliable score signal: the game never subtracts
            # score on death, so pellets/ghosts/fruit are all positive
            # deltas and the death penalty is added separately.
            score_delta = self.game.pacman.score - score_before
            ghost_points = 0
            if self.game.combo > combo_before:
                # collide() awards 200 * 2**combo_before for the next ghost.
                ghost_points = 200 * (2 ** combo_before)
                ghost_points = min(ghost_points, max(0, score_delta))
                ghost_score_total += ghost_points
            base_points = score_delta - ghost_points
            reward += base_points * self.cfg.reward.score_scale
            reward += (
                ghost_points
                * self.cfg.reward.score_scale
                * self.cfg.reward.ghost_score_scale
            )

            if self.game.pacman.lives < lives_before:
                died = True
                reward += self._death_reward()

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
                done = True
                break

            if self.frames >= self.cfg.max_frames:
                truncated = True
                done = True
                break

        # Charge the anti-camping penalty once per decision, not per game
        # frame.  A death, level clear, or timeout is already a terminal
        # event and must not also be labelled as "staying still".
        stayed = (
            not moved
            and not died
            and not level_cleared
            and not levels_cleared_any
            and not truncated
        )
        if stayed:
            reward += self.cfg.reward.stay_penalty

        # Progress pressure: once Pac-Man has gone too long without eating a
        # dot, charge a small penalty for every further decision.  This is
        # separate from the anti-camping penalty: Pac-Man can be moving a lot
        # (for example, circling an empty corridor) and still make no progress.
        if dot_eaten or died or level_cleared:
            self.decisions_since_dot = 0
        else:
            self.decisions_since_dot += 1

        grace = max(0, int(self.cfg.reward.dot_not_eaten_grace))
        dot_not_eaten = (
            not done
            and not dot_eaten
            and self.decisions_since_dot >= grace
        )
        if dot_not_eaten:
            reward += self.cfg.reward.dot_not_eaten_penalty

        # Endless loitering is a failure, not a way to farm step penalties.
        # This is a real terminal state (truncated stays False), so the
        # n-step accumulator does not bootstrap past it.
        no_progress = False
        limit = int(self.cfg.reward.no_progress_limit)
        if limit > 0 and not done and self.decisions_since_dot >= limit:
            no_progress = True
            reward += self.cfg.reward.no_progress_penalty
            done = True

        # Shaping: being close to a dangerous ghost is bad before it becomes
        # a death.  Skip terminal decisions, where reset_round() has already
        # moved the ghosts back to the house and the death penalty is enough.
        danger = 0.0
        danger_penalty = 0.0
        if not done and not died and not level_cleared:
            danger = self._danger_level()
            if danger > 0.0:
                danger_penalty = danger * self.cfg.reward.danger_penalty
                reward -= danger_penalty

        self.episode_return += reward
        info = {
            "episode_seed": self.episode_seed,
            "frames": self.frames,
            "level": self.game.level,
            "score": self.game.pacman.score,
            "lives": self.game.pacman.lives,
            "dots_left": len(maze.pellets),
            "dots_eaten": self.dots_eaten,
            "dots_removed": self.dots_removed,
            "dots_at_start": self.dots_at_start,
            "died": died,
            "level_cleared": level_cleared,
            "levels_cleared": levels_cleared_any,
            "truncated": truncated,
            "stayed": stayed,
            "dot_eaten": dot_eaten,
            "decisions_since_dot": self.decisions_since_dot,
            "dot_not_eaten_penalty": dot_not_eaten,
            "no_progress": no_progress,
            "ghost_score": ghost_score_total,
            "danger": danger,
            "danger_penalty": danger_penalty,
            "episode_return": self.episode_return,
        }
        return self._obs(), reward, done, info

    def _obs(self):
        return self.obs_builder.build(self.game)

    def close(self):
        """Nothing to release: the game owns no OS resources."""
        return None
