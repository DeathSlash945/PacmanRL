"""Stage-0 verification: shapes, determinism, death/level/time-limit logic.

Run from the project root:
    SDL_VIDEODRIVER=dummy python -m rl.smoke_test
"""

import os
import random
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np

import map as maze
from .config import GHOST_DIR_DIM, EnvConfig, RewardConfig
from .pacman_env import PacmanEnv


def check_observation(obs):
    grid, vec, ghost_dir = obs["grid"], obs["vec"], obs["ghost_dir"]
    assert grid.shape == (8, 31, 28), grid.shape
    assert grid.dtype == np.float32, grid.dtype
    assert vec.shape == (18,), vec.shape
    assert vec.dtype == np.float32, vec.dtype
    assert ghost_dir.shape == (GHOST_DIR_DIM,), ghost_dir.shape
    assert ghost_dir.dtype == np.float32, ghost_dir.dtype
    assert np.isfinite(grid).all() and np.isfinite(vec).all()
    assert np.isfinite(ghost_dir).all()


def test_shapes_and_reset():
    env = PacmanEnv()
    obs = env.reset(seed=0)
    check_observation(obs)
    assert env.game.level == 1
    assert env.game.pacman.lives == 3
    assert len(maze.pellets) == 244
    print("ok  shapes/reset")


def test_determinism():
    env = PacmanEnv()
    actions = [random.Random(0).randrange(4) for _ in range(300)]

    def rollout():
        obs = env.reset(seed=123)
        trace = []
        for a in actions:
            obs, r, done, info = env.step(a)
            trace.append((round(env.game.pacman.x, 4),
                          round(env.game.pacman.y, 4),
                          env.game.pacman.score))
            if done:
                break
        return trace

    first, second = rollout(), rollout()
    assert first == second, "same seed + same actions must replay identically"
    print(f"ok  determinism ({len(first)} decisions replayed)")


def test_death_detection():
    env = PacmanEnv()
    env.reset(seed=7)
    rng = random.Random(7)
    deaths = 0
    for i in range(4000):
        obs, r, done, info = env.step(rng.randrange(4))
        if info["died"]:
            deaths += 1
        if done:
            break
    assert deaths >= 1, "random play should die at least once in 4000 decisions"
    print(f"ok  death detection ({deaths} death(s), frames={info['frames']})")


def test_level_clear():
    cfg = EnvConfig(frame_skip=4, max_frames=18000, level_cap=1)
    env = PacmanEnv(cfg)
    env.reset(seed=1)
    maze.pellets.clear()          # simulate having eaten the last pellet
    obs, reward, done, info = env.step(0)
    assert done and info["level_cleared"], info
    assert env.game.level == 2, env.game.level
    assert reward >= (
        cfg.reward.level_clear_bonus
        + cfg.reward.step_penalty
        - 1e-9
    ), reward
    assert len(maze.pellets) == 244, "next_level must refill the maze"
    print(f"ok  level clear (reward={reward:.2f})")


def test_time_limit():
    cfg = EnvConfig(frame_skip=4, max_frames=10, level_cap=None,
                    reward=RewardConfig())
    env = PacmanEnv(cfg)
    env.reset(seed=2)
    done = False
    for _ in range(10):
        obs, reward, done, info = env.step(0)
        if done:
            break
    assert done and info["truncated"], info
    assert info["frames"] == 10, info["frames"]  # stops exactly at max_frames
    print(f"ok  time limit (frames={info['frames']})")


def test_reward_signs():
    env = PacmanEnv()
    env.reset(seed=3)
    rng = random.Random(3)
    positive = False
    for _ in range(300):
        obs, reward, done, info = env.step(rng.randrange(4))
        if reward > 0:
            positive = True
            break
        if done:
            break
    assert positive, "eating at least one pellet should give positive reward"
    print("ok  reward signs")


def test_stay_penalty():
    env = PacmanEnv()
    env.reset(seed=4)
    env.cfg.reward.danger_penalty = 0.0  # isolate the stay-penalty term
    env.cfg.reward.step_penalty = 0.0

    # (14,23) is open left/right but walled above and below.  Put Pac-Man
    # exactly on the tile centre facing up so the requested direction is
    # blocked and no frame moves him.
    pac = env.game.pacman
    pac.x, pac.y = 14, 23
    pac.dir = pac.next_dir = "up"
    maze.pellets.pop((14, 23), None)

    obs, reward, done, info = env.step(0)  # action 0 = up
    assert info["stayed"], info
    assert abs(reward - env.cfg.reward.stay_penalty) < 1e-9, reward
    print(f"ok  stay penalty ({reward:.3f})")


def test_dot_not_eaten_penalty():
    reward_cfg = RewardConfig(
        step_penalty=0.0,
        stay_penalty=0.0,
        dot_not_eaten_penalty=-0.02,
        dot_not_eaten_grace=0,
        danger_penalty=0.0,
    )
    cfg = EnvConfig(frame_skip=4, max_frames=18000, level_cap=1,
                    reward=reward_cfg)
    env = PacmanEnv(cfg)

    def park_without_a_dot(seed=4):
        env.reset(seed=seed)
        pac = env.game.pacman
        pac.x, pac.y = 14, 23
        pac.dir = pac.next_dir = "up"  # walled above, no pellet here
        maze.pellets.pop((14, 23), None)

    # Zero grace: the first decision without a dot is penalised.
    park_without_a_dot()
    obs, reward, done, info = env.step(0)
    assert not info["dot_eaten"], info
    assert info["decisions_since_dot"] == 1, info
    assert info["dot_not_eaten_penalty"], info
    assert abs(reward - (-0.02)) < 1e-9, reward

    # Grace gives the agent time before the penalty starts.
    env.cfg.reward.dot_not_eaten_grace = 5
    park_without_a_dot()
    obs, reward, done, info = env.step(0)
    assert not info["dot_not_eaten_penalty"], info
    assert abs(reward) < 1e-9, reward

    # Terminal/truncated frames are exempt.
    env.cfg.reward.dot_not_eaten_grace = 0
    env.cfg.max_frames = 4
    park_without_a_dot()
    obs, reward, done, info = env.step(0)
    assert done and info["truncated"], info
    assert not info["dot_not_eaten_penalty"], info
    assert abs(reward) < 1e-9, reward
    print("ok  dot-not-eaten penalty")


def test_ghost_direction_observation():
    env = PacmanEnv()
    obs = env.reset(seed=11)
    check_observation(obs)
    directions = obs["ghost_dir"].reshape(4, 4)
    assert np.all(directions.sum(axis=1) <= 1.0), directions
    assert directions.sum() >= 1.0, "at least one ghost must have a direction"
    print("ok  ghost-direction observation")


def test_dot_curriculum():
    cfg = EnvConfig(
        frame_skip=4,
        max_frames=18000,
        level_cap=1,
        dots_removed_min=80,
        dots_removed_max=80,
    )
    env = PacmanEnv(cfg)
    obs = env.reset(seed=5)
    check_observation(obs)
    assert env.dots_removed == 80, env.dots_removed
    assert len(maze.pellets) == 164, len(maze.pellets)
    assert obs["grid"][1].sum() == 160, obs["grid"][1].sum()  # ordinary dots
    assert obs["grid"][2].sum() == 4, obs["grid"][2].sum()    # power pellets
    initial_dots = obs["grid"][1].copy()
    obs, reward, done, info = env.step(0)
    assert info["dots_removed"] == 80, info
    assert info["dots_eaten"] >= 0, info

    # Same seed must remove the same dots.
    env2 = PacmanEnv(cfg)
    obs2 = env2.reset(seed=5)
    assert np.array_equal(initial_dots, obs2["grid"][1])
    print("ok  endgame dot curriculum")


def test_danger_shaping():
    reward_cfg = RewardConfig(
        step_penalty=0.0,
        stay_penalty=0.0,
        dot_not_eaten_penalty=0.0,
        danger_penalty=1.0,
        danger_radius=5,
    )
    env = PacmanEnv(EnvConfig(
        frame_skip=4, max_frames=18000, level_cap=1, reward=reward_cfg
    ))
    env.reset(seed=12)
    pac = env.game.pacman
    pac.x, pac.y = 14, 23
    ghost = env.game.ghosts[0]
    ghost.mode = "chase"
    ghost.x, ghost.y = 16, 23
    danger = env._danger_level()
    assert 0.0 < danger <= 1.0, danger
    print(f"ok  danger shaping (level={danger:.2f})")


def benchmark(decisions=20000):
    env = PacmanEnv()
    env.reset(seed=0)
    rng = random.Random(0)
    episodes = deaths = cleared = 0
    returns = []
    t0 = time.perf_counter()
    for _ in range(decisions):
        obs, reward, done, info = env.step(rng.randrange(4))
        if done:
            returns.append(info["episode_return"])
            episodes += 1
            deaths += info["died"]
            cleared += info["level_cleared"]
            env.reset()
    dt = time.perf_counter() - t0
    print(
        f"benchmark: {decisions} decisions in {dt:.2f}s "
        f"= {decisions / dt:,.0f} decisions/s "
        f"= {decisions * env.cfg.frame_skip / dt:,.0f} game frames/s"
    )
    print(
        f"           episodes={episodes} deaths={deaths} clears={cleared} "
        f"mean_return={np.mean(returns) if returns else float('nan'):.3f}"
    )


if __name__ == "__main__":
    test_shapes_and_reset()
    test_determinism()
    test_death_detection()
    test_level_clear()
    test_time_limit()
    test_reward_signs()
    test_stay_penalty()
    test_dot_not_eaten_penalty()
    test_ghost_direction_observation()
    test_dot_curriculum()
    test_danger_shaping()
    benchmark()
    print("ALL STAGE-0 TESTS PASSED")
