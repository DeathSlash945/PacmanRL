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
from .config import EnvConfig, RewardConfig
from .pacman_env import PacmanEnv


def check_observation(obs):
    grid, vec = obs["grid"], obs["vec"]
    assert grid.shape == (8, 31, 28), grid.shape
    assert grid.dtype == np.float32, grid.dtype
    assert vec.shape == (18,), vec.shape
    assert vec.dtype == np.float32, vec.dtype
    assert np.isfinite(grid).all() and np.isfinite(vec).all()


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
    assert reward >= cfg.reward.level_clear_bonus, reward
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
    benchmark()
    print("ALL STAGE-0 TESTS PASSED")
