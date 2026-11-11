"""Shared evaluation: run greedy episodes and aggregate the numbers we
care about (return, score, pellets, survival time, deaths, level clears).
"""

import numpy as np

from .config import N_PELLETS


def run_episode(env, act_fn, seed):
    obs = env.reset(seed=seed)
    total_return = 0.0
    done = False
    info = {}

    while not done:
        action = act_fn(obs)
        obs, reward, done, info = env.step(action)
        total_return += reward

    levels_cleared = info["level"] - 1
    dots_eaten = levels_cleared * N_PELLETS + (N_PELLETS - info["dots_left"])
    deaths = 3 - info["lives"]

    return {
        "return": total_return,
        "score": info["score"],
        "dots": dots_eaten,
        "frames": info["frames"],
        "deaths": deaths,
        "clears": levels_cleared,
        "level": info["level"],
    }


def evaluate(env, act_fn, episodes=10, seed0=10_000):
    rows = [run_episode(env, act_fn, seed0 + i) for i in range(episodes)]

    def stats(key):
        values = np.array([r[key] for r in rows], dtype=np.float64)
        return float(values.mean()), float(values.std())

    result = {}
    for key in ("return", "score", "dots", "frames", "deaths", "clears"):
        mean, std = stats(key)
        result[f"{key}_mean"] = mean
        result[f"{key}_std"] = std
    result["episodes"] = episodes
    return result
