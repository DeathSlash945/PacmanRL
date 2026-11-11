"""Train the DQN agent.

Example:
    SDL_VIDEODRIVER=dummy /home/deathslash/venvs/mlasgn/bin/python -m rl.train_dqn \
        --total-decisions 1000000 --out runs/dqn

The first `warmup` decisions are random and only fill the replay buffer.
After that the agent acts epsilon-greedily and takes one gradient step
per decision (configurable with --train-every).
"""

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np

from .agents.dqn import DQNAgent
from .config import EnvConfig, N_PELLETS
from .evaluate import evaluate
from .pacman_env import PacmanEnv

METRIC_FIELDS = [
    "decision", "episode", "epsilon", "updates", "loss_mean",
    "return_mean", "return_std", "score_mean", "score_std",
    "dots_mean", "dots_std", "frames_mean", "frames_std",
    "deaths_mean", "deaths_std", "clears_mean", "clears_std",
    "wall_seconds",
]
EPISODE_FIELDS = [
    "episode", "decision", "return", "score", "dots", "frames",
    "deaths", "clears", "epsilon", "loss_mean",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--total-decisions", type=int, default=1_000_000)
    p.add_argument("--warmup", type=int, default=5_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=int, default=200_000)
    p.add_argument("--target-update", type=int, default=2_000)
    p.add_argument("--replay-size", type=int, default=100_000)
    p.add_argument("--train-every", type=int, default=2)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--max-frames", type=int, default=18_000)
    p.add_argument("--level-cap", type=int, default=1,
                   help="stop episodes after this level; <=0 means no cap")
    p.add_argument("--eval-every", type=int, default=50_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default="runs/dqn")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    level_cap = args.level_cap if args.level_cap > 0 else None
    cfg = EnvConfig(
        frame_skip=args.frame_skip,
        max_frames=args.max_frames,
        level_cap=level_cap,
        seed=args.seed,
    )
    env = PacmanEnv(cfg)
    agent = DQNAgent(
        device=args.device,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        target_update=args.target_update,
        replay_size=args.replay_size,
        width=args.width,
        seed=args.seed,
    )
    print(f"device: {agent.device}")
    rng = random.Random(args.seed)

    metrics_file = open(out / "metrics.csv", "w", newline="")
    episodes_file = open(out / "episodes.csv", "w", newline="")
    metrics_writer = csv.DictWriter(metrics_file, fieldnames=METRIC_FIELDS)
    episodes_writer = csv.DictWriter(episodes_file, fieldnames=EPISODE_FIELDS)
    metrics_writer.writeheader()
    episodes_writer.writeheader()

    obs = env.reset()
    episode = 0
    episode_return = 0.0
    episode_losses = []
    next_eval = args.eval_every
    started = time.perf_counter()

    def greedy_action(o):
        return agent.act(o, greedy=True)

    for decision in range(1, args.total_decisions + 1):
        if decision <= args.warmup:
            action = rng.randrange(4)
        else:
            action = agent.act(obs)

        next_obs, reward, done, info = env.step(action)
        terminal = done and not info["truncated"]
        agent.remember(obs, action, reward, next_obs, terminal)
        agent.update_epsilon()

        if decision > args.warmup and decision % args.train_every == 0:
            loss = agent.learn()
            if loss is not None:
                episode_losses.append(loss)

        obs = next_obs
        episode_return += reward

        if done:
            episode += 1
            levels_cleared = info["level"] - 1
            dots = levels_cleared * N_PELLETS + (N_PELLETS - info["dots_left"])
            loss_mean = float(np.mean(episode_losses)) if episode_losses else 0.0
            episodes_writer.writerow({
                "episode": episode,
                "decision": decision,
                "return": round(episode_return, 4),
                "score": info["score"],
                "dots": dots,
                "frames": info["frames"],
                "deaths": 3 - info["lives"],
                "clears": levels_cleared,
                "epsilon": round(agent.epsilon, 4),
                "loss_mean": round(loss_mean, 5),
            })
            episodes_file.flush()

            if decision >= next_eval:
                result = evaluate(env, greedy_action, episodes=args.eval_episodes,
                                  seed0=10_000)
                elapsed = time.perf_counter() - started
                row = {
                    "decision": decision,
                    "episode": episode,
                    "epsilon": round(agent.epsilon, 4),
                    "updates": agent.updates,
                    "loss_mean": round(loss_mean, 5),
                    "wall_seconds": round(elapsed, 2),
                }
                row.update({k: round(v, 4) for k, v in result.items()
                            if isinstance(v, float)})
                metrics_writer.writerow(row)
                metrics_file.flush()
                agent.save(out / "dqn.pt")
                print(
                    f"decision {decision:>8} | eps {agent.epsilon:.3f} | "
                    f"upd {agent.updates:>7} | loss {loss_mean:.4f} | "
                    f"return {result['return_mean']:7.2f} | "
                    f"score {result['score_mean']:7.1f} | "
                    f"dots {result['dots_mean']:6.1f} | "
                    f"frames {result['frames_mean']:7.1f} | "
                    f"deaths {result['deaths_mean']:.2f} | "
                    f"clears {result['clears_mean']:.2f} | {elapsed:.0f}s"
                )
                next_eval += args.eval_every

            obs = env.reset()
            episode_return = 0.0
            episode_losses = []

    agent.save(out / "dqn.pt")
    metrics_file.close()
    episodes_file.close()
    print(f"done: {args.total_decisions} decisions, {agent.updates} updates, "
          f"{time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
