"""Train the tabular Q-learning baseline.

Example:
    SDL_VIDEODRIVER=dummy /home/deathslash/venvs/mlasgn/bin/python -m rl.train_q \
        --total-decisions 1000000 --out runs/qlearning

Writes runs/qlearning/q.pkl, metrics.csv (evaluation every N decisions)
and episodes.csv (one row per training episode).
"""

import argparse
import csv
import time
from pathlib import Path

from .agents.qlearning import QLearningAgent
from .config import EnvConfig, N_PELLETS
from .evaluate import evaluate
from .pacman_env import PacmanEnv

METRIC_FIELDS = [
    "decision", "episode", "epsilon", "return_mean", "return_std",
    "score_mean", "score_std", "dots_mean", "dots_std",
    "frames_mean", "frames_std", "deaths_mean", "deaths_std",
    "clears_mean", "clears_std", "wall_seconds",
]
EPISODE_FIELDS = [
    "episode", "decision", "return", "score", "dots", "frames",
    "deaths", "clears", "epsilon",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--total-decisions", type=int, default=1_000_000)
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--max-frames", type=int, default=18_000)
    p.add_argument("--level-cap", type=int, default=1,
                   help="stop episodes after this level; <=0 means no cap")
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=int, default=200_000)
    p.add_argument("--eval-every", type=int, default=50_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="runs/qlearning")
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
    agent = QLearningAgent(
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
    )

    metrics_file = open(out / "metrics.csv", "w", newline="")
    episodes_file = open(out / "episodes.csv", "w", newline="")
    metrics_writer = csv.DictWriter(metrics_file, fieldnames=METRIC_FIELDS)
    episodes_writer = csv.DictWriter(episodes_file, fieldnames=EPISODE_FIELDS)
    metrics_writer.writeheader()
    episodes_writer.writeheader()

    obs = env.reset()
    state = agent.state_from_game(env.game)
    episode = 0
    episode_return = 0.0
    next_eval = args.eval_every
    started = time.perf_counter()

    def greedy_action(_obs):
        return agent.act(agent.state_from_game(env.game), greedy=True)

    for decision in range(1, args.total_decisions + 1):
        action = agent.act(state)
        obs, reward, done, info = env.step(action)

        next_state = agent.state_from_game(env.game)
        terminal = done and not info["truncated"]
        agent.learn(state, action, reward, next_state, terminal)
        state = next_state
        episode_return += reward

        if done:
            episode += 1
            levels_cleared = info["level"] - 1
            dots = levels_cleared * N_PELLETS + (N_PELLETS - info["dots_left"])
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
            })
            episodes_file.flush()

            if agent.decisions >= next_eval:
                result = evaluate(env, greedy_action, episodes=args.eval_episodes,
                                  seed0=10_000)
                elapsed = time.perf_counter() - started
                row = {
                    "decision": decision,
                    "episode": episode,
                    "epsilon": round(agent.epsilon, 4),
                    "wall_seconds": round(elapsed, 2),
                }
                row.update({k: round(v, 4) for k, v in result.items()
                            if isinstance(v, float)})
                metrics_writer.writerow(row)
                metrics_file.flush()
                agent.save(out / "q.pkl")
                print(
                    f"decision {decision:>8} | eps {agent.epsilon:.3f} | "
                    f"return {result['return_mean']:7.2f} | "
                    f"score {result['score_mean']:7.1f} | "
                    f"dots {result['dots_mean']:6.1f} | "
                    f"frames {result['frames_mean']:7.1f} | "
                    f"deaths {result['deaths_mean']:.2f} | "
                    f"clears {result['clears_mean']:.2f} | {elapsed:.0f}s"
                )
                next_eval += args.eval_every

            obs = env.reset()
            state = agent.state_from_game(env.game)
            episode_return = 0.0

    agent.save(out / "q.pkl")
    metrics_file.close()
    episodes_file.close()
    print(f"done: {args.total_decisions} decisions, {len(agent.q)} states, "
          f"{time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
