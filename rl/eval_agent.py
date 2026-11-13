"""Evaluate a saved DQN agent on fixed seeds and print/save the metrics.

Examples:
    python -m rl.eval_agent --agent dqn --checkpoint runs/dqn/dqn.pt
    python -m rl.eval_agent --agent dqn --checkpoint runs/dqn/best.pt \
        --width 64 --level-cap 0
    python -m rl.eval_agent --agent random
"""

import argparse
import json
import random
from pathlib import Path

from .agents.dqn import DQNAgent
from .config import EnvConfig
from .evaluate import run_episode
from .pacman_env import PacmanEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=("dqn", "random"), required=True)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--max-frames", type=int, default=18_000)
    p.add_argument("--level-cap", type=int, default=1,
                   help="stop after this level; 0 evaluates the full game")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    level_cap = args.level_cap if args.level_cap > 0 else None
    env = PacmanEnv(EnvConfig(
        frame_skip=args.frame_skip,
        max_frames=args.max_frames,
        level_cap=level_cap,
        seed=args.seed,
    ))

    if args.agent == "random":
        rng = random.Random(args.seed)
        act_fn = lambda _obs: rng.randrange(4)
        label = "random"
    else:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for --agent dqn")
        agent = DQNAgent(replay_size=1, width=args.width, device=args.device)
        agent.load(args.checkpoint)
        act_fn = lambda obs: agent.act(obs, greedy=True)
        label = "dqn"

    rows = [run_episode(env, act_fn, seed=10_000 + i) for i in range(args.episodes)]

    keys = ("return", "score", "dots", "frames", "deaths", "clears")
    summary = {}
    for key in keys:
        values = [row[key] for row in rows]
        summary[key] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    print(f"{label}: {args.episodes} episodes")
    for key in keys:
        print(f"  {key:7s} mean={summary[key]['mean']:8.2f} "
              f"min={summary[key]['min']:8.2f} max={summary[key]['max']:8.2f}")

    out = Path(args.out) if args.out else Path("runs") / f"eval_{label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"agent": label, "summary": summary, "episodes": rows}, f, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
