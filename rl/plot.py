"""Plot the evaluation curves for Q-learning and DQN side by side.

    python -m rl.plot \
        --q runs/qlearning/metrics.csv \
        --dqn runs/dqn/metrics.csv \
        --out runs/comparison.png

The shaded band is mean +/- one standard deviation over the evaluation
episodes.  If runs/eval_random.json exists, random-play baselines are
drawn as dashed horizontal lines.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRICS = (
    ("return", "Episode return"),
    ("score", "Score"),
    ("dots", "Pellets eaten"),
    ("frames", "Survival (frames)"),
    ("deaths", "Deaths"),
    ("clears", "Levels cleared"),
)
COLORS = {"q": "tab:blue", "dqn": "tab:orange"}


def load_metrics(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            parsed = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
    return rows


def metric_values(rows, name):
    x = np.array([r["decision"] for r in rows], dtype=float)
    mean = np.array([r[f"{name}_mean"] for r in rows], dtype=float)
    std = np.array([r.get(f"{name}_std", 0.0) for r in rows], dtype=float)
    return x, mean, std


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--q", type=str, default="runs/qlearning/metrics.csv")
    p.add_argument("--dqn", type=str, default="runs/dqn/metrics.csv")
    p.add_argument("--random", type=str, default="runs/eval_random.json")
    p.add_argument("--out", type=str, default="runs/comparison.png")
    args = p.parse_args()

    runs = {}
    for label, path in (("Q-learning", args.q), ("DQN", args.dqn)):
        if Path(path).exists():
            runs[label] = load_metrics(path)
        else:
            print(f"missing: {path}")

    if not runs:
        raise SystemExit("no metrics files found")

    random_summary = None
    if Path(args.random).exists():
        with open(args.random) as f:
            random_summary = json.load(f)["summary"]

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for ax, (name, title) in zip(axes.flat, METRICS):
        for label, rows in runs.items():
            x, mean, std = metric_values(rows, name)
            color = COLORS["q"] if "Q" in label else COLORS["dqn"]
            ax.plot(x, mean, label=label, color=color, linewidth=2)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)
        if random_summary is not None:
            ax.axhline(random_summary[name]["mean"], color="gray",
                       linestyle="--", linewidth=1.2, label="random")
        ax.set_title(title)
        ax.set_xlabel("training decisions")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle("Pac-Man RL: Q-learning vs DQN (greedy evaluation)")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"saved {args.out}")

    # Final checkpoint comparison bars.
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
    final_metrics = (("score", "Final score"), ("dots", "Final pellets"),
                     ("frames", "Final survival (frames)"))
    labels = list(runs)
    for ax, (name, title) in zip(axes2, final_metrics):
        values = [runs[label][-1][f"{name}_mean"] for label in labels]
        ax.bar(labels, values, color=[COLORS["q"] if "Q" in l else COLORS["dqn"]
                                      for l in labels])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    out2 = str(Path(args.out).with_name("comparison_final.png"))
    fig2.savefig(out2, dpi=140)
    print(f"saved {out2}")


if __name__ == "__main__":
    main()
