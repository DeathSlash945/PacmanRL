"""Aggregate the 30-episode DQN evaluation JSONs across training seeds.

    python -m rl.aggregate

Reads runs/eval_dqn_seed*.json and runs/eval_random.json, prints a markdown
table, and saves a bar chart with error bars (mean +/- std across training
seeds).  Q-learning was removed from the final system; the only comparison
here is DQN versus the random-action baseline.
"""

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRICS = ("return", "score", "dots", "frames", "deaths", "clears")


def load_group(pattern):
    runs = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            data = json.load(f)
        runs.append((Path(path).stem, data["summary"]))
    return runs


def group_stats(runs):
    stats = {}
    for metric in METRICS:
        means = np.array([summary[metric]["mean"] for _, summary in runs], dtype=float)
        stats[metric] = {
            "mean": float(means.mean()),
            "std": float(means.std(ddof=1)) if len(means) > 1 else 0.0,
            "seed_means": means.tolist(),
        }
    return stats


def main():
    groups = {
        "DQN": load_group("runs/eval_dqn_seed*.json"),
        "Random": load_group("runs/eval_random.json"),
    }

    summary = {}
    for name, runs in groups.items():
        if runs:
            summary[name] = group_stats(runs)

    print("| Agent | Return | Score | Pellets | Survival | Deaths | Clears |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, stats in summary.items():
        cells = []
        for metric in METRICS:
            m, s = stats[metric]["mean"], stats[metric]["std"]
            cells.append(f"{m:.2f} +/- {s:.2f}")
        print(f"| {name} | " + " | ".join(cells) + " |")

    out = Path("runs/seed_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {out}")

    names = [name for name in summary if name != "Random"]
    if not names:
        return

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, metric in zip(axes, ("score", "dots", "frames", "deaths")):
        values = [summary[name][metric]["mean"] for name in names]
        errors = [summary[name][metric]["std"] for name in names]
        ax.bar(names, values, yerr=errors, capsize=6, color="tab:orange")
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Final greedy evaluation (mean +/- std across training seeds)")
    fig.tight_layout()
    out2 = Path("runs/seed_summary.png")
    fig.savefig(out2, dpi=140)
    print(f"saved {out2}")


if __name__ == "__main__":
    main()
