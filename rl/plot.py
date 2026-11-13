"""Single-page six-panel training showcase for the DQN experiments.

The old report generated three separate figures: an evaluation curve, a
"final bars" chart and a training-episode plot.  The final-bars chart was
mostly empty and the evaluation curve was unreadable when a run had only
one generation.  This module instead builds one presentation-ready figure
that always has six useful panels:

1. training score + greedy-evaluation score
2. training pellets + greedy-evaluation pellets
3. training survival + greedy-evaluation survival
4. training episode return
5. training DQN loss
6. greedy-evaluation score and pellets by generation

Faint lines are individual training episodes, bold lines are rolling
means, and markers are the fixed-seed greedy evaluations.  The function
tolerates a live run whose ``run.json`` has not been written yet: it only
needs ``episodes.csv`` and/or ``metrics.csv``.

Examples::

    python -m rl.plot --run runs/train_dqn_curriculum
    python -m rl.plot --episodes runs/train_dqn/episodes.csv \
        --metrics runs/train_dqn/metrics.csv \
        --out runs/train_dqn/training_showcase.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np

TRAIN_COLOR = "#2563eb"
EVAL_COLOR = "#f97316"
PELLET_COLOR = "#0f766e"
RAW_COLOR = "#94a3b8"
GRID_COLOR = "#cbd5e1"
TEXT_COLOR = "#334155"
MUTED_COLOR = "#64748b"


def load_rows(path: str | Path | None) -> list[dict]:
    """Load a CSV into a list of dicts, converting numeric fields to float."""
    if path is None:
        return []
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []

    rows: list[dict] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if not row:
                continue
            parsed = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            # A live run can be caught while writing its last line.  Ignore
            # obviously incomplete rows instead of plotting zeros for them.
            if (parsed.get("score") in (None, "")
                    and parsed.get("score_mean") in (None, "")):
                continue
            rows.append(parsed)
    return rows


def _number(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _rolling_mean(values, window: int):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    window = max(1, min(int(window), len(values)))
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="valid")


def _format_decisions(value, _position=None) -> str:
    value = float(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def _style_axis(ax) -> None:
    ax.grid(color=GRID_COLOR, linewidth=0.7, alpha=0.55)
    ax.set_facecolor("#ffffff")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#94a3b8")
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.xaxis.set_major_formatter(FuncFormatter(_format_decisions))


def _plot_training_series(ax, episodes: list[dict], key: str,
                          window: int, ylabel: str) -> None:
    x = np.array([_number(row, "decision") for row in episodes], dtype=float)
    y = np.array([_number(row, key) for row in episodes], dtype=float)

    ax.plot(x, y, color=RAW_COLOR, alpha=0.18, linewidth=0.7,
            zorder=1)
    roll = _rolling_mean(y, window)
    if len(roll):
        effective = min(max(1, int(window)), len(x))
        ax.plot(x[effective - 1:], roll, color=TRAIN_COLOR, linewidth=2.2,
                zorder=3)
    ax.set_ylabel(ylabel, color=TEXT_COLOR, fontsize=9)
    _style_axis(ax)


def _plot_eval_overlay(ax, evals: list[dict], metric: str) -> None:
    if not evals:
        return
    x = np.array([_number(row, "decision") for row in evals], dtype=float)
    mean = np.array(
        [_number(row, f"{metric}_mean") for row in evals], dtype=float
    )
    std = np.array(
        [_number(row, f"{metric}_std") for row in evals], dtype=float
    )
    ax.errorbar(
        x, mean, yerr=std, fmt="o", color=EVAL_COLOR, ecolor=EVAL_COLOR,
        elinewidth=1.0, capsize=2.5, markersize=4.5, zorder=5,
        markeredgecolor="white", markeredgewidth=0.6,
    )


def _eval_panel(ax, evals: list[dict]) -> None:
    """Panel 6: score and pellet means by generation."""
    if not evals:
        ax.set_title("Greedy evaluation", color=TEXT_COLOR, fontsize=11,
                     fontweight="bold")
        ax.text(0.5, 0.55, "No evaluation data yet",
                ha="center", va="center", transform=ax.transAxes,
                color=MUTED_COLOR, fontsize=10)
        _style_axis(ax)
        return

    generations = [int(_number(row, "generation")) for row in evals]
    scores = np.array([_number(row, "score_mean") for row in evals])
    score_std = np.array([_number(row, "score_std") for row in evals])
    pellets = np.array([_number(row, "dots_mean") for row in evals])
    pellet_std = np.array([_number(row, "dots_std") for row in evals])

    ax.errorbar(
        generations, scores, yerr=score_std, fmt="-o", color=EVAL_COLOR,
        ecolor=EVAL_COLOR, elinewidth=1.0, capsize=2.5, markersize=5,
        linewidth=2.0, label="score", zorder=3,
        markeredgecolor="white", markeredgewidth=0.6,
    )
    ax.set_ylabel("greedy score", color=EVAL_COLOR, fontsize=9)
    ax.tick_params(axis="y", labelcolor=EVAL_COLOR, labelsize=8)
    if len(generations) == 1:
        # A single evaluation point is still useful, but the automatic
        # integer locator would repeat the same tick label across the axis.
        ax.set_xlim(generations[0] - 0.5, generations[0] + 0.5)
        ax.set_xticks([generations[0]])
    elif len(generations) <= 12:
        ax.set_xticks(generations)
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    ax.set_xlabel("generation", color=TEXT_COLOR, fontsize=9)
    ax.set_title("Greedy evaluation by generation", color=TEXT_COLOR,
                 fontsize=11, fontweight="bold")
    _style_axis(ax)
    ax.tick_params(axis="y", labelcolor=EVAL_COLOR, labelsize=8)

    twin = ax.twinx()
    twin.errorbar(
        generations, pellets, yerr=pellet_std, fmt="-s", color=PELLET_COLOR,
        ecolor=PELLET_COLOR, elinewidth=1.0, capsize=2.5, markersize=4.2,
        linewidth=1.8, label="pellets", zorder=3,
        markeredgecolor="white", markeredgewidth=0.5,
    )
    twin.set_ylabel("pellets eaten", color=PELLET_COLOR, fontsize=9)
    twin.tick_params(axis="y", labelcolor=PELLET_COLOR, labelsize=8)
    twin.set_ylim(bottom=0)
    for spine in ("top", "left"):
        twin.spines[spine].set_visible(False)
    twin.spines["right"].set_color(PELLET_COLOR)

    # The twin errorbar can re-autoscale the shared x-axis; restore the
    # single-generation window after both artists exist.
    if len(generations) == 1:
        ax.set_xlim(generations[0] - 0.5, generations[0] + 0.5)
        ax.set_xticks([generations[0]])

    handles = [
        Line2D([0], [0], color=EVAL_COLOR, marker="o", linewidth=2,
               label="score"),
        Line2D([0], [0], color=PELLET_COLOR, marker="s", linewidth=1.8,
               label="pellets"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False,
              fontsize=8, ncol=2)

    final = evals[-1]
    info = (
        f"final eval (gen {int(_number(final, 'generation'))})\n"
        f"score   {_number(final, 'score_mean'):,.0f}\n"
        f"pellets {_number(final, 'dots_mean'):.1f} / 244\n"
        f"survival {_number(final, 'frames_mean'):,.0f} frames\n"
        f"clears  {_number(final, 'clears_mean'):.2f}"
    )
    ax.text(
        0.98, 0.04, info, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7.2, color=TEXT_COLOR,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor=GRID_COLOR, alpha=0.92),
    )


def plot_showcase(
    episodes_path: str | Path | None,
    metrics_path: str | Path | None,
    out_path: str | Path,
    window: int = 50,
    title: str = "Pac-Man DQN — training showcase",
):
    """Build the six-panel report figure and save it to ``out_path``."""
    episodes = load_rows(episodes_path)
    evals = load_rows(metrics_path)
    if not episodes and not evals:
        raise ValueError("need episodes.csv or metrics.csv to build a showcase")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # If only evaluation data exists, use it as the "training" series.
    training = episodes if episodes else evals
    training_is_eval = not episodes

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlecolor": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "figure.facecolor": "white",
    })

    fig, axes = plt.subplots(2, 3, figsize=(18.5, 10.2))
    axes = axes.reshape(2, 3)

    series = (
        ("score", "Training score", "score"),
        ("dots", "Training pellets", "pellets eaten"),
        ("frames", "Training survival", "game frames"),
        ("return", "Training episode return", "return"),
        ("loss_mean", "Training DQN loss", "Huber loss"),
    )

    for ax, (key, title_text, ylabel) in zip(axes.flat[:5], series):
        if training_is_eval:
            x = np.array([_number(row, "decision") for row in training],
                         dtype=float)
            mean_key = f"{key}_mean" if f"{key}_mean" in training[0] else key
            y = np.array([_number(row, mean_key) for row in training],
                         dtype=float)
            ax.plot(x, y, color=TRAIN_COLOR, linewidth=2.0, marker="o",
                    markersize=3.5)
            ax.set_ylabel(ylabel, color=TEXT_COLOR, fontsize=9)
            _style_axis(ax)
        else:
            _plot_training_series(ax, training, key, window, ylabel)
            if key in ("score", "dots", "frames", "return"):
                _plot_eval_overlay(ax, evals, key)

        ax.set_title(title_text)
        ax.set_xlabel("training decisions", fontsize=8.5,
                      color=TEXT_COLOR)
        if key in ("score", "dots", "frames"):
            ax.set_ylim(bottom=0)

    _eval_panel(axes.flat[5], evals)

    # Figure-level labels and legend.
    latest_decision = max(
        [_number(row, "decision") for row in training] or [0.0]
    )
    episode_count = len(episodes) if episodes else 0
    subtitle = (
        f"{episode_count:,} training episodes  ·  "
        f"{_format_decisions(latest_decision)} decisions  ·  "
        f"{len(evals)} greedy evaluation point"
        f"{'' if len(evals) == 1 else 's'}"
    )
    fig.suptitle(title, x=0.5, y=0.975, fontsize=20,
                 fontweight="bold", color="#0f172a")
    fig.text(0.5, 0.942, subtitle, ha="center", va="top",
             fontsize=9.5, color=MUTED_COLOR)

    legend_handles = [
        Line2D([0], [0], color=RAW_COLOR, alpha=0.5, linewidth=1.2,
               label="individual episode"),
        Line2D([0], [0], color=TRAIN_COLOR, linewidth=2.2,
               label=f"{max(1, int(window))}-episode rolling mean"),
        Line2D([0], [0], color=EVAL_COLOR, marker="o", linewidth=0,
               markersize=5, label="greedy evaluation (mean ± std)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.012))

    fig.text(
        0.5, 0.002,
        "Greedy evaluation uses the same fixed seeds for every generation. "
        "Shaded/vertical bars are one standard deviation across episodes.",
        ha="center", va="bottom", fontsize=7.8, color=MUTED_COLOR,
    )

    fig.subplots_adjust(
        left=0.055, right=0.945, top=0.895, bottom=0.09,
        hspace=0.38, wspace=0.28,
    )
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the six-panel DQN training showcase."
    )
    parser.add_argument("--run", default=None,
                        help="run directory containing episodes.csv/metrics.csv")
    parser.add_argument("--episodes", default=None,
                        help="path to episodes.csv (overrides --run)")
    parser.add_argument("--metrics", default=None,
                        help="path to metrics.csv (overrides --run)")
    parser.add_argument("--out", default=None,
                        help="output PNG (default: <run>/training_showcase.png)")
    parser.add_argument("--window", type=int, default=50,
                        help="rolling-mean window in episodes (default: 50)")
    parser.add_argument("--title", default="Pac-Man DQN — training showcase")
    parser.add_argument("--dqn", default=None,
                        help=argparse.SUPPRESS)  # old CLI compatibility
    args = parser.parse_args()

    if args.run:
        run = Path(args.run)
        episodes = Path(args.episodes) if args.episodes else run / "episodes.csv"
        metrics = Path(args.metrics) if args.metrics else run / "metrics.csv"
        out = Path(args.out) if args.out else run / "training_showcase.png"
    elif args.dqn:
        metrics = Path(args.dqn)
        episodes = Path(args.episodes) if args.episodes else (
            metrics.with_name("episodes.csv")
        )
        out = Path(args.out) if args.out else (
            metrics.with_name("training_showcase.png")
        )
    else:
        episodes = Path(args.episodes or "runs/train_dqn/episodes.csv")
        metrics = Path(args.metrics) if args.metrics else (
            episodes.with_name("metrics.csv")
        )
        out = Path(args.out or episodes.with_name("training_showcase.png"))

    plot_showcase(
        episodes if episodes.exists() else None,
        metrics if metrics.exists() else None,
        out,
        window=args.window,
        title=args.title,
    )


if __name__ == "__main__":
    main()
