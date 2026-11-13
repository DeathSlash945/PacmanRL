"""Build one presentable training showcase, metrics and a summary.

`train.py` calls :func:`build_report` after every run.  The same command
also works on an existing output directory, and on a run that is still
training: if ``run.json`` has not been written yet, the report falls back
to ``progress.json``, ``eval_gen*.json`` and ``episodes.csv``.

    python -m rl.report --run runs/train_dqn_curriculum

The module deliberately produces one six-panel figure,
``training_showcase.png``.  The old separate ``training_curves*.png`` and
``training_episodes.png`` files were confusing: the evaluation curve could
contain a single point, and the final-bars chart had no comparison once
Q-learning was removed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

METRIC_FIELDS = [
    "generation",
    "decision",
    "episode",
    "epsilon",
    "updates",
    "loss_mean",
    "return_mean",
    "return_std",
    "score_mean",
    "score_std",
    "dots_mean",
    "dots_std",
    "frames_mean",
    "frames_std",
    "deaths_mean",
    "deaths_std",
    "clears_mean",
    "clears_std",
    "wall_seconds",
]

RESULT_NAMES = (
    "return",
    "score",
    "dots",
    "frames",
    "deaths",
    "clears",
)

LEGACY_PLOTS = (
    "training_curves.png",
    "training_curves_final.png",
    "training_episodes.png",
)


def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _as_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _eval_files(out: Path) -> list[tuple[int, dict]]:
    """Return ``(generation, result)`` pairs from ``eval_genN.json`` files."""
    found = []
    for path in out.glob("eval_gen*.json"):
        match = re.fullmatch(r"eval_gen(\d+)\.json", path.name)
        if not match:
            continue
        try:
            found.append((int(match.group(1)), _read_json(path)))
        except (OSError, json.JSONDecodeError):
            continue
    found.sort(key=lambda item: item[0])
    return found


def _episodes_by_generation(path: Path) -> dict[int, int]:
    """Map generation -> latest decision seen in ``episodes.csv``."""
    if not path.exists():
        return {}
    latest: dict[int, int] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            generation = int(_as_float(row.get("generation"), 0))
            if generation <= 0:
                continue
            decision = int(_as_float(row.get("decision"), 0))
            latest[generation] = max(latest.get(generation, 0), decision)
    return latest


def _live_run_info(out: Path) -> dict:
    """Build a run.json-shaped dict from partial logs while a run is active."""
    progress_path = out / "progress.json"
    progress = _read_json(progress_path) if progress_path.exists() else {}
    best_path = out / "best.json"
    best = _read_json(best_path) if best_path.exists() else {}

    episodes_path = out / "episodes.csv"
    decisions_by_generation = _episodes_by_generation(episodes_path)
    eval_files = _eval_files(out)

    generation_results = []
    for generation, result in eval_files:
        generation_results.append({
            "generation": generation,
            "training_level_cap": None,
            "evaluation_level_cap": None,
            "decisions": decisions_by_generation.get(generation, 0),
            "epsilon": 0.0,
            "updates": 0,
            "loss_mean": 0.0,
            "wall_seconds": 0.0,
            "result": result,
        })

    if not generation_results and progress.get("result"):
        generation = int(_as_float(progress.get("last_generation"), 0))
        generation_results.append({
            "generation": generation,
            "training_level_cap": progress.get("training_level_cap"),
            "evaluation_level_cap": progress.get("evaluation_level_cap"),
            "decisions": int(_as_float(progress.get("decisions"), 0)),
            "epsilon": 0.0,
            "updates": 0,
            "loss_mean": 0.0,
            "wall_seconds": 0.0,
            "result": progress["result"],
        })

    last_evaluated_generation = (
        generation_results[-1]["generation"] if generation_results else 0
    )
    latest_episode_generation = max(decisions_by_generation, default=0)
    latest_generation = max(last_evaluated_generation,
                            latest_episode_generation,
                            int(_as_float(progress.get("last_generation"), 0)))
    latest_result = (
        generation_results[-1]["result"]
        if generation_results
        else progress.get("result")
    )
    best_result = best.get("result") or progress.get("result") or latest_result
    if best.get("generation") is not None:
        best_generation = int(_as_float(best.get("generation"), 0))
    elif progress.get("best_generation") is not None:
        best_generation = int(_as_float(progress.get("best_generation"), 0))
    else:
        best_generation = None
    best_source = best.get("source") or progress.get("best_source")
    logged_decisions = max(
        decisions_by_generation.values(),
        default=int(_as_float(progress.get("decisions"), 0)),
    )

    return {
        "algorithm": "dqn",
        "_live": True,
        "total_decisions": int(_as_float(progress.get("decisions"), 0)),
        "completed_decisions": logged_decisions,
        "generations": latest_generation,
        "start_generation": 1,
        "curriculum": None,
        "curriculum_max_level": None,
        "interrupted": False,
        "seed": 0,
        "frame_skip": None,
        "max_frames": None,
        "level_cap": progress.get("training_level_cap"),
        "score_scale": None,
        "death_penalty": None,
        "stay_penalty": None,
        "dot_not_eaten_penalty": None,
        "dot_not_eaten_grace": None,
        "level_clear_bonus": None,
        "ghost_score_scale": None,
        "danger_penalty": None,
        "danger_radius": None,
        "dots_removed_min": None,
        "dots_removed_max": None,
        "use_ghost_dir": None,
        "width": None,
        "warmup": None,
        "train_every": None,
        "lr": None,
        "lr_final": None,
        "epsilon_end": None,
        "epsilon_decay": None,
        "replay_size": None,
        "wall_seconds": 0.0,
        "generation_results": generation_results,
        "best_generation": best_generation,
        "best_source": best_source,
        "selected_generation": (
            best_generation
            if best_generation is not None
            else (latest_generation or None)
        ),
        "best_eval": best_result,
        "final_eval": latest_result,
        "updated_at": progress.get("updated_at"),
        "checkpoint": str(out / "dqn.pt") if (out / "dqn.pt").exists() else None,
        "best_checkpoint": (
            str(out / "best.pt") if (out / "best.pt").exists() else None
        ),
    }


def _generation_rows(run_info: dict) -> list[dict]:
    """Turn generation results into metrics.csv rows."""
    rows = []
    for item in run_info.get("generation_results", []) or []:
        result = item.get("result") or {}
        if not result:
            continue
        row = {
            "generation": item.get("generation", 0),
            "decision": item.get("decisions", 0),
            "episode": item.get("episode", 0),
            "epsilon": item.get("epsilon", 0.0),
            "updates": item.get("updates", 0),
            "loss_mean": item.get("loss_mean", 0.0),
            "wall_seconds": item.get("wall_seconds", 0.0),
        }
        for name in RESULT_NAMES:
            row[f"{name}_mean"] = result.get(f"{name}_mean", 0.0)
            row[f"{name}_std"] = result.get(f"{name}_std", 0.0)
        rows.append(row)

    if not rows and run_info.get("final_eval"):
        result = run_info["final_eval"]
        row = {
            "generation": run_info.get("selected_generation", 0),
            "decision": run_info.get("completed_decisions", 0),
            "episode": 0,
            "epsilon": 0.0,
            "updates": 0,
            "loss_mean": 0.0,
            "wall_seconds": run_info.get("wall_seconds", 0.0),
        }
        for name in RESULT_NAMES:
            row[f"{name}_mean"] = result.get(f"{name}_mean", 0.0)
            row[f"{name}_std"] = result.get(f"{name}_std", 0.0)
        rows.append(row)

    return rows


def _write_metrics(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _episode_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        rows = [row for row in csv.DictReader(f)
                if row.get("score") not in (None, "")]
    if not rows:
        return {}

    def values(name):
        return [_as_float(row.get(name), 0.0) for row in rows]

    scores = values("score")
    dots = values("dots")
    dots_removed = values("dots_removed")
    frames = values("frames")
    deaths = values("deaths")
    clears = values("clears")
    decisions = values("decision")

    def mean(items):
        return sum(items) / len(items) if items else 0.0

    def last_mean(items, n):
        return mean(items[-n:])

    best = max(range(len(scores)), key=lambda i: scores[i])
    return {
        "episodes": len(rows),
        "decisions": int(decisions[-1]) if decisions else 0,
        "score_mean": round(mean(scores), 2),
        "score_max": round(scores[best], 2),
        "score_last_100_mean": round(last_mean(scores, 100), 2),
        "score_last_20_mean": round(last_mean(scores, 20), 2),
        "dots_mean": round(mean(dots), 2),
        "dots_last_100_mean": round(last_mean(dots, 100), 2),
        "dots_removed_mean": round(mean(dots_removed), 2),
        "frames_mean": round(mean(frames), 2),
        "deaths_mean": round(mean(deaths), 2),
        "clears_mean": round(mean(clears), 2),
        "best_episode": int(_as_float(rows[best].get("episode"), 0)),
        "best_score": round(scores[best], 2),
    }


def _cap(value) -> str:
    if value is None:
        return "none"
    return str(value)


def _cap_for_report(value, live: bool) -> str:
    """A live partial log cannot know the cap that was passed on the CLI."""
    if value is None and live:
        return "?"
    return _cap(value)


def _show(value, fmt="{:.2f}") -> str:
    if value is None:
        return "not recorded"
    try:
        return fmt.format(float(value))
    except (TypeError, ValueError):
        return str(value)


def _generation_label(run_info: dict, generation) -> str:
    if generation is None:
        return "not evaluated"
    if int(generation) == 0:
        return "resumed checkpoint (generation 0)"
    return f"generation {int(generation)}"


def _best_lines(run_info: dict) -> list[str]:
    return [
        f"- Best generation: **{_generation_label(run_info, run_info.get('best_generation'))}**"
    ]


def _selected_label(run_info: dict) -> str:
    return _generation_label(run_info, run_info.get("selected_generation"))


def _format_result(result: dict | None) -> str:
    if not result:
        return "not evaluated"
    return (
        f"score {result.get('score_mean', 0):.1f} +/- "
        f"{result.get('score_std', 0):.1f}, "
        f"pellets {result.get('dots_mean', 0):.1f}, "
        f"survival {result.get('frames_mean', 0):.0f} frames, "
        f"clears {result.get('clears_mean', 0):.2f}"
    )


def _write_summary(path: Path, run_info: dict, episode_summary: dict) -> None:
    live = bool(run_info.get("_live"))
    generation_rows = run_info.get("generation_results", []) or []
    status = "in progress (partial log)" if live else "finished"
    grace = run_info.get("dot_not_eaten_grace")
    grace_text = str(int(grace)) if grace is not None else "n/a"

    lines = [
        "# Training summary",
        "",
        f"- Status: **{status}**",
        f"- Algorithm: **{run_info.get('algorithm', 'dqn')}**",
    ]
    if live:
        lines.extend([
            f"- Decisions in completed generations: "
            f"**{int(_as_float(run_info.get('total_decisions'), 0)):,}**",
            f"- Decisions logged so far: "
            f"**{int(_as_float(run_info.get('completed_decisions'), 0)):,}**",
        ])
    else:
        lines.extend([
            f"- Decisions in this run: "
            f"**{int(_as_float(run_info.get('total_decisions'), 0)):,}**",
            f"- Completed decisions: "
            f"**{int(_as_float(run_info.get('completed_decisions'), 0)):,}**",
        ])
    lines.extend([
        f"- Generations: **{run_info.get('start_generation', 1)}.."
        f"{run_info.get('generations', 0)}**",
        f"- Seed: **{run_info.get('seed', 0)}**",
        f"- Training cap at start: "
        f"**{_cap_for_report(run_info.get('level_cap'), live)}**",
        "- Reward: "
        f"score_scale={_show(run_info.get('score_scale'), '{:.2f}')}, "
        f"death={_show(run_info.get('death_penalty'), '{:.1f}')}, "
        f"stay={_show(run_info.get('stay_penalty'), '{:.2f}')}, "
        f"dot_not_eaten={_show(run_info.get('dot_not_eaten_penalty'), '{:.2f}')} "
        f"after {grace_text} decisions, "
        f"clear={_show(run_info.get('level_clear_bonus'), '{:.1f}')}, "
        f"ghost_score={_show(run_info.get('ghost_score_scale'), '{:.2f}')}, "
        f"danger={_show(run_info.get('danger_penalty'), '{:.3f}')}"
        f"/{run_info.get('danger_radius') if run_info.get('danger_radius') is not None else 'n/a'} tiles",
        f"- Endgame curriculum: **dots removed at reset "
        f"{run_info.get('dots_removed_min') if run_info.get('dots_removed_min') is not None else 'n/a'}"
        f"..{run_info.get('dots_removed_max') if run_info.get('dots_removed_max') is not None else 'n/a'}**",
        f"- Observation: **ghost directions "
        f"{'on' if run_info.get('use_ghost_dir') else ('off' if run_info.get('use_ghost_dir') is not None else 'n/a')}**",
        *_best_lines(run_info),
        f"- Checkpoint used for final evaluation/live play: "
        f"**{_selected_label(run_info)}**",
        f"- Wall time: **{_as_float(run_info.get('wall_seconds'), 0.0):.1f}s**",
        f"- Interrupted: **{run_info.get('interrupted', False)}**",
        "",
        "## Latest greedy evaluation" if live
        else "## Final greedy evaluation",
        "",
        f"- {_format_result(run_info.get('final_eval'))}",
        f"- Best evaluated generation: "
        f"{_format_result(run_info.get('best_eval'))}",
        "",
        "## Generation evaluations",
        "",
    ])

    if generation_rows:
        lines.extend([
            "| Gen | Decisions | Train cap | Eval cap | Score | Pellets | "
            "Frames | Deaths | Clears |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for item in generation_rows:
            result = item.get("result") or {}
            lines.append(
                f"| {item.get('generation', 0)} | "
                f"{item.get('decisions', 0):,} | "
                f"{_cap_for_report(item.get('training_level_cap'), live)} | "
                f"{_cap_for_report(item.get('evaluation_level_cap'), live)} | "
                f"{result.get('score_mean', 0):.1f} | "
                f"{result.get('dots_mean', 0):.1f} | "
                f"{result.get('frames_mean', 0):.0f} | "
                f"{result.get('deaths_mean', 0):.2f} | "
                f"{result.get('clears_mean', 0):.2f} |"
            )
    else:
        lines.append("No generation evaluation was written for this run.")

    if any(item.get("generation") == 0 for item in generation_rows):
        lines.extend([
            "",
            "Generation 0 is the resumed checkpoint evaluated before any new "
            "training.  It is included so a continuation cannot silently "
            "replace a better starting policy.",
        ])

    lines.extend(["", "## Training episodes", ""])
    if episode_summary:
        lines.extend([
            f"- Episodes: **{episode_summary['episodes']:,}**",
            f"- Mean score over all episodes: "
            f"**{episode_summary['score_mean']:.1f}**",
            f"- Mean score over the last 100 episodes: "
            f"**{episode_summary['score_last_100_mean']:.1f}**",
            f"- Best training score: **{episode_summary['best_score']:.1f}** "
            f"(episode {episode_summary['best_episode']})",
            f"- Mean pellets eaten: **{episode_summary['dots_mean']:.1f}**",
            f"- Mean pellets pre-removed by the curriculum: "
            f"**{episode_summary.get('dots_removed_mean', 0.0):.1f}**",
            f"- Mean survival: **{episode_summary['frames_mean']:.0f} frames**",
            f"- Mean levels cleared: **{episode_summary['clears_mean']:.2f}**",
            "",
            "Training pellets are actual dots eaten.  Endgame-curriculum "
            "episodes start with some ordinary dots already removed, so their "
            "pellet count is lower even when the agent clears the level.  The "
            "fixed-seed evaluation always uses the full 244-pellet maze.",
        ])
    else:
        lines.append("No `episodes.csv` data was found.")

    lines.extend([
        "",
        "## What the six-panel showcase shows",
        "",
        "- **Score, pellets and survival:** training episodes (faint lines), "
        "50-episode rolling means (bold), and fixed-seed greedy evaluation "
        "(orange markers, mean +/- std).",
        "- **Episode return and DQN loss:** the optimisation and reward "
        "signals over training.",
        "- **Greedy evaluation by generation:** score and pellets on the same "
        "generation axis, with the latest evaluation in the text box.",
        "",
        "## Plots and data",
        "",
        "- `training_showcase.png`: the single-page six-panel figure.",
        "- `metrics.csv`: per-generation fixed-seed evaluation data.",
        "- `episodes.csv`: raw per-episode training data.",
        "- `run.json`: full configuration and results (once the run ends).",
        "- `stats.json`: a compact machine-readable copy of this summary.",
        "",
        "Rebuild this report at any time with:",
        "",
        "```bash",
        "python -m rl.report --run <this directory>",
        "```",
        "",
    ])
    if live:
        lines.extend([
            "This partial report was built while the run was still training. "
            "The final report replaces it when `run.json` is written.",
            "",
        ])
    path.write_text("\n".join(lines))


def _remove_legacy_plots(out: Path) -> None:
    for name in LEGACY_PLOTS:
        path = out / name
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass


def build_report(
    out_dir: str | Path,
    make_plots: bool = True,
    window: int = 50,
) -> dict:
    """Create the showcase, metrics, `stats.json` and `summary.md` for a run.

    The function tolerates missing optional files, so it also works for old
    runs, runs with `--eval-episodes 0`, and a live run whose ``run.json``
    has not been written yet.
    """
    out = Path(out_dir).expanduser().resolve()
    run_path = out / "run.json"
    live = not run_path.exists()
    if live:
        run_info = _live_run_info(out)
        if (not run_info["generation_results"]
                and not (out / "episodes.csv").exists()):
            raise FileNotFoundError(
                f"{run_path} is missing and no partial logs were found in {out}"
            )
    else:
        run_info = _read_json(run_path)

    rows = _generation_rows(run_info)
    metrics_path = out / "metrics.csv"
    if rows:
        _write_metrics(metrics_path, rows)

    episodes_path = out / "episodes.csv"
    episode_summary = _episode_summary(episodes_path)
    showcase_path = out / "training_showcase.png"

    artifacts = {
        "metrics": str(metrics_path) if rows else None,
        "episodes": str(episodes_path) if episodes_path.exists() else None,
        "training_showcase": (
            str(showcase_path) if (rows or episode_summary) else None
        ),
        "summary": str(out / "summary.md"),
        "stats": str(out / "stats.json"),
    }
    stats = {
        "run": run_info,
        "live": live,
        "training_episode_summary": episode_summary,
        "evaluation_rows": rows,
        "artifacts": artifacts,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    _write_summary(out / "summary.md", run_info, episode_summary)

    if make_plots and artifacts["training_showcase"]:
        from .plot import plot_showcase

        plot_showcase(
            episodes_path if episode_summary else None,
            metrics_path if rows else None,
            showcase_path,
            window=window,
        )
        _remove_legacy_plots(out)

    return {
        "out": str(out),
        "metrics": artifacts["metrics"],
        "summary": artifacts["summary"],
        "stats": artifacts["stats"],
        "training_showcase": artifacts["training_showcase"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the six-panel showcase, stats and summary.md "
                    "for a finished or live run."
    )
    parser.add_argument("--run", type=str, default="runs/train_dqn",
                        help="output directory containing run.json or logs")
    parser.add_argument("--no-plots", action="store_true",
                        help="write metrics/stats/summary but skip the PNG")
    parser.add_argument("--window", type=int, default=50,
                        help="rolling-mean window in episodes")
    args = parser.parse_args()

    result = build_report(
        args.run,
        make_plots=not args.no_plots,
        window=args.window,
    )
    print("report written:")
    for name, path in result.items():
        if path:
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
