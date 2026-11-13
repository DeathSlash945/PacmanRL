#!/usr/bin/env python3
"""Train a DQN Pac-Man agent from scratch, save a GIF, then watch it play.

This is the single-file entry point for the project.  It deliberately uses
the existing `rl` package instead of duplicating the environment or agents:

    python train.py                          # DQN, 1M decisions, 100k-frame GIF
    python train.py --gif-frames 0 --no-watch  # training only
    python train.py --headless --no-watch       # server / smoke test

    # For example a long train sesh: ten generations and level curriculum.
    python train.py --generations 10 --total-decisions 10000000 \
        --curriculum --curriculum-max-level 4 --max-frames 60000 \
        --width 64 --train-every 1 --gamma 0.995 \
        --epsilon-decay 1000000 --epsilon-end 0.02 \
        --lr-final 1e-5 --eval-episodes 30 --eval-level-cap 0 \
        --out runs/overnight

What happens, in order
----------------------
1. A brand-new DQN is created (or `--resume` loads the latest checkpoint).
2. Training is split into `--generations` generations.  With `--curriculum`,
   the earliest generations only have to clear one level, then two, and so
   on; later generations are free to play the whole game.  The replay buffer
   and network survive between generations, so this is one continuous run.
3. Every Nth decision we render one frame and stream the raw RGB pixels into
   ffmpeg, so RAM does not grow with the number of recorded frames.
4. After each generation the agent is evaluated on fixed seeds, checkpointed,
   and compared with the best checkpoint so far.
5. ffmpeg converts the temporary MP4 into an optimised palette GIF.
6. `metrics.csv`, `stats.json`, `summary.md` and `training_showcase.png`
   are written so the run's development is visible in one place.
7. A fresh `Game(fx=True)` is started and the best agent plays greedily until
   all three lives are gone.  It may continue into level 2+ if it clears a
   level.

To continue an existing run, either use `--resume` with the same output
directory (the next unfinished generation is read from
`progress.json`/`run.json`) or use `--resume-from` with a new directory so
the old logs are preserved:

    python train.py --resume --generations 10 --total-decisions 9000000 \
        --curriculum --curriculum-max-level 4 --out runs/train_dqn

    python train.py --resume-from runs/train_dqn/dqn.pt \
        --generations 10 --total-decisions 9000000 \
        --curriculum --curriculum-max-level 4 --out runs/train_dqn_cont

Add `--reset-epsilon` for a fresh exploration phase, or pass
`--start-generation G` to repeat a particular generation.

The original game files are never modified.  The script only calls the
public API used by the rest of the project: `Game.update()`,
`Pacman.set_dir()`, and read-only attributes.

Why no Q-learning?  It was tried and discarded; the short version is in
README.md and REPORT.md.  The project intentionally ships one working path.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

# ---------------------------------------------------------------------------
# SDL setup must happen *before* pygame is imported.
#
# On a normal desktop DISPLAY/WAYLAND_DISPLAY is set, so pygame opens a real
# window for the final live game.  On a server (or with --headless) we use
# SDL's dummy video driver: pygame can still create surfaces, draw, and save
# pixels, but no window is required.  Audio is unused by the game, so the
# dummy audio driver avoids ALSA warnings.
# ---------------------------------------------------------------------------
_HEADLESS_FLAG = "--headless" in sys.argv
_NO_DISPLAY = not (
    os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
)
if _HEADLESS_FLAG or _NO_DISPLAY:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame as pg

import map as maze
from env import Game
from rl.agents.dqn import DQNAgent
from rl.config import ACTIONS, EnvConfig, N_PELLETS, RewardConfig
from rl.evaluate import evaluate
from rl.obs import ObservationBuilder
from rl.pacman_env import PacmanEnv

# Import paths are relative to this file, but output paths should be stable
# even if the user launches the script from another directory.
ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------
def validate_args(args: argparse.Namespace) -> None:
    """Reject values that would otherwise fail in a confusing way."""
    if args.total_decisions <= 0:
        raise SystemExit("--total-decisions must be positive")
    if args.generations <= 0:
        raise SystemExit("--generations must be positive")
    if args.start_generation is not None and not (
        1 <= args.start_generation <= args.generations
    ):
        raise SystemExit("--start-generation must be between 1 and --generations")
    if args.curriculum and args.curriculum_max_level < 1:
        raise SystemExit("--curriculum-max-level must be at least 1")
    if args.eval_level_cap < -1:
        raise SystemExit("--eval-level-cap must be -1, 0, or a positive level")
    if args.frame_skip <= 0:
        raise SystemExit("--frame-skip must be positive")
    if args.train_every <= 0:
        raise SystemExit("--train-every must be positive")
    if args.log_every <= 0:
        raise SystemExit("--log-every must be positive")
    if args.live_fps <= 0:
        raise SystemExit("--live-fps must be positive")
    if args.width <= 0:
        raise SystemExit("--width must be positive")
    if args.epsilon_decay <= 0:
        raise SystemExit("--epsilon-decay must be positive")
    if args.lr_final is not None and args.lr_final <= 0:
        raise SystemExit("--lr-final must be positive")
    if args.eval_episodes < 0:
        raise SystemExit("--eval-episodes cannot be negative")
    if args.gif_frames < 0:
        raise SystemExit("--gif-frames cannot be negative")
    if args.dot_not_eaten_grace is not None and args.dot_not_eaten_grace < 0:
        raise SystemExit("--dot-not-eaten-grace cannot be negative")
    if args.dot_not_eaten_penalty is not None and args.dot_not_eaten_penalty > 0:
        raise SystemExit("--dot-not-eaten-penalty must be zero or negative")
    if args.dots_removed_min < 0 or args.dots_removed_max < 0:
        raise SystemExit("--dots-removed-min/max cannot be negative")
    if args.dots_removed_max < args.dots_removed_min:
        raise SystemExit("--dots-removed-max must be >= --dots-removed-min")
    if args.dots_removed_max > 240:
        raise SystemExit("--dots-removed-max cannot exceed 240 (four power "
                         "pellets are never removed)")
    if args.ghost_score_scale < 0:
        raise SystemExit("--ghost-score-scale cannot be negative")
    if args.danger_penalty < 0:
        raise SystemExit("--danger-penalty cannot be negative")
    if args.danger_radius < 0:
        raise SystemExit("--danger-radius cannot be negative")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a DQN Pac-Man agent from scratch, record a GIF, then play."
    )

    # How long and how many generations.
    p.add_argument("--total-decisions", type=int, default=1_000_000,
                   help="total environment steps across all generations; "
                        "one step = --frame-skip game frames")
    p.add_argument("--generations", type=int, default=1,
                   help="split training into this many evaluation/checkpoint "
                        "generations; the agent and replay buffer are shared")
    p.add_argument("--start-generation", type=int, default=None,
                   help="first generation to run; omit to auto-detect from "
                        "progress.json/run.json when --resume is used")
    p.add_argument("--curriculum", action="store_true",
                   help="increase the level cap each generation: 1, 2, 3, ... "
                        "then no cap after --curriculum-max-level")
    p.add_argument("--curriculum-max-level", type=int, default=4,
                   help="last level used as a curriculum cap; later generations "
                        "play with no cap")
    p.add_argument("--resume", action="store_true",
                   help="continue from --out/dqn.pt instead of starting fresh; "
                        "the next generation is detected from progress.json")
    p.add_argument("--resume-from", type=str, default=None,
                   help="load this checkpoint path instead of --out/dqn.pt; "
                        "useful for continuing into a new output directory "
                        "without overwriting the old run")
    p.add_argument("--reset-epsilon", action="store_true",
                   help="when resuming, restart epsilon decay at 1.0 for a new "
                        "exploration phase instead of keeping the low checkpoint "
                        "epsilon")
    p.add_argument("--partial-load", action="store_true",
                   help="when resuming an old checkpoint into a different "
                        "observation architecture, transfer all shape-compatible "
                        "weights instead of rebuilding the old architecture")
    p.add_argument("--no-ghost-dir", action="store_true",
                   help="disable the 16 ghost-direction input; new runs use it "
                        "by default, old checkpoints load without it automatically")
    p.add_argument("--no-best", action="store_true",
                   help="use the final checkpoint for the live game instead of "
                        "the best evaluated checkpoint")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None,
                   help="torch device for DQN, e.g. cuda or cpu; auto by default")

    # Environment settings.  These match the experiment configuration used
    # in REPORT.md so this script reproduces the same task.
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--max-frames", type=int, default=18_000,
                   help="episode time limit in game frames")
    p.add_argument("--level-cap", type=int, default=1,
                   help="stop episodes after this level; 0 means no cap")
    p.add_argument("--score-scale", type=float, default=None,
                   help="override RewardConfig.score_scale; default comes from rl/config.py")
    p.add_argument("--death-penalty", type=float, default=None,
                   help="override RewardConfig.death_penalty; default comes from rl/config.py")
    p.add_argument("--stay-penalty", type=float, default=None,
                   help="override RewardConfig.stay_penalty; default comes from rl/config.py")
    p.add_argument("--dot-not-eaten-penalty", type=float, default=None,
                   help="override RewardConfig.dot_not_eaten_penalty; charged "
                        "once per decision after the grace period")
    p.add_argument("--dot-not-eaten-grace", type=int, default=None,
                   help="override RewardConfig.dot_not_eaten_grace; number of "
                        "decisions without a dot before the penalty starts")
    p.add_argument("--level-clear-bonus", type=float, default=None,
                   help="override RewardConfig.level_clear_bonus; default comes from rl/config.py")
    p.add_argument("--ghost-score-scale", type=float, default=None,
                   help="scale only the ghost-combo part of the score reward; "
                        "1.0 keeps the old behaviour, 0.5 halves it")
    p.add_argument("--danger-penalty", type=float, default=None,
                   help="per-decision penalty when a dangerous ghost is within "
                        "--danger-radius tiles (scaled by closeness)")
    p.add_argument("--danger-radius", type=int, default=None,
                   help="danger shaping radius in tiles; 0 disables it")
    p.add_argument("--dots-removed-min", type=int, default=0,
                   help="endgame curriculum: minimum ordinary dots pre-removed "
                        "at each training episode reset")
    p.add_argument("--dots-removed-max", type=int, default=0,
                   help="endgame curriculum: maximum ordinary dots pre-removed "
                        "at each training episode reset; evaluation always uses 0")

    # DQN hyperparameters.
    p.add_argument("--warmup", type=int, default=5_000,
                   help="random decisions before DQN starts learning")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-final", type=float, default=None,
                   help="linearly decay Adam lr from --lr to this value over "
                        "the run; omit for a constant lr")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--epsilon-end", type=float, default=0.02)
    p.add_argument("--epsilon-decay", type=int, default=500_000)
    p.add_argument("--target-update", type=int, default=2_000)
    p.add_argument("--replay-size", type=int, default=100_000)
    p.add_argument("--train-every", type=int, default=2,
                   help="one gradient step per N decisions; use 1 for the "
                        "strongest (but slowest) long run")
    p.add_argument("--width", type=int, default=64,
                   help="DQN convolution width; 64 is the recommended value")

    # Bookkeeping.
    p.add_argument("--out", type=str, default=str(ROOT / "runs" / "train_dqn"))
    p.add_argument("--save-every", type=int, default=500_000,
                   help="checkpoint interval in decisions; 0 disables")
    p.add_argument("--log-every", type=int, default=50_000,
                   help="console progress interval in decisions")
    p.add_argument("--no-report", action="store_true",
                   help="skip metrics.csv, stats.json, summary.md and the PNG "
                        "training plots at the end of the run")

    # GIF recording.  Frames are streamed to ffmpeg, never accumulated in RAM.
    p.add_argument("--gif-frames", type=int, default=100_000,
                   help="number of training frames to capture; 0 disables the GIF")
    p.add_argument("--gif-fps", type=int, default=30,
                   help="playback rate of the recorded frames")
    p.add_argument("--gif-width", type=int, default=224,
                   help="GIF width; 224 is the game's native width")
    p.add_argument("--gif-height", type=int, default=288,
                   help="GIF height; 288 is the game's native height")
    p.add_argument("--gif-colors", type=int, default=64,
                   help="palette size; lower = smaller file")
    p.add_argument("--keep-mp4", action="store_true",
                   help="keep the intermediate MP4 after making the GIF")

    # Final live game.
    p.add_argument("--no-watch", action="store_true",
                   help="run the final game headlessly instead of opening a window")
    p.add_argument("--headless", action="store_true",
                   help="force SDL's dummy video driver")
    p.add_argument("--live-seed", type=int, default=12_345)
    p.add_argument("--live-scale", type=int, default=3,
                   help="window scale for the final game (native scale is 4)")
    p.add_argument("--live-fps", type=int, default=60)
    p.add_argument("--live-max-frames", type=int, default=0,
                   help="safety cap for the live game; 0 means play until death")

    # Evaluation is useful as a final number, but it costs extra episodes.
    p.add_argument("--eval-episodes", type=int, default=10,
                   help="greedy evaluation episodes per generation; 0 disables")
    p.add_argument("--eval-seed0", type=int, default=10_000)
    p.add_argument("--eval-level-cap", type=int, default=-1,
                   help="level cap used for evaluation: -1 follows the current "
                        "training cap, 0 means no cap (full game)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def set_maze_scale(scale: int) -> None:
    """Change map.py's drawing globals without editing map.py.

    map.py stores the window scale in module-level variables.  The game logic
    never reads them, so changing them is safe: only pixel positions change.
    Training records at scale 1 (224x288); the live window uses a larger
    scale so the same drawing code fills the screen.
    """
    scale = max(1, int(scale))
    maze.winScale = scale
    maze.gridSize = 8 * scale
    maze.offset_y = 24 * scale
    maze.offset_x = 0
    maze.t = 2 * scale
    maze.r = 2 * scale


def draw_game(
    surface: pg.Surface,
    game: Game,
    anim_t: int,
    hud_font: pg.font.Font,
    small_font: pg.font.Font,
    ready_font: pg.font.Font,
    bottom_text: str = "",
) -> None:
    """Render one complete frame using the original drawing functions.

    This mirrors game.py's draw() but writes to an arbitrary Surface, which
    lets the training loop capture frames without opening a window.
    """
    surface.fill(maze.bgColor)
    maze.build_walls(surface, maze.walls)
    maze.draw_pellets(surface, anim_t)

    # Fruit is drawn as a simple red circle, just like game.py.
    if game.fruit:
        x, y = maze.cellPos(13.5, 17)
        pg.draw.circle(surface, "red", (int(x), int(y)),
                       int(0.7 * maze.gridSize))

    # Death animation: ghosts vanish and Pac-Man's wedge shrinks.
    if game.dying:
        mouth = int(180 * (1 - game.freeze / 90))
        if mouth < 175:
            game.pacman.draw(surface, anim_t, mouth)
    else:
        popup = game.popup
        if not popup:
            game.pacman.draw(surface, anim_t)

        for ghost in game.ghosts:
            # While a ghost-eaten score popup is shown, hide that ghost.
            if popup and ghost.mode == "eaten" and ghost.tile == popup[0]:
                continue
            ghost.draw(surface, anim_t)

        if popup:
            x, y = maze.cellPos(*popup[0])
            text = small_font.render(str(popup[1]), False, "cyan")
            surface.blit(text, text.get_rect(center=(int(x), int(y))))
        elif game.freeze > 0 and not game.game_over:
            # "READY!" pause at the start of a round.
            text = ready_font.render("READY!", False, "yellow")
            surface.blit(text, text.get_rect(center=maze.cellPos(13.5, 17)))

    # Top HUD.
    hud = (f"score {game.pacman.score}  lives {game.pacman.lives}  "
           f"level {game.level}")
    surface.blit(hud_font.render(hud, False, "yellow"),
                 (6 * maze.winScale, 3 * maze.winScale))

    # Optional bottom line (training progress or live status).
    if bottom_text:
        text = small_font.render(bottom_text, False, "white")
        surface.blit(
            text,
            (6 * maze.winScale,
             surface.get_height() - text.get_height() - 3 * maze.winScale),
        )

    # Game-over banner.  The training loop captures the terminal frame before
    # resetting, so this also appears in the GIF when an episode ends.
    if getattr(game, "game_over", False):
        text = ready_font.render("GAME OVER", False, "red")
        surface.blit(text, text.get_rect(center=maze.cellPos(13.5, 17)))


# ---------------------------------------------------------------------------
# Streaming GIF recorder
# ---------------------------------------------------------------------------
class FFmpegGIFRecorder:
    """Pipe raw RGB frames into ffmpeg, then turn the MP4 into a GIF.

    Why not Pillow directly?  Keeping 100,000 full frames in a Python list
    would need several gigabytes of RAM.  ffmpeg consumes each frame as soon
    as it is written, so memory use stays constant.  The two-pass palette
    conversion also gives a much smaller and better-looking GIF than
    Pillow's default quantisation.
    """

    def __init__(
        self,
        out_dir: Path,
        width: int,
        height: int,
        fps: int,
        colors: int,
        keep_mp4: bool,
    ):
        self.out_dir = out_dir
        self.width = int(width)
        self.height = int(height)
        self.fps = max(1, int(fps))
        self.colors = max(2, min(256, int(colors)))
        self.keep_mp4 = keep_mp4

        # yuv420p (the MP4 pixel format) requires even dimensions.
        if self.width % 2:
            self.width -= 1
        if self.height % 2:
            self.height -= 1
        if self.width < 2 or self.height < 2:
            raise RuntimeError("GIF dimensions are too small")

        self.ffmpeg = shutil.which("ffmpeg")
        if not self.ffmpeg:
            raise RuntimeError(
                "ffmpeg was not found on PATH. Install it or pass "
                "--gif-frames 0."
            )

        self.mp4_path = out_dir / "training.mp4"
        self.gif_path = out_dir / "training.gif"
        self.palette_path = out_dir / "training_palette.png"
        self.log_path = out_dir / "ffmpeg_encode.log"
        self.count = 0
        self.enabled = True
        self.abort_reason = ""
        self.proc: subprocess.Popen | None = None
        self.log = None

        cmd = [
            self.ffmpeg, "-hide_banner", "-loglevel", "warning", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "26",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(self.mp4_path),
        ]
        try:
            self.log = open(self.log_path, "wb")
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self.log,
            )
        except Exception:
            if self.log is not None:
                self.log.close()
            raise

        print(f"recording GIF frames: {self.width}x{self.height} @ {self.fps} fps")

    def write(self, surface: pg.Surface) -> None:
        """Downscale one rendered frame and send its RGB bytes to ffmpeg."""
        if not self.enabled or self.proc is None or self.proc.stdin is None:
            return

        if surface.get_size() == (self.width, self.height):
            frame = surface
        else:
            # Nearest-neighbour keeps the pixel-art edges crisp.
            frame = pg.transform.scale(surface, (self.width, self.height))

        raw = pg.image.tobytes(frame, "RGB")
        try:
            self.proc.stdin.write(raw)
        except (BrokenPipeError, OSError) as exc:
            self.enabled = False
            self.abort_reason = f"ffmpeg pipe failed: {exc}"
            print(f"warning: {self.abort_reason}")
            return

        self.count += 1

        # If the disk is nearly full, stop recording rather than corrupting
        # the run or filling the filesystem.  Training itself continues.
        if self.count % 2_000 == 0:
            free = shutil.disk_usage(self.out_dir).free
            if free < 250 * 1024**2:
                self.enabled = False
                self.abort_reason = (
                    f"low disk space ({free / 1024**2:.0f} MiB free)"
                )
                print(f"warning: {self.abort_reason}; stopping GIF capture")

    def close(self) -> None:
        """Close ffmpeg's stdin so it writes the MP4 trailer."""
        if self.proc is None:
            return
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
        except OSError:
            pass
        returncode = self.proc.wait()
        if self.log is not None:
            self.log.close()
            self.log = None
        self.proc = None
        if returncode != 0:
            print(f"warning: ffmpeg MP4 encoder exited with code {returncode}")
        if self.abort_reason:
            print(f"GIF capture note: {self.abort_reason}")

    def finish(self) -> Path | None:
        """Convert the MP4 to a palette GIF and optionally delete the MP4."""
        self.close()

        if self.count == 0:
            print("no GIF frames were captured")
            return None
        if not self.mp4_path.exists() or self.mp4_path.stat().st_size == 0:
            print("MP4 was not created; cannot build the GIF")
            return None

        free = shutil.disk_usage(self.out_dir).free
        if free < 350 * 1024**2:
            print(
                f"warning: only {free / 1024**2:.0f} MiB free; "
                f"keeping {self.mp4_path.name} and skipping GIF conversion"
            )
            return self.mp4_path

        # Do the palette pass and the apply pass separately.  A single
        # palettegen+paletteuse filter graph makes paletteuse buffer every
        # input frame while it waits for the palette, which can exhaust RAM
        # on a 100,000-frame clip.  Two passes keep memory bounded: the
        # palette is a small PNG and paletteuse streams the GIF.
        palette_cmd = [
            self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(self.mp4_path),
            "-vf", f"fps={self.fps},"
                   f"palettegen=max_colors={self.colors}:stats_mode=diff",
            str(self.palette_path),
        ]
        gif_cmd = [
            self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(self.mp4_path),
            "-i", str(self.palette_path),
            "-lavfi", f"fps={self.fps},paletteuse=dither=bayer:bayer_scale=3",
            "-loop", "0",
            "-an",
            str(self.gif_path),
        ]

        print(f"converting {self.count} frames to GIF (this can take a while)...")
        started = time.perf_counter()
        try:
            subprocess.run(palette_cmd, check=True)
            subprocess.run(gif_cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"warning: GIF conversion failed ({exc}); MP4 kept")
            return self.mp4_path

        self.palette_path.unlink(missing_ok=True)

        gif_mb = self.gif_path.stat().st_size / 1024**2
        mp4_mb = self.mp4_path.stat().st_size / 1024**2
        print(
            f"GIF saved: {self.gif_path} "
            f"({self.count} frames, {gif_mb:.1f} MiB, "
            f"{time.perf_counter() - started:.1f}s); "
            f"MP4 {mp4_mb:.1f} MiB"
        )

        if not self.keep_mp4:
            self.mp4_path.unlink(missing_ok=True)
            print("removed the intermediate MP4 (pass --keep-mp4 to keep it)")
            return self.gif_path
        return self.gif_path


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------
def build_agent(args: argparse.Namespace) -> DQNAgent:
    """Create a fresh DQN.  `--resume` loads a checkpoint later."""
    return DQNAgent(
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
        use_ghost_dir=not args.no_ghost_dir,
    )


def checkpoint_path(out: Path) -> Path:
    return out / "dqn.pt"


def detect_last_generation(out: Path) -> int:
    """Return the last completed generation recorded in an output directory.

    `progress.json` is written after every generation, so it is the most
    reliable source for an interrupted run.  `run.json` is the fallback for
    runs created before progress tracking was added.
    """
    for name in ("progress.json", "run.json"):
        path = out / name
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        last = data.get("last_generation")
        if isinstance(last, int) and last > 0:
            return last
        generations = data.get("generation_results") or []
        numbers = [
            int(item.get("generation", 0))
            for item in generations
            if isinstance(item, dict) and item.get("generation")
        ]
        if numbers:
            return max(numbers)
        # run.json is only written after training finishes, so its
        # `generations` value is safe to use as a fallback.
        if name == "run.json" and data.get("generations"):
            return int(data["generations"])
    return 0


def load_previous_generation_rows(out: Path) -> list[dict]:
    """Recover earlier generation rows when continuing an existing run.

    A completed run has ``run.json`` with the full table.  An interrupted run
    only has ``eval_genN.json`` files, one per finished generation.  Reading
    them keeps the final report and the six-panel showcase continuous across
    a resume instead of showing only the new generations.
    """
    rows: list[dict] = []
    run_path = out / "run.json"
    if run_path.exists():
        try:
            data = json.loads(run_path.read_text())
            for item in data.get("generation_results") or []:
                if isinstance(item, dict) and item.get("result"):
                    rows.append(item)
        except (OSError, json.JSONDecodeError):
            pass

    known = {
        int(item.get("generation", 0))
        for item in rows
        if isinstance(item, dict)
    }
    for path in sorted(out.glob("eval_gen*.json")):
        name = path.stem
        try:
            generation = int(name.removeprefix("eval_gen"))
        except ValueError:
            continue
        if generation in known:
            continue
        try:
            result = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        decisions = 0
        checkpoint = out / f"dqn_gen{generation}.pt"
        if checkpoint.exists():
            try:
                import torch

                state = torch.load(checkpoint, map_location="cpu")
                decisions = int(state.get("decisions", 0))
            except Exception:
                pass
        rows.append({
            "generation": generation,
            "source": "resume" if generation == 0 else "generation",
            "training_level_cap": None,
            "evaluation_level_cap": None,
            "decisions": decisions,
            "episode": 0,
            "epsilon": 0.0,
            "updates": 0,
            "loss_mean": 0.0,
            "wall_seconds": 0.0,
            "result": result,
        })
        known.add(generation)

    rows.sort(key=lambda item: int(item.get("generation", 0)))
    return rows


def _format_eval(result: dict | None) -> str:
    if not result:
        return "not evaluated"
    return (
        f"score {result.get('score_mean', 0.0):.1f}, "
        f"pellets {result.get('dots_mean', 0.0):.1f}, "
        f"clears {result.get('clears_mean', 0.0):.2f}"
    )


def greedy_policy(agent: DQNAgent):
    """Return a greedy DQN action function for evaluation/live play."""
    return lambda obs: agent.act(obs, greedy=True)


def set_learning_rate(agent: DQNAgent, lr: float) -> None:
    for group in agent.optimizer.param_groups:
        group["lr"] = lr


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace) -> DQNAgent:
    """Run the from-scratch DQN loop and return the selected agent.

    The loop is split into generations so a long run produces useful
    checkpoints and evaluation numbers instead of one opaque final blob.
    The network and replay buffer survive between generations; only the
    level cap (curriculum) changes.
    """
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    resume_requested = args.resume or args.resume_from is not None

    start_generation = args.start_generation
    if start_generation is None:
        if resume_requested:
            last_generation = detect_last_generation(out)
            if last_generation <= 0:
                start_generation = 1
                print("resume: no completed generation found; starting at 1")
            elif last_generation < args.generations:
                start_generation = last_generation + 1
                print(
                    f"resume: generation {last_generation} is complete; "
                    f"continuing at generation {start_generation}"
                )
            else:
                # The requested schedule is already complete.  Re-running the
                # final generation is the least surprising way to spend more
                # decisions at the final (usually uncapped) level.
                start_generation = args.generations
                print(
                    f"resume: {last_generation} generation(s) already complete; "
                    f"re-running generation {start_generation}"
                )
        else:
            start_generation = 1

    remaining_generations = args.generations - start_generation + 1
    if args.total_decisions < remaining_generations:
        raise SystemExit("--total-decisions must be at least the number of "
                         "remaining generations")
    base_decisions, extra = divmod(args.total_decisions, remaining_generations)

    print(f"output directory: {out}")
    print(
        f"DQN | decisions: {args.total_decisions:,} | "
        f"generations: {start_generation}..{args.generations} | "
        f"seed: {args.seed}"
    )
    free_gb = shutil.disk_usage(out).free / 1024**3
    print(f"free disk: {free_gb:.2f} GiB")

    # Reward defaults live in rl/config.py.  The overrides make one run
    # reproducible without editing shared code.
    reward_cfg = RewardConfig()
    if args.score_scale is not None:
        reward_cfg.score_scale = args.score_scale
    if args.death_penalty is not None:
        reward_cfg.death_penalty = args.death_penalty
    if args.stay_penalty is not None:
        reward_cfg.stay_penalty = args.stay_penalty
    if args.dot_not_eaten_penalty is not None:
        reward_cfg.dot_not_eaten_penalty = args.dot_not_eaten_penalty
    if args.dot_not_eaten_grace is not None:
        reward_cfg.dot_not_eaten_grace = args.dot_not_eaten_grace
    if args.level_clear_bonus is not None:
        reward_cfg.level_clear_bonus = args.level_clear_bonus
    if args.ghost_score_scale is not None:
        reward_cfg.ghost_score_scale = args.ghost_score_scale
    if args.danger_penalty is not None:
        reward_cfg.danger_penalty = args.danger_penalty
    if args.danger_radius is not None:
        reward_cfg.danger_radius = args.danger_radius
    print(
        "reward: "
        f"score_scale={reward_cfg.score_scale} "
        f"death_penalty={reward_cfg.death_penalty} "
        f"stay_penalty={reward_cfg.stay_penalty} "
        f"dot_not_eaten_penalty={reward_cfg.dot_not_eaten_penalty} "
        f"dot_not_eaten_grace={reward_cfg.dot_not_eaten_grace} "
        f"level_clear_bonus={reward_cfg.level_clear_bonus} "
        f"ghost_score_scale={reward_cfg.ghost_score_scale} "
        f"danger_penalty={reward_cfg.danger_penalty} "
        f"danger_radius={reward_cfg.danger_radius}"
    )
    print(
        f"endgame curriculum: dots removed at reset "
        f"{args.dots_removed_min}..{args.dots_removed_max} | "
        f"ghost-direction input: {not args.no_ghost_dir}"
    )

    def training_cap(generation: int) -> int | None:
        if args.curriculum:
            return generation if generation <= args.curriculum_max_level else None
        return args.level_cap if args.level_cap > 0 else None

    def evaluation_cap(train_cap: int | None) -> int | None:
        if args.eval_level_cap == -1:
            return train_cap
        return args.eval_level_cap if args.eval_level_cap > 0 else None

    first_cap = training_cap(start_generation)
    env = PacmanEnv(EnvConfig(
        frame_skip=args.frame_skip,
        max_frames=args.max_frames,
        level_cap=first_cap,
        seed=args.seed,
        dots_removed_min=args.dots_removed_min,
        dots_removed_max=args.dots_removed_max,
        reward=reward_cfg,
    ))
    agent = build_agent(args)
    rng = random.Random(args.seed)

    resumed = False
    resume_path = None
    if resume_requested:
        resume_path = (
            Path(args.resume_from).expanduser().resolve()
            if args.resume_from else checkpoint_path(out)
        )
        if resume_path.exists():
            agent.load(resume_path, allow_partial=args.partial_load)
            resumed = True
            print(
                f"resumed from {resume_path} "
                f"(epsilon {agent.epsilon:.3f}, decisions {agent.decisions:,}, "
                f"ghost_dir={agent.use_ghost_dir}, width={agent.width}"
                f"{', partial transfer' if agent.loaded_partial else ''})"
            )
            if args.reset_epsilon:
                agent.decisions = 0
                agent.epsilon = agent.epsilon_start
                print(
                    "reset exploration schedule: "
                    f"epsilon={agent.epsilon:.2f}, decay restarts from here"
                )
        else:
            print(f"warning: resume requested but {resume_path} does not exist; "
                  "starting fresh")

    # If we resumed, do not throw away the learned policy with a new random
    # warmup.  For very short smoke runs, shrink warmup so the run is not
    # 100% random actions.
    warmup = 0 if resumed else min(
        args.warmup, max(0, args.total_decisions // 4)
    )
    if not resumed and warmup != args.warmup:
        print(f"note: warmup reduced to {warmup} for this short run")
    print(f"warmup: {warmup} | train_every: {args.train_every} | width: {args.width}")
    print(f"device: {agent.device}")

    # How often to capture a frame so the GIF has approximately --gif-frames
    # images across the whole run.
    record_every = 0
    if args.gif_frames > 0:
        record_every = max(1, args.total_decisions // args.gif_frames)
        expected = args.total_decisions // record_every
        print(f"GIF capture: every {record_every} decisions "
              f"(~{expected:,} frames)")
        if expected < args.gif_frames:
            print(f"note: only {expected:,} frames can be captured from "
                  f"{args.total_decisions:,} decisions")

    recorder = None
    if record_every:
        try:
            recorder = FFmpegGIFRecorder(
                out,
                args.gif_width,
                args.gif_height,
                args.gif_fps,
                args.gif_colors,
                args.keep_mp4,
            )
        except RuntimeError as exc:
            print(f"GIF recording disabled: {exc}")

    # Training always draws at native resolution, regardless of the final
    # window scale.
    set_maze_scale(1)
    screen = pg.Surface((224, 288))
    hud_font = pg.font.SysFont("consolas", 9, bold=True)
    small_font = pg.font.SysFont("consolas", 8)
    ready_font = pg.font.SysFont("consolas", 14, bold=True)

    episodes_path = out / "episodes.csv"
    episode_fields = [
        "generation", "episode", "decision", "return", "score", "dots",
        "dots_removed", "frames", "deaths", "clears", "epsilon", "loss_mean",
    ]
    episodes_file = open(episodes_path, "w", newline="")
    episodes_writer = csv.DictWriter(episodes_file, fieldnames=episode_fields)
    episodes_writer.writeheader()

    obs = env.reset()
    episode = 0
    episode_return = 0.0
    episode_losses: list[float] = []
    recent_losses = deque(maxlen=20)
    recent_scores = deque(maxlen=20)
    anim_t = 0
    global_decision = 0
    started = time.perf_counter()
    interrupted = False
    gif_path = None
    generation_rows: list[dict] = load_previous_generation_rows(out)
    best_metric = None
    best_generation = None
    best_eval = None
    best_source = None

    def run_eval(cap: int | None) -> dict:
        """Evaluate with the endgame curriculum disabled and cap restored."""
        old_cap = env.cfg.level_cap
        old_min = env.cfg.dots_removed_min
        old_max = env.cfg.dots_removed_max
        env.cfg.level_cap = cap
        env.cfg.dots_removed_min = 0
        env.cfg.dots_removed_max = 0
        try:
            return evaluate(
                env,
                greedy_policy(agent),
                episodes=args.eval_episodes,
                seed0=args.eval_seed0,
            )
        finally:
            env.cfg.level_cap = old_cap
            env.cfg.dots_removed_min = old_min
            env.cfg.dots_removed_max = old_max

    # A resumed checkpoint is generation 0 in the comparison table.  Without
    # this baseline the best-checkpoint rule can happily return a continuation
    # that is worse than the policy it started from.
    existing_best = None
    best_path = out / "best.json"
    if best_path.exists():
        try:
            existing_best = json.loads(best_path.read_text())
        except (OSError, json.JSONDecodeError):
            existing_best = None

    if (resumed and args.eval_episodes > 0
            and existing_best and existing_best.get("result")):
        # Continuing an existing run: keep the saved best instead of
        # overwriting it with the latest (possibly worse) checkpoint.
        best_generation = int(existing_best.get("generation", 0))
        best_eval = existing_best["result"]
        best_metric = (
            best_eval["clears_mean"],
            best_eval["dots_mean"],
            best_eval["score_mean"],
        )
        best_source = existing_best.get("source") or (
            "resume" if best_generation == 0 else "generation"
        )
        print(
            f"keeping existing best checkpoint from generation "
            f"{best_generation} ({_format_eval(best_eval)})",
            flush=True,
        )
    elif resumed and args.eval_episodes > 0:
        baseline_cap = evaluation_cap(first_cap)
        print(
            f"evaluating resumed checkpoint as generation 0 "
            f"({args.eval_episodes} greedy episodes, "
            f"level cap {'none' if baseline_cap is None else baseline_cap}, "
            f"curriculum disabled)...",
            flush=True,
        )
        baseline_result = run_eval(baseline_cap)
        (out / "eval_gen0.json").write_text(
            json.dumps(baseline_result, indent=2) + "\n"
        )
        print(
            f"gen 0 (resume) eval: "
            f"score {baseline_result['score_mean']:.1f} +/- "
            f"{baseline_result['score_std']:.1f} | "
            f"dots {baseline_result['dots_mean']:.1f} | "
            f"frames {baseline_result['frames_mean']:.0f} | "
            f"deaths {baseline_result['deaths_mean']:.2f} | "
            f"clears {baseline_result['clears_mean']:.2f}",
            flush=True,
        )
        baseline_metric = (
            baseline_result["clears_mean"],
            baseline_result["dots_mean"],
            baseline_result["score_mean"],
        )
        best_metric = baseline_metric
        best_generation = 0
        best_eval = baseline_result
        best_source = "resume"
        agent.save(out / "best.pt")
        (out / "best.json").write_text(json.dumps({
            "generation": 0,
            "source": "resume",
            "training_level_cap": first_cap,
            "evaluation_level_cap": baseline_cap,
            "decisions": 0,
            "result": baseline_result,
        }, indent=2) + "\n")
        generation_rows.append({
            "generation": 0,
            "source": "resume",
            "training_level_cap": first_cap,
            "evaluation_level_cap": baseline_cap,
            "decisions": 0,
            "episode": 0,
            "epsilon": round(agent.epsilon, 4),
            "updates": agent.updates,
            "loss_mean": 0.0,
            "wall_seconds": round(time.perf_counter() - started, 2),
            "result": baseline_result,
        })
        obs = env.reset()

    try:
        for generation in range(start_generation, args.generations + 1):
            generation_decisions = base_decisions + (
                1 if generation <= extra else 0
            )
            if generation_decisions <= 0:
                continue

            cap = training_cap(generation)
            env.cfg.level_cap = cap
            obs = env.reset()
            cap_text = "no cap" if cap is None else str(cap)
            print(
                f"\n--- generation {generation}/{args.generations} | "
                f"level cap {cap_text} | {generation_decisions:,} decisions ---",
                flush=True,
            )

            for _ in range(generation_decisions):
                global_decision += 1

                # ---- choose an action ---------------------------------
                if global_decision <= warmup:
                    action = rng.randrange(4)
                else:
                    action = agent.act(obs)

                # ---- interact with the real game ----------------------
                next_obs, reward, done, info = env.step(action)
                terminal = done and not info["truncated"]

                # ---- learn --------------------------------------------
                agent.remember(obs, action, reward, next_obs, terminal)
                agent.update_epsilon()
                if global_decision > warmup and global_decision % args.train_every == 0:
                    loss = agent.learn()
                    if loss is not None:
                        episode_losses.append(loss)

                if args.lr_final is not None:
                    frac = min(1.0, global_decision / args.total_decisions)
                    set_learning_rate(
                        agent, args.lr + frac * (args.lr_final - args.lr)
                    )

                obs = next_obs
                episode_return += reward
                anim_t += args.frame_skip

                # ---- capture a training frame -------------------------
                # Capture *before* a possible reset so terminal frames
                # (death or level clear) actually appear in the GIF.
                if recorder is not None and global_decision % record_every == 0:
                    bottom = (
                        f"g {generation}/{args.generations} "
                        f"d {global_decision:,} ep {episode} "
                        f"eps {agent.epsilon:.2f}"
                    )
                    draw_game(screen, env.game, anim_t, hud_font, small_font,
                              ready_font, bottom)
                    recorder.write(screen)

                # ---- episode bookkeeping ------------------------------
                if done:
                    episode += 1
                    levels_cleared = info["level"] - 1
                    dots = info.get(
                        "dots_eaten",
                        levels_cleared * N_PELLETS
                        + (N_PELLETS - info["dots_left"]),
                    )
                    loss_mean = (float(np.mean(episode_losses))
                                 if episode_losses else 0.0)
                    row = {
                        "generation": generation,
                        "episode": episode,
                        "decision": global_decision,
                        "return": round(episode_return, 4),
                        "score": info["score"],
                        "dots": dots,
                        "dots_removed": info.get("dots_removed", 0),
                        "frames": info["frames"],
                        "deaths": 3 - info["lives"],
                        "clears": levels_cleared,
                        "epsilon": round(agent.epsilon, 4),
                        "loss_mean": round(loss_mean, 5),
                    }
                    episodes_writer.writerow(row)
                    episodes_file.flush()
                    recent_scores.append(info["score"])
                    if episode_losses:
                        recent_losses.append(loss_mean)

                    obs = env.reset()
                    episode_return = 0.0
                    episode_losses = []

                # ---- progress + checkpoint ---------------------------
                if global_decision % args.log_every == 0:
                    elapsed = time.perf_counter() - started
                    mean_score = (float(np.mean(recent_scores))
                                  if recent_scores else 0.0)
                    lr = agent.optimizer.param_groups[0]["lr"]
                    print(
                        f"decision {global_decision:>9,} | gen {generation} | "
                        f"episode {episode:>6,} | eps {agent.epsilon:.3f} | "
                        f"lr {lr:.1e} | updates {agent.updates:>8,} | "
                        f"recent score {mean_score:7.1f} | "
                        f"{elapsed:7.1f}s",
                        flush=True,
                    )

                if args.save_every and global_decision % args.save_every == 0:
                    agent.save(checkpoint_path(out))
                    print(
                        f"checkpoint saved at decision {global_decision:,}",
                        flush=True,
                    )

            # ---- end of generation ------------------------------------
            agent.save(checkpoint_path(out))
            gen_checkpoint = out / f"dqn_gen{generation}.pt"
            agent.save(gen_checkpoint)
            print(
                f"generation {generation} checkpoint saved: {gen_checkpoint}",
                flush=True,
            )

            generation_result = None
            eval_cap = None
            if args.eval_episodes > 0:
                eval_cap = evaluation_cap(cap)
                print(
                    f"evaluating generation {generation} "
                    f"({args.eval_episodes} greedy episodes, "
                    f"level cap {'none' if eval_cap is None else eval_cap}, "
                    f"curriculum disabled)...",
                    flush=True,
                )
                generation_result = run_eval(eval_cap)
                eval_path = out / f"eval_gen{generation}.json"
                eval_path.write_text(json.dumps(generation_result, indent=2) + "\n")
                print(
                    f"gen {generation} eval: "
                    f"score {generation_result['score_mean']:.1f} +/- "
                    f"{generation_result['score_std']:.1f} | "
                    f"dots {generation_result['dots_mean']:.1f} | "
                    f"frames {generation_result['frames_mean']:.0f} | "
                    f"deaths {generation_result['deaths_mean']:.2f} | "
                    f"clears {generation_result['clears_mean']:.2f}",
                    flush=True,
                )

                metric = (
                    generation_result["clears_mean"],
                    generation_result["dots_mean"],
                    generation_result["score_mean"],
                )
                generation_rows.append({
                    "generation": generation,
                    "training_level_cap": cap,
                    "evaluation_level_cap": eval_cap,
                    "decisions": global_decision,
                    "episode": episode,
                    "epsilon": round(agent.epsilon, 4),
                    "updates": agent.updates,
                    "loss_mean": (
                        round(float(np.mean(recent_losses)), 5)
                        if recent_losses else 0.0
                    ),
                    "wall_seconds": round(time.perf_counter() - started, 2),
                    "result": generation_result,
                })
                if best_metric is None or metric > best_metric:
                    best_metric = metric
                    best_generation = generation
                    best_eval = generation_result
                    best_source = "generation"
                    agent.save(out / "best.pt")
                    (out / "best.json").write_text(json.dumps({
                        "generation": generation,
                        "source": "generation",
                        "training_level_cap": cap,
                        "evaluation_level_cap": eval_cap,
                        "decisions": global_decision,
                        "result": generation_result,
                    }, indent=2) + "\n")
                    print(
                        f"new best checkpoint: generation {generation} "
                        f"(score {generation_result['score_mean']:.1f}, "
                        f"dots {generation_result['dots_mean']:.1f}, "
                        f"clears {generation_result['clears_mean']:.2f})",
                        flush=True,
                    )

                # Evaluation resets the shared game; start the next
                # generation from a fresh episode.
                obs = env.reset()

            # Let --resume find the next generation even if the process is
            # interrupted after this point.
            progress = {
                "last_generation": generation,
                "decisions": global_decision,
                "training_level_cap": cap,
                "evaluation_level_cap": eval_cap,
                "result": generation_result,
                "best_generation": best_generation,
                "best_source": best_source,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            (out / "progress.json").write_text(
                json.dumps(progress, indent=2) + "\n"
            )

    except KeyboardInterrupt:
        interrupted = True
        print("\ninterrupted: saving the current agent and partial GIF...")

    finally:
        episodes_file.close()
        agent.save(checkpoint_path(out))
        print(f"checkpoint saved: {checkpoint_path(out)}")

        if recorder is not None:
            # Encoding is not part of the experiment.  A missing codec or a
            # full disk should not throw away the checkpoint we just saved.
            try:
                result = recorder.finish()
                gif_path = str(result) if result is not None else None
            except Exception as exc:
                print(f"warning: GIF finalisation failed: {exc}")

    elapsed = time.perf_counter() - started

    # Use the best evaluated checkpoint for the live game unless the user
    # explicitly asks for the final one.  Training is already over, so this
    # does not affect the optimisation.
    selected_generation = args.generations
    if not args.no_best and best_generation is not None:
        best_path = out / "best.pt"
        if best_path.exists():
            agent.load(best_path)
            selected_generation = best_generation
            label = (
                "the resumed checkpoint (generation 0)"
                if best_generation == 0
                else f"generation {best_generation}"
            )
            print(f"selected best checkpoint from {label} for final "
                  f"evaluation/live play")

    # Final greedy evaluation on the selected agent.  It is separate from the
    # live game: these episodes are seeded and comparable across runs.
    final_result = None
    if args.eval_episodes > 0:
        final_train_cap = training_cap(selected_generation)
        final_cap = evaluation_cap(final_train_cap)
        print(
            f"evaluating final agent ({args.eval_episodes} greedy episodes, "
            f"level cap {'none' if final_cap is None else final_cap}, "
            f"curriculum disabled)...",
            flush=True,
        )
        final_result = run_eval(final_cap)
        (out / "eval.json").write_text(json.dumps(final_result, indent=2) + "\n")
        print(
            f"final eval: score {final_result['score_mean']:.1f} +/- "
            f"{final_result['score_std']:.1f} | "
            f"dots {final_result['dots_mean']:.1f} | "
            f"frames {final_result['frames_mean']:.0f} | "
            f"deaths {final_result['deaths_mean']:.2f} | "
            f"clears {final_result['clears_mean']:.2f}",
            flush=True,
        )

    # A repeated generation (or a new baseline appended to an old table) must
    # appear once, with the newest row winning.
    by_generation: dict[int, dict] = {}
    for row in generation_rows:
        by_generation[int(row.get("generation", 0))] = row
    generation_rows = [
        by_generation[number] for number in sorted(by_generation)
    ]

    run_info = {
        "algorithm": "dqn",
        "total_decisions": args.total_decisions,
        "completed_decisions": global_decision,
        "generations": args.generations,
        "start_generation": start_generation,
        "curriculum": args.curriculum,
        "curriculum_max_level": args.curriculum_max_level,
        "interrupted": interrupted,
        "seed": args.seed,
        "frame_skip": args.frame_skip,
        "max_frames": args.max_frames,
        "level_cap": first_cap,
        "score_scale": reward_cfg.score_scale,
        "death_penalty": reward_cfg.death_penalty,
        "stay_penalty": reward_cfg.stay_penalty,
        "dot_not_eaten_penalty": reward_cfg.dot_not_eaten_penalty,
        "dot_not_eaten_grace": reward_cfg.dot_not_eaten_grace,
        "level_clear_bonus": reward_cfg.level_clear_bonus,
        "ghost_score_scale": reward_cfg.ghost_score_scale,
        "danger_penalty": reward_cfg.danger_penalty,
        "danger_radius": reward_cfg.danger_radius,
        "dots_removed_min": args.dots_removed_min,
        "dots_removed_max": args.dots_removed_max,
        "use_ghost_dir": agent.use_ghost_dir,
        "partial_load": args.partial_load,
        "width": args.width,
        "warmup": warmup,
        "train_every": args.train_every,
        "lr": args.lr,
        "lr_final": args.lr_final,
        "epsilon_end": args.epsilon_end,
        "epsilon_decay": args.epsilon_decay,
        "replay_size": args.replay_size,
        "wall_seconds": round(elapsed, 2),
        "resumed": resumed,
        "resumed_from": str(resume_path) if resumed else None,
        "generation_results": generation_rows,
        "best_generation": best_generation,
        "best_source": best_source,
        "selected_generation": selected_generation,
        "best_eval": best_eval,
        "final_eval": final_result,
        "gif_frames": recorder.count if recorder is not None else 0,
        "gif_path": gif_path,
        "checkpoint": str(checkpoint_path(out)),
        "best_checkpoint": (
            str(out / "best.pt") if best_generation is not None else None
        ),
    }
    if gif_path and Path(gif_path).suffix.lower() == ".gif":
        run_info["gif_size_mb"] = round(
            Path(gif_path).stat().st_size / 1024**2, 1
        )
    (out / "run.json").write_text(json.dumps(run_info, indent=2) + "\n")
    print(f"training finished in {elapsed:.1f}s")

    if not args.no_report:
        try:
            from rl.report import build_report

            report = build_report(out)
            print("training report:")
            for name, path in report.items():
                if path:
                    print(f"  {name}: {path}")
        except Exception as exc:
            print(f"warning: could not build the training report: {exc}")

    return agent


# ---------------------------------------------------------------------------
# Final live game: play until all lives are gone
# ---------------------------------------------------------------------------
def live_play(agent, args: argparse.Namespace) -> None:
    """Run a fresh game with fx=True and stop only on game over."""
    show_window = (
        not args.no_watch
        and not _HEADLESS_FLAG
        and not _NO_DISPLAY
    )
    if not show_window and not args.no_watch:
        print("no display detected: running the final game headlessly")

    # fx=True restores READY!, death pauses and score popups.  The game also
    # handles level progression itself, so it keeps playing into level 2+.
    set_maze_scale(args.live_scale if show_window else 1)
    game = Game(fx=True)
    game.reset(seed=args.live_seed)
    obs_builder = ObservationBuilder()

    screen = None
    clock = None
    hud_font = small_font = ready_font = None
    if show_window:
        pg.display.set_caption("PacmanRL - trained agent (plays until death)")
        screen = pg.display.set_mode(
            (224 * maze.winScale, 288 * maze.winScale)
        )
        clock = pg.time.Clock()
        hud_font = pg.font.SysFont("consolas", max(9, 5 * maze.winScale),
                                    bold=True)
        small_font = pg.font.SysFont("consolas", max(8, 4 * maze.winScale))
        ready_font = pg.font.SysFont("consolas", max(14, 9 * maze.winScale),
                                     bold=True)
        print("starting live game; close the window or wait for game over")

    action = 0
    frame = 0
    anim_t = 0
    running = True

    while running:
        if show_window:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False

        # Ask for a new direction every frame_skip game frames, exactly like
        # training.  During READY/death freezes update() does not move
        # anything, but we still keep a valid requested direction.
        if frame % args.frame_skip == 0:
            obs = obs_builder.build(game)
            action = agent.act(obs, greedy=True)

        game.pacman.set_dir(ACTIONS[action])
        game.update()
        if game.freeze == 0:
            anim_t += 1

        if show_window:
            draw_game(
                screen, game, anim_t, hud_font, small_font, ready_font,
                bottom_text=f"greedy DQN | frame {frame}",
            )
            pg.display.flip()
            clock.tick(args.live_fps)

        if game.game_over:
            break
        frame += 1
        if args.live_max_frames and frame >= args.live_max_frames:
            print(f"live frame cap reached ({args.live_max_frames})")
            break

    print(
        f"live game over: score {game.pacman.score}, "
        f"level {game.level}, frames {frame}"
    )

    # Give the player a moment to read the final frame before the window
    # closes.  Keep pumping events so the close button still works.
    if show_window and running:
        end = time.time() + 4.0
        while time.time() < end:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return
            clock.tick(60)


def main() -> None:
    args = parse_args()
    validate_args(args)
    pg.init()
    try:
        agent = train(args)
        live_play(agent, args)
    finally:
        pg.quit()


if __name__ == "__main__":
    main()
