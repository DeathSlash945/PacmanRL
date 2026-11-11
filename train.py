#!/usr/bin/env python3
"""Train Pac-Man RL from scratch, save a training GIF, then watch it play.

This is the single-file entry point for the project.  It deliberately uses
the existing `rl` package instead of duplicating the environment or agents:

    python train.py                          # DQN, 1M decisions, 100k-frame GIF
    python train.py --agent q --total-decisions 3000000
    python train.py --gif-frames 0 --no-watch  # training only
    python train.py --headless --no-watch       # server / smoke test

What happens, in order
----------------------
1. A brand-new agent is created (no checkpoint is loaded).
2. It trains in the real game.  Every Nth decision we render one frame and
   stream the raw RGB pixels into ffmpeg, so RAM does not grow with the
   number of recorded frames.
3. The checkpoint, episode log and evaluation JSON are written to --out.
4. ffmpeg converts the temporary MP4 into an optimised palette GIF.
5. A fresh `Game(fx=True)` is started and the trained agent plays greedily
   until all three lives are gone.  It may continue into level 2 if it
   clears level 1.

The original game files are never modified.  The script only calls the
public API used by the rest of the project: `Game.update()`,
`Pacman.set_dir()`, and read-only attributes.
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
from rl.agents.qlearning import QLearningAgent
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
    if args.eval_episodes < 0:
        raise SystemExit("--eval-episodes cannot be negative")
    if args.gif_frames < 0:
        raise SystemExit("--gif-frames cannot be negative")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Pac-Man RL from scratch, record a GIF, then play."
    )

    # Which algorithm / how long.
    p.add_argument("--agent", choices=("dqn", "q"), default="dqn",
                   help="dqn (default, best score) or q (tabular baseline)")
    p.add_argument("--total-decisions", type=int, default=1_000_000,
                   help="environment steps; one step = --frame-skip game frames")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None,
                   help="torch device for DQN, e.g. cuda or cpu; auto by default")

    # Environment settings.  These match the experiment configuration used
    # in REPORT.md so this script reproduces the same task.
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--max-frames", type=int, default=18_000)
    p.add_argument("--level-cap", type=int, default=1,
                   help="stop episodes after this level; 0 means no cap")
    p.add_argument("--score-scale", type=float, default=None,
                   help="override RewardConfig.score_scale; default comes from rl/config.py")
    p.add_argument("--death-penalty", type=float, default=None,
                   help="override RewardConfig.death_penalty; default comes from rl/config.py")

    # DQN hyperparameters.
    p.add_argument("--warmup", type=int, default=5_000,
                   help="random decisions before DQN starts learning")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=int, default=200_000)
    p.add_argument("--target-update", type=int, default=2_000)
    p.add_argument("--replay-size", type=int, default=100_000)
    p.add_argument("--train-every", type=int, default=2,
                   help="one gradient step per N decisions")
    p.add_argument("--width", type=int, default=32,
                   help="DQN convolution width; 32 was used for the final runs")

    # Tabular Q-learning hyperparameters.
    p.add_argument("--alpha", type=float, default=0.10,
                   help="Q-learning step size (ignored for DQN)")

    # Bookkeeping.
    p.add_argument("--out", type=str, default=str(ROOT / "runs" / "train_demo"))
    p.add_argument("--save-every", type=int, default=500_000,
                   help="checkpoint interval in decisions; 0 disables")
    p.add_argument("--log-every", type=int, default=50_000,
                   help="console progress interval in decisions")

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
                   help="greedy evaluation episodes after training; 0 disables")
    p.add_argument("--eval-seed0", type=int, default=10_000)

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
def build_agent(args: argparse.Namespace):
    """Create a fresh agent.  No checkpoint is loaded."""
    if args.agent == "q":
        return QLearningAgent(
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            seed=args.seed,
        )

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
    )


def checkpoint_path(args: argparse.Namespace, out: Path) -> Path:
    return out / ("q.pkl" if args.agent == "q" else "dqn.pt")


def greedy_policy(agent, args: argparse.Namespace, env: PacmanEnv):
    """Return an action function for evaluation.

    DQN consumes the observation dict.  Q-learning is tabular and reads its
    hand-designed state directly from the live game, matching training.
    """
    def act(obs):
        if args.agent == "q":
            return agent.act(agent.state_from_game(env.game), greedy=True)
        return agent.act(obs, greedy=True)

    return act


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace):
    """Run the full from-scratch training loop and return the trained agent."""
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"output directory: {out}")
    print(f"agent: {args.agent} | decisions: {args.total_decisions:,} | seed: {args.seed}")
    free_gb = shutil.disk_usage(out).free / 1024**3
    print(f"free disk: {free_gb:.2f} GiB")

    # level_cap <= 0 means "no cap": episodes only end on death or timeout.
    level_cap = args.level_cap if args.level_cap > 0 else None

    # The reward defaults live in rl/config.py.  The optional overrides make
    # it possible to reproduce an older experiment without editing that file.
    reward_cfg = RewardConfig()
    if args.score_scale is not None:
        reward_cfg.score_scale = args.score_scale
    if args.death_penalty is not None:
        reward_cfg.death_penalty = args.death_penalty
    print(f"reward: score_scale={reward_cfg.score_scale} "
          f"death_penalty={reward_cfg.death_penalty}")

    env = PacmanEnv(EnvConfig(
        frame_skip=args.frame_skip,
        max_frames=args.max_frames,
        level_cap=level_cap,
        seed=args.seed,
        reward=reward_cfg,
    ))
    agent = build_agent(args)
    rng = random.Random(args.seed)

    # Q-learning has no replay buffer, so it can learn from the first step.
    # For very short smoke tests, shrink DQN's warmup so the run is not 100%
    # random actions.
    warmup = 0
    if args.agent == "dqn":
        warmup = min(args.warmup, max(0, args.total_decisions // 4))
        if warmup != args.warmup:
            print(f"note: warmup reduced to {warmup} for this short run")
    print(f"warmup: {warmup} | train_every: {args.train_every}")

    # How often to capture a frame so the GIF has approximately --gif-frames
    # images.  For 1M decisions and 100k frames this records every 10th step.
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
        "episode", "decision", "return", "score", "dots", "frames",
        "deaths", "clears", "epsilon", "loss_mean",
    ]
    episodes_file = open(episodes_path, "w", newline="")
    episodes_writer = csv.DictWriter(episodes_file, fieldnames=episode_fields)
    episodes_writer.writeheader()

    obs = env.reset()
    episode = 0
    episode_return = 0.0
    episode_losses: list[float] = []
    recent_scores = deque(maxlen=20)
    anim_t = 0
    started = time.perf_counter()
    interrupted = False
    gif_path = None

    try:
        for decision in range(1, args.total_decisions + 1):
            # ---- choose an action -------------------------------------
            state = None
            if args.agent == "q":
                state = agent.state_from_game(env.game)
                action = agent.act(state)
            elif decision <= warmup:
                action = rng.randrange(4)
            else:
                action = agent.act(obs)

            # ---- interact with the real game --------------------------
            next_obs, reward, done, info = env.step(action)
            terminal = done and not info["truncated"]

            # ---- learn ------------------------------------------------
            if args.agent == "q":
                next_state = agent.state_from_game(env.game)
                agent.learn(state, action, reward, next_state, terminal)
            else:
                agent.remember(obs, action, reward, next_obs, terminal)
                agent.update_epsilon()
                if decision > warmup and decision % args.train_every == 0:
                    loss = agent.learn()
                    if loss is not None:
                        episode_losses.append(loss)

            obs = next_obs
            episode_return += reward
            anim_t += args.frame_skip

            # ---- capture a training frame -----------------------------
            # Capture *before* a possible reset so terminal frames (death or
            # level clear) actually appear in the GIF.
            if recorder is not None and decision % record_every == 0:
                bottom = (
                    f"d {decision:,} ep {episode} "
                    f"eps {agent.epsilon:.2f}"
                )
                draw_game(screen, env.game, anim_t, hud_font, small_font,
                          ready_font, bottom)
                recorder.write(screen)

            # ---- episode bookkeeping ----------------------------------
            if done:
                episode += 1
                levels_cleared = info["level"] - 1
                dots = (levels_cleared * N_PELLETS
                        + (N_PELLETS - info["dots_left"]))
                loss_mean = (float(np.mean(episode_losses))
                             if episode_losses else 0.0)
                row = {
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
                }
                episodes_writer.writerow(row)
                episodes_file.flush()
                recent_scores.append(info["score"])

                obs = env.reset()
                episode_return = 0.0
                episode_losses = []

            # ---- progress + checkpoint -------------------------------
            if decision % args.log_every == 0:
                elapsed = time.perf_counter() - started
                mean_score = (float(np.mean(recent_scores))
                              if recent_scores else 0.0)
                print(
                    f"decision {decision:>9,} | episode {episode:>6,} | "
                    f"eps {agent.epsilon:.3f} | "
                    f"updates {getattr(agent, 'updates', 0):>8,} | "
                    f"recent score {mean_score:7.1f} | {elapsed:7.1f}s",
                    flush=True,
                )

            if args.save_every and decision % args.save_every == 0:
                agent.save(checkpoint_path(args, out))
                print(f"checkpoint saved at decision {decision:,}", flush=True)

    except KeyboardInterrupt:
        interrupted = True
        print("\ninterrupted: saving the current agent and partial GIF...")

    finally:
        episodes_file.close()
        agent.save(checkpoint_path(args, out))
        print(f"checkpoint saved: {checkpoint_path(args, out)}")

        if recorder is not None:
            # Encoding is not part of the experiment.  A missing codec or a
            # full disk should not throw away the checkpoint we just saved.
            try:
                result = recorder.finish()
                gif_path = str(result) if result is not None else None
            except Exception as exc:
                print(f"warning: GIF finalisation failed: {exc}")

    elapsed = time.perf_counter() - started
    run_info = {
        "agent": args.agent,
        "total_decisions": args.total_decisions,
        "completed_decisions": decision if "decision" in locals() else 0,
        "interrupted": interrupted,
        "seed": args.seed,
        "frame_skip": args.frame_skip,
        "level_cap": level_cap,
        "score_scale": reward_cfg.score_scale,
        "death_penalty": reward_cfg.death_penalty,
        "width": args.width if args.agent == "dqn" else None,
        "warmup": warmup,
        "train_every": args.train_every,
        "wall_seconds": round(elapsed, 2),
        "gif_frames": recorder.count if recorder is not None else 0,
        "gif_path": gif_path,
        "checkpoint": str(checkpoint_path(args, out)),
    }
    (out / "run.json").write_text(json.dumps(run_info, indent=2) + "\n")
    print(f"training finished in {elapsed:.1f}s")

    # Final greedy evaluation.  It is separate from the live game: these
    # episodes are seeded and easy to compare against REPORT.md.
    if args.eval_episodes > 0:
        print(f"evaluating {args.eval_episodes} greedy episodes...")
        result = evaluate(
            env,
            greedy_policy(agent, args, env),
            episodes=args.eval_episodes,
            seed0=args.eval_seed0,
        )
        (out / "eval.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            f"eval: score {result['score_mean']:.1f} +/- {result['score_std']:.1f} | "
            f"dots {result['dots_mean']:.1f} | "
            f"frames {result['frames_mean']:.0f} | "
            f"deaths {result['deaths_mean']:.2f} | "
            f"clears {result['clears_mean']:.2f}"
        )

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
            if args.agent == "q":
                action = agent.act(agent.state_from_game(game), greedy=True)
            else:
                action = agent.act(obs, greedy=True)

        game.pacman.set_dir(ACTIONS[action])
        game.update()
        if game.freeze == 0:
            anim_t += 1

        if show_window:
            draw_game(
                screen, game, anim_t, hud_font, small_font, ready_font,
                bottom_text=f"greedy {args.agent} | frame {frame}",
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
