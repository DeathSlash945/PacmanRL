"""Record a trained DQN agent to a GIF (no display needed).

Example:
    SDL_VIDEODRIVER=dummy /home/deathslash/venvs/mlasgn/bin/python \
        -m rl.record --agent dqn --checkpoint runs/dqn/best.pt \
        --frames 1800 --out runs/dqn_demo.gif

Frames are rendered with the same drawing functions as the game, then
downscaled to the native 224x288 resolution and packed into an animated
GIF with Pillow.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import argparse
import random

import pygame as pg
from PIL import Image

import map as maze
from .agents.dqn import DQNAgent
from .config import EnvConfig
from .pacman_env import PacmanEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=("dqn", "random"), required=True)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frames", type=int, default=900)
    p.add_argument("--decide-every", type=int, default=4)
    p.add_argument("--every", type=int, default=2,
                   help="save every Nth game frame")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--level-cap", type=int, default=1,
                   help="stop after this level; 0 records the full game")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default="runs/demo.gif")
    return p.parse_args()


def draw(screen, game, frame, font):
    screen.fill(maze.bgColor)
    maze.build_walls(screen, maze.walls)
    maze.draw_pellets(screen, frame)

    if game.fruit:
        x, y = maze.cellPos(13.5, 17)
        pg.draw.circle(screen, "red", (int(x), int(y)), int(0.7 * maze.gridSize))

    game.pacman.draw(screen, frame)
    for ghost in game.ghosts:
        ghost.draw(screen, frame)

    hud = f"score {game.pacman.score}   lives {game.pacman.lives}   level {game.level}"
    screen.blit(font.render(hud, False, "yellow"), (10, 6))


def main():
    args = parse_args()
    pg.init()
    screen = pg.display.set_mode((224 * maze.winScale, 288 * maze.winScale))
    font = pg.font.SysFont("consolas", 20, bold=True)

    level_cap = args.level_cap if args.level_cap > 0 else None
    env = PacmanEnv(EnvConfig(frame_skip=1, max_frames=10**9,
                              level_cap=level_cap, seed=args.seed))

    if args.agent == "random":
        rng = random.Random(args.seed)
        policy = lambda _obs: rng.randrange(4)
    else:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for --agent dqn")
        agent = DQNAgent(replay_size=1, width=args.width, device=args.device)
        agent.load(args.checkpoint)
        policy = lambda obs: agent.act(obs, greedy=True)

    obs = env.reset(seed=args.seed)
    frames = []
    action = 0

    for frame in range(args.frames):
        if frame % args.decide_every == 0:
            action = policy(obs)
        obs, reward, done, info = env.step(action)
        draw(screen, env.game, frame, font)

        if frame % args.every == 0:
            small = pg.transform.scale(screen, (224, 288))
            array = pg.surfarray.array3d(small).swapaxes(0, 1)
            frames.append(Image.fromarray(array))

        if done:
            break

    if not frames:
        raise SystemExit("no frames recorded")

    frames[0].save(
        args.out,
        save_all=True,
        append_images=frames[1:],
        duration=max(1, 1000 // args.fps),
        loop=0,
        optimize=True,
    )
    print(f"saved {args.out}: {len(frames)} frames, score {env.game.pacman.score}")


if __name__ == "__main__":
    main()
