"""Watch a trained agent play in a real pygame window.

Examples:
    python -m rl.watch --agent q   --checkpoint runs/qlearning/q.pkl
    python -m rl.watch --agent dqn --checkpoint runs/dqn/dqn.pt --width 32
    python -m rl.watch --agent random

The viewer runs the game at 60 FPS and asks the agent for a new action
every `--decide-every` frames (default 4), matching training.  This is a
separate file on purpose: game.py stays the original human-playable
viewer and is never imported here.
"""

import argparse
import random

import pygame as pg

import map as maze
from .agents.dqn import DQNAgent
from .agents.qlearning import QLearningAgent
from .config import EnvConfig
from .pacman_env import PacmanEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=("q", "dqn", "random"), required=True)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--decide-every", type=int, default=4)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max-frames", type=int, default=0,
                   help="stop after N frames (useful for headless smoke tests)")
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
    pg.display.flip()


def main():
    args = parse_args()
    pg.init()
    pg.display.set_caption("PacmanRL - trained agent")
    screen = pg.display.set_mode((224 * maze.winScale, 288 * maze.winScale))
    clock = pg.time.Clock()
    font = pg.font.SysFont("consolas", 20, bold=True)

    env = PacmanEnv(EnvConfig(frame_skip=1, max_frames=10**9,
                              level_cap=1, seed=args.seed))

    if args.agent == "random":
        rng = random.Random(args.seed)
        policy = lambda _obs: rng.randrange(4)
    elif args.agent == "q":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for --agent q")
        agent = QLearningAgent.load(args.checkpoint)
        policy = lambda _obs: agent.act(agent.state_from_game(env.game), greedy=True)
    else:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for --agent dqn")
        agent = DQNAgent(replay_size=1, width=args.width, device=args.device)
        agent.load(args.checkpoint)
        policy = lambda obs: agent.act(obs, greedy=True)

    obs = env.reset(seed=args.seed)
    action = 0
    frame = 0
    episode = 1
    running = True

    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

        if frame % args.decide_every == 0:
            action = policy(obs)

        obs, reward, done, info = env.step(action)
        frame += 1
        draw(screen, env.game, frame, font)
        clock.tick(args.fps)

        if args.max_frames and frame >= args.max_frames:
            break

        if done:
            episode += 1
            if args.episodes and episode > args.episodes:
                break
            pg.time.delay(600)
            obs = env.reset(seed=args.seed + 1000 * episode)
            action = 0
            frame = 0

    pg.quit()


if __name__ == "__main__":
    main()
