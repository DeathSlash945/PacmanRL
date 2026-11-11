# Reinforcement learning layer for PacmanRL

The original game files (`env.py`, `pacman.py`, `ghost.py`, `map.py`,
`levels.py`, `game.py`) are **not modified**.  Everything here is a new
`rl/` package that reads the live game state and drives Pac-Man through
`pacman.set_dir()` and `game.update()`.

## Files

| File | Purpose |
| --- | --- |
| `config.py` | actions, observation sizes, reward and episode settings |
| `obs.py` | turns a live `Game` into an 8x31x28 grid + 18-vector |
| `pacman_env.py` | Gym-style `reset()` / `step()` wrapper |
| `evaluate.py` | fixed-seed episode runner and metrics |
| `agents/qlearning.py` | tabular Q-learning with a compact state |
| `agents/dqn.py` | DQN: replay, target network, Double DQN |
| `networks.py` | CNN + scalar-vector Q-network |
| `buffer.py` | int8 replay buffer |
| `train_q.py` | Q-learning training loop |
| `train_dqn.py` | DQN training loop |
| `eval_agent.py` | evaluate a saved checkpoint |
| `plot.py` | learning-curve comparison plots |
| `aggregate.py` | multi-seed mean/std table and bar chart |
| `watch.py` | watch a trained agent in pygame |
| `record.py` | record a trained agent to an animated GIF |

The single-file entry point is one directory up:

| File | Purpose |
| --- | --- |
| `../train.py` | from-scratch DQN/Q-learning training, 100k-frame training GIF, checkpoint, fixed-seed evaluation, then a live game until death |

Run it from the project root:

```bash
# DQN, 1M decisions, 100,000-frame training GIF, live game at the end
/home/deathslash/venvs/mlasgn/bin/python train.py

# No display / server: still trains, records and evaluates, no window
/home/deathslash/venvs/mlasgn/bin/python train.py --headless --no-watch

# Fast smoke test
/home/deathslash/venvs/mlasgn/bin/python train.py --agent q \
  --total-decisions 2000 --gif-frames 200 --eval-episodes 2 --no-watch

# Reproduce the reward values used in REPORT.md, even if rl/config.py
# has since been tuned
/home/deathslash/venvs/mlasgn/bin/python train.py \
  --score-scale 0.01 --death-penalty -5
```

The GIF is a training timelapse: with the defaults it captures one frame
every 10 decisions, so 1M decisions produce exactly 100,000 frames.  Frames
are streamed to ffmpeg instead of being kept in RAM.  The palette pass and
the palette-application pass are separate, which keeps memory bounded for
a 100,000-frame clip.  The native 224x288 GIF is about 300 MiB; use
`--gif-width 140 --gif-height 180` for a smaller one.

## Observation

`obs = {"grid": float32[8,31,28], "vec": float32[18]}`.

Grid channels:

| Channel | Contents |
| --- | --- |
| 0 | wall (including the ghost door) |
| 1 | dot |
| 2 | power pellet |
| 3 | Pac-Man |
| 4 | Blinky |
| 5 | Pinky |
| 6 | Inky |
| 7 | Clyde |

Ghost channel values: `1` dangerous (scatter/chase), `-1` frightened
(edible), `0.5` harmless (house/leaving/eaten/entering), `0` absent.

Vector: Pac-Man direction one-hot (4), buffered direction one-hot (4),
fright timer, scatter/chase flag, pellets left, Cruise Elroy stage,
lives, level, moving flag, and counts of dangerous/frightened/harmless
ghosts.  This gives the network the timers and mode information that a
spatial grid cannot express.

## Actions and time

Four actions: `up`, `left`, `down`, `right`.  The action is a *requested*
direction; Pac-Man's existing buffered-turn logic applies it at the next
tile centre.

`frame_skip = 4`: one agent decision every four 60 Hz game frames, so
the agent acts 15 times per second.  Rewards are summed over the skipped
frames.  This is the standard Atari-style action repeat and keeps the
environment fast without giving the agent super-human reaction speed.

## Reward

| Event | Reward |
| --- | --- |
| dot | `+10 * 0.01 = +0.1` |
| power pellet | `+50 * 0.01 = +0.5` |
| ghost | `+200/400/800/1600 * 0.01` |
| fruit | score value `* 0.01` |
| death | `-5` |
| level clear | `+10` |
| per decision | `0` (available for tuning) |

The game never subtracts score on death, so the death penalty is added
explicitly.  `score_scale`, `death_penalty`, `step_penalty` and
`level_clear_bonus` live in `config.RewardConfig` and can be changed
without touching the game.

## Episodes

Some subtleties of this game matter for a correct wrapper:

- With `Game(fx=False)` there are no READY/death/eaten pauses, so every
  `update()` is exactly one game frame.
- A death is **not** visible through `pacman.alive` after `update()`:
  `collide()` sets `alive=False`, then `finish_death()` immediately
  respawns Pac-Man.  The wrapper therefore detects death by
  `pacman.lives` decreasing.
- After a death the game resets Pac-Man's direction to `left`, so the
  wrapper stops the current `frame_skip` immediately.  Otherwise the
  remaining skipped frames would use a stale action.
- `level_cap=1` ends the episode when level 1 is cleared; `level_cap=None`
  lets training continue into later levels.
- `max_frames` is an episode time limit (default 18,000 frames = 5
  minutes of game time).  This is a truncation, not a terminal state, and
  DQN still bootstraps from the last observation.

## Q-learning baseline

A full tabular Q-table is impossible: 244 pellets give `2**244` maze
configurations.  The baseline compresses the game to:

```
(col, row, 4-neighbour pellet bits, nearest-ghost sector,
 nearest-ghost distance bucket, fright bucket, current direction)
```

That is still millions of possible states, so the Q-table is a Python
dict and rows are allocated only for visited states.  Update:

```
Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
```

with `alpha=0.1`, `gamma=0.99`, epsilon decayed linearly from 1.0 to
0.05 over 200k decisions.  This is a fair "classic RL" baseline, not a
hand-coded pellet chaser.

## DQN

- Input: the grid + vector above.
- Network: 3 convolutions (32, 64, 64 channels, two stride-2 layers),
  concatenated with the 18-vector, then 512 hidden units and 4 Q-values.
- Replay buffer: 100k transitions, observations stored as int8 and
  converted to float on the GPU.
- Double DQN: the online net selects the next action, the target net
  evaluates it.
- Target network copied every 2,000 gradient steps.
- Huber loss, Adam (`lr=1e-4`), gradient clipping at 10.
- Epsilon-greedy: 1.0 -> 0.05 over 200k decisions.
- One gradient step every 2 decisions by default (`--train-every 2`).

## Commands

Use the existing CUDA environment (torch 2.14.1 + cu126, Python 3.10):

```bash
PY=/home/deathslash/venvs/mlasgn/bin/python

# verification
SDL_VIDEODRIVER=dummy $PY -m rl.smoke_test

# Q-learning
SDL_VIDEODRIVER=dummy $PY -m rl.train_q \
    --total-decisions 1000000 --eval-every 100000 --out runs/qlearning

# DQN
SDL_VIDEODRIVER=dummy $PY -m rl.train_dqn \
    --total-decisions 1000000 --width 32 --train-every 2 \
    --eval-every 50000 --out runs/dqn

# evaluate on 30 fixed seeds
SDL_VIDEODRIVER=dummy $PY -m rl.eval_agent --agent q \
    --checkpoint runs/qlearning/q.pkl --episodes 30
SDL_VIDEODRIVER=dummy $PY -m rl.eval_agent --agent dqn \
    --checkpoint runs/dqn/dqn.pt --width 32 --episodes 30

# plots
$PY -m rl.plot --q runs/qlearning/metrics.csv --dqn runs/dqn/metrics.csv
$PY -m rl.aggregate

# watch (needs a display)
$PY -m rl.watch --agent q --checkpoint runs/qlearning/q.pkl
$PY -m rl.watch --agent dqn --checkpoint runs/dqn/dqn.pt --width 32

# record a GIF (no display needed)
SDL_VIDEODRIVER=dummy $PY -m rl.record --agent dqn \
    --checkpoint runs/dqn_seed0/dqn.pt --width 32 \
    --frames 1800 --out runs/dqn_demo.gif
```

## Metrics

Evaluation is greedy (no epsilon) on seeds `10000..10000+N-1`.  Reported
per episode: return, score, pellets eaten, survival frames, deaths, and
levels cleared.  The comparison uses the same reward, actions, frame
skip, episode limit and evaluation seeds for both algorithms.

## Results

### 30-episode greedy evaluation, 3 training seeds

| Agent | Score | Pellets | Survival (frames) |
| --- | ---: | ---: | ---: |
| Random | 342 ± 0 | 34.2 ± 0 | 2057 ± 0 |
| Q-learning, 1M | 1528 ± 101 | 125.7 ± 12.1 | 3437 ± 277 |
| Q-learning, 3M | 1588 ± 105 | 129.8 ± 13.4 | 3690 ± 663 |
| DQN, 1M | TBD | TBD | TBD |

DQN seed 0 at 1M: 2481 score, 173.0 pellets, 3139 frames.  Seeds 1 and
2 are running; the final mean/std will replace the TBD row.

DQN is weaker for the first ~400k decisions, then improves sharply and
overtakes both Q-learning runs.  The full analysis is in `REPORT.md`.

## Known limitations

- The Q-learning state is a hand-designed compression; it cannot see
  the full maze, so its ceiling is lower than DQN's.
- DQN sees the full static maze, which makes level-specific overfitting
  easy.  Generalising across levels would need an egocentric crop or a
  recurrent policy.
- Only level 1 is trained initially.  `level_cap=None` enables later
  levels, but the first version does not use a curriculum.
- No prioritised replay, dueling heads, n-step returns or distributional
  RL yet; those are the natural next upgrades if DQN plateaus.
