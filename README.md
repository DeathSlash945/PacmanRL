# PacmanRL

A from-scratch, tile-based Pac-Man in Python and pygame-ce. Every piece of game state (positions, directions, ghost modes, targets, timers, level parameters) is a plain Python attribute, so the game can be played by hand, inspected live, or driven by a reinforcement learning agent.

![screenshot](docs/screenshot.png)

## Features

- Original arcade maze: 28 x 31 tiles, 244 pellets (240 dots and 4 power pellets), warp tunnel
- Ghost AI following the Pac-Man Dossier: per-ghost targeting, scatter/chase schedule, frightened and eaten modes, house exit with dot counters, Cruise Elroy
- Level progression driven by a lookup table (speeds, fright time, schedules, fruit, Elroy), like the original, capped at level 21+
- Fruit at 70 and 170 dots eaten, with level-dependent type and score
- Animations: Pac-Man mouth, ghost feet and eyes, power pellet blink, frightened flashing, death animation, READY! pause, ghost score popups
- Live debug panel showing all game, Pac-Man and ghost state
- Logic and rendering are fully separated: the game runs headless with no window
- Fixed timestep and seedable randomness, so runs are reproducible

## Quick start

```bash
git clone https://github.com/<your-username>/PacmanRL.git
cd PacmanRL
pip install pygame-ce
python game.py
```

Tested on Python 3.14 with pygame-ce. Install `pygame-ce`, not `pygame` (they conflict, uninstall `pygame` first if you have it).

## Controls

| Key | Action |
| --- | --- |
| Arrow keys or WASD | Set Pac-Man's next direction (buffered turn) |
| Close window | Quit |

Keyboard control is a debug option. Set `HUMAN=False` in `game.py` to ignore the keyboard so an agent can drive Pac-Man instead. With `HUMAN=True`, the game restarts two seconds after game over.

## Reinforcement learning

The game is wrapped as a Gym-style environment in `rl/`.  The single-file
entry point trains a Double/Dueling DQN from scratch, records a
100,000-frame training timelapse GIF, evaluates the agent, and then opens
a live game that keeps playing until all three lives are gone:

```bash
python train.py
# server / no display:
python train.py --headless --no-watch
# fast smoke test (not a useful policy):
python train.py --total-decisions 2000 --generations 2 \
  --curriculum --gif-frames 0 --eval-episodes 2 --no-watch
```

Outputs go to `runs/train_dqn/`: `dqn.pt`, `best.pt`, per-generation
checkpoints, `episodes.csv`, `eval.json`, `run.json`, `progress.json`,
`metrics.csv`, `summary.md`, `stats.json`, `training_showcase.png` and
`training.gif`.  The GIF is streamed through ffmpeg, so 100,000 frames do
not sit in RAM.  At the native 224x288 resolution the full GIF is about
300 MiB; use `--gif-width 140 --gif-height 180` for a smaller file, or
`--gif-frames 0` to skip recording.

The latest 10M-decision curriculum run (preserved in
`runs/train_dqn_curriculum/`) scores **2129.0 +/- 368.9** on 30 fixed seeds,
eats **171.7 of 244 pellets**, and still clears **0 levels**.  It is better
than the original 1M checkpoint (1981.7 +/- 368.6 on the same 30 seeds),
but it is not a clear.  The next run therefore starts from that best
checkpoint and adds three targeted changes: ghost-direction observations,
an endgame dot-removal curriculum, and reward shaping that makes ghost
chasing less dominant.

The report is written automatically at the end of every run:

- `summary.md` is the human-readable result table;
- `stats.json` is the same information for scripts;
- `training_showcase.png` is one presentation-ready six-panel figure:
  training score, pellets and survival with greedy-evaluation markers,
  training return, DQN loss, and greedy evaluation by generation.

The old separate `training_curves*.png` and `training_episodes.png` files
were removed.  They duplicated the same data, and the final-bars chart had
nothing useful to compare once Q-learning was dropped.

For a run created before the report existed, rebuild it with
`python -m rl.report --run runs/train_dqn`.  The same command also works
while a run is still training: if `run.json` is missing it reads
`progress.json`, `eval_gen*.json` and `episodes.csv` instead.

### Training longer, in generations

`train.py` supports a continuous multi-generation run.  The network and
replay buffer survive between generations; only the level cap and the
evaluation point change.  This is the recommended shape for an overnight
attempt at clearing the game:

```bash
PY=/home/deathslash/venvs/mlasgn/bin/python
$PY train.py \
  --resume-from runs/train_dqn_curriculum/best.pt --partial-load \
  --warmstart-buffer 100000 --warmstart-epsilon 0.05 \
  --generations 10 --total-decisions 10000000 \
  --curriculum --curriculum-max-level 4 \
  --max-frames 60000 --width 64 \
  --n-step 3 --train-every 2 --batch-size 256 --target-update 1000 \
  --gamma 0.995 --replay-size 150000 \
  --epsilon-start 0.1 --reset-epsilon \
  --epsilon-decay 3000000 --epsilon-end 0.01 \
  --lr 3e-5 --lr-final 1e-6 --eval-episodes 30 --eval-level-cap 0 \
  --ghost-score-scale 0.5 --danger-penalty 0.02 --danger-radius 5 \
  --death-penalty -20 --level-clear-bonus 50 \
  --stay-penalty -0.03 --step-penalty -0.005 \
  --dot-not-eaten-penalty -0.02 --dot-not-eaten-grace 45 \
  --dots-removed-min 0 --dots-removed-max 120 \
  --gif-frames 0 --out runs/train_dqn_finetune
```

This is a **fine-tuning** run, not a fresh from-scratch run.  The earlier
`runs/train_dqn_final` experiment reset epsilon to 1.0, changed gamma to
0.997 and used aggressive shaping; after 8M decisions its best generation
was still the 2129-point starting checkpoint (generation 8 reached only
1157.0 score and 101.7 pellets).  The recipe above keeps the old policy
alive with three changes: 100k warm-start transitions from the loaded
policy, a gentle epsilon reset to 0.1, and a lower Adam learning rate.
It also uses n-step returns (3), a larger batch with one update every two
decisions, and AMP, so the GPU does useful work and the run should finish
substantially faster than the previous 10M run.  A 20k-decision benchmark
on this machine measured about 189 decisions/s for the old
batch-128/every-decision/no-AMP loop and 261 decisions/s for the new
batch-256/every-two-decisions/bf16 loop, roughly 1.4x faster.  GIF capture
is off in the command, which removes another source of overhead.  `--gif-frames 0` skips
recording; the existing 100k-frame GIF remains in `runs/train_dqn`.  To
record a fresh demo after training, use
`python -m rl.record --agent dqn --checkpoint runs/train_dqn_finetune/best.pt
--out runs/train_dqn_finetune/demo.gif --frames 6000`.


If you want more than one candidate per wall-clock hour, launch two or
three independent fine-tunes in parallel.  They share the GPU and use
different CPU cores; each writes its own run directory, so you can compare
`eval.json` afterwards:

```bash
scripts/train_parallel.sh 3 10000000   # 3 candidates, 10M decisions each
```

The helper disables the GIF and live window for candidates, uses the same
fine-tuning recipe, and varies `ghost_score_scale`, `danger_penalty` and the
endgame dot range so the runs are not identical.

What the important flags mean:

| Flag | Effect |
| --- | --- |
| `--generations N` | Split the decision budget into N evaluation/checkpoint stages |
| `--resume` | Continue from `<out>/dqn.pt` in the same directory |
| `--resume-from PATH` | Continue from an explicit checkpoint, e.g. into a new output directory |
| `--partial-load` | Transfer compatible weights from an older architecture; new ghost-direction columns start at zero so the old policy is preserved exactly |
| `--start-generation G` | With `--resume`, continue at generation G; omit it to auto-detect the next generation |
| `--curriculum` | Generation 1 is capped at level 1, generation 2 at level 2, ... Every episode still starts at level 1; the cap only decides when it ends |
| `--curriculum-max-level K` | After level K, later generations play the full game |
| `--dots-removed-min/max` | Endgame curriculum: each training episode starts with a random number of ordinary dots already gone (power pellets stay) |
| `--danger-penalty`, `--danger-radius` | Small per-decision penalty for being close to a dangerous ghost; gives a gradient before the death penalty |
| `--step-penalty` | Per-decision movement cost; the default is -0.005, small enough that 20 moves still cost less than one dot |
| `--ghost-score-scale` | Scales only the ghost-combo score reward; 0.5 halves it, 0.35 makes risky chasing much less attractive |
| `--eval-level-cap 0` | Evaluate every generation on the full game, not just the training cap |
| `--warmstart-buffer N` | On resume, fill the replay buffer with N decisions from the loaded policy before any gradient step; the cheapest defence against catastrophic forgetting |
| `--warmstart-epsilon` | Exploration used only while filling that buffer (0.05 is enough) |
| `--epsilon-start` | Starting epsilon for a fresh run or `--reset-epsilon`; use 0.1 to fine-tune, not 1.0 |
| `--n-step N` | Store N-step returns (3 is a good compromise); gives sparse deaths and clears a shorter credit path |
| `--train-every N` + `--batch-size M` | Fewer, larger GPU updates.  2/256 has the same samples per decision as 1/128 but much better GPU utilisation |
| `--gamma 0.995` | Keep the checkpoint's horizon when fine-tuning; changing gamma changes every Q-target |
| `--lr`, `--lr-final` | Fine-tune with `3e-5 -> 1e-6`, not the fresh-run `1e-4` |
| `--epsilon-decay` | Spend longer exploring; important for a multi-million-decision run |
| `--reset-epsilon` | When resuming, restart exploration from `--epsilon-start` instead of keeping the checkpoint's low epsilon |
| `--no-amp` | Disable CUDA mixed precision (AMP is on by default and roughly 1.5x faster) |
| `--no-ghost-dir` | Disable the 16 ghost-direction inputs (they are on by default for new runs) |
| `--no-best` | Use the final checkpoint instead of the best evaluated one |
| `--no-live` | Skip the final live game; useful for background or parallel candidates |
| `--no-report` | Skip `metrics.csv`, `stats.json`, `summary.md` and `training_showcase.png` |

The script saves `best.pt` whenever a generation's fixed-seed evaluation
improves on `(level clears, pellets eaten, score)`, then uses that
checkpoint for the final evaluation and live game.  A resumed checkpoint is
evaluated first and entered as generation 0, so a continuation cannot
regress below the policy it started from.  On a later `--resume`, an
existing `best.json` is kept instead of being overwritten by the latest
(possibly worse) checkpoint, and the old `eval_genN.json` rows are merged
back into the table.  Older checkpoints are kept as `dqn_gen1.pt`,
`dqn_gen2.pt`, etc.  `--resume` restores the network, optimizer, epsilon and
counters from `dqn.pt`.  The replay buffer is not stored in the checkpoint,
so it refills from new experience after a resume.  Pass
`--warmstart-buffer 100000` to fill it from the loaded policy first; without
that, the first gradient steps can wash out a good policy.

### Continuing an existing run

There are two ways to continue:

- `--resume` loads `<out>/dqn.pt` and writes into the same directory.
  `train.py` reads `progress.json` (or `run.json` for older runs) and
  starts at the next unfinished generation, so the generation number does
  not have to be guessed.
- `--resume-from PATH` loads an explicit checkpoint and can write to a new
  output directory.  This preserves the old run's `episodes.csv`,
  `eval_gen*.json`, GIF and report while the continuation builds its own.

In both cases `--total-decisions` is the budget for the remaining
generations, not the original total.  To continue the existing level-1
checkpoint for another 5M decisions without overwriting the old results:

```bash
PY=/home/deathslash/venvs/mlasgn/bin/python
$PY train.py --resume-from runs/train_dqn/dqn.pt \
  --start-generation 1 --generations 5 \
  --total-decisions 5000000 --level-cap 1 --max-frames 18000 \
  --width 64 --train-every 2 --batch-size 256 --n-step 3 --gamma 0.995 \
  --warmstart-buffer 50000 --warmstart-epsilon 0.05 \
  --epsilon-start 0.1 --reset-epsilon --epsilon-decay 1000000 \
  --epsilon-end 0.02 --lr 3e-5 --lr-final 1e-6 \
  --eval-episodes 30 --eval-level-cap 1 \
  --gif-frames 0 --out runs/train_dqn_cont
```

`--reset-epsilon` now restarts exploration from `--epsilon-start` (0.1 in
this example), not from 1.0.  A full reset to 1.0 is still available with
`--epsilon-start 1.0`, but it should only be used for a genuine fresh
policy; on a fine-tune it usually destroys the checkpoint before the new
reward signal can take effect.  Remove `--reset-epsilon` entirely to keep
the checkpoint's epsilon and decay counter.  The 5-generation split gives one fixed-seed
evaluation and checkpoint per 1M decisions, so `training_showcase.png`
shows the continuation rather than a single evaluation point.  On a server
add `--headless --no-watch`.

If the long run above is interrupted, resume it in the same directory.
`train.py` detects the next unfinished generation and keeps the existing
generation 0 baseline:

```bash
$PY train.py --resume --generations 10 \
  --total-decisions 4000000 --curriculum --curriculum-max-level 4 \
  --max-frames 60000 --width 64 --n-step 3 --train-every 2 \
  --batch-size 256 --target-update 1000 --gamma 0.995 \
  --replay-size 150000 --warmstart-buffer 100000 --warmstart-epsilon 0.05 \
  --epsilon-start 0.1 --reset-epsilon --epsilon-decay 3000000 \
  --epsilon-end 0.01 --lr 3e-5 --lr-final 1e-6 --eval-episodes 30 \
  --eval-level-cap 0 --ghost-score-scale 0.5 --danger-penalty 0.02 \
  --danger-radius 5 --death-penalty -20 --level-clear-bonus 50 \
  --stay-penalty -0.03 --step-penalty -0.005 \
  --dot-not-eaten-penalty -0.02 --dot-not-eaten-grace 45 \
  --dots-removed-min 0 --dots-removed-max 120 \
  --out runs/train_dqn_finetune
```

Both continuation styles write a fresh `summary.md`, `stats.json`,
`metrics.csv` and `training_showcase.png` when they finish.  The old
`runs/train_dqn` report remains available for comparison.

### Why not Q-learning?

Q-learning was implemented and measured, then removed from the working
system.  It used a hand-designed local state (Pac-Man's tile, four nearby
pellet bits, the nearest ghost, a fright bucket and the direction).  That
cannot see the full maze or all four ghosts, and its Python-dict Q-table
scales poorly.  In the earlier trial it scored lower than DQN despite
being much cheaper to train.  The final project therefore ships one
working DQN path; the full reasons are in `REPORT.md`.

See `REPORT.md` for the design/results template and
`rl/WALKTHROUGH.md` for the line-by-line explanation.

## Project layout

| File | Purpose |
| --- | --- |
| `game.py` | Viewer: window, input, drawing, debug panel |
| `env.py` | `Game` class: all game logic, no drawing |
| `pacman.py` | `Pacman` class: movement, wall collision, speed, drawing |
| `ghost.py` | `Ghost` class: modes, targeting, movement, house walks, drawing |
| `levels.py` | Per-level parameter table and the `params(level)` lookup |
| `map.py` | Maze tile map, wall outlines, pellets, wall and tunnel checks |
| `train.py` | One-file DQN training, generations/curriculum, GIF, evaluation and live play |
| `rl/` | Environment wrapper, observations, DQN, evaluation and tools |

## Specs

| Item | Value |
| --- | --- |
| Maze | 28 x 31 tiles (original layout) |
| Tile size | 8 px at native resolution |
| Native resolution | 224 x 288 |
| Default window | 896 x 1152 game area (scale 4) plus a 340 px side panel |
| Update rate | Fixed 60 logic steps per second, no delta time |
| Position units | Tile units (floats). Integers are tile centers, `13.5` is the gap between two tiles |
| Tunnel | Row 14. Tiles at `x=-1` and `x=28` are virtual and wrap around |
| Speed units | Tiles per frame. 100% speed is `0.15625` |
| Pellets | 244 total |
| Lives | 3 |
| Pac-Man and ghost turns | Only at tile centers |
| Pac-Man vs ghost collision | Same tile |

Maze coordinates are `(col, row)` with `col` 0 to 27 and `row` 0 to 30.

## Parameters shown in the debug panel

**Game**

| Name | Meaning |
| --- | --- |
| `level` | Current level |
| `fruit` | `[name, points, frames_left]` while a fruit is on screen, else `None` |
| `fright` | Frames of frightened time left (0 means not frightened) |
| `phase` | Current global mode, `scatter` or `chase` |
| `dots_left` | Pellets remaining (dots and power pellets) |
| `elroy` | Blinky's Cruise Elroy stage: 0, 1 or 2 |
| `game_over` | True when all lives are lost (clearing a level starts the next one) |
| `freeze` | Frames left in the current pause (READY!, death, ghost eaten) |

**Pac-Man**

| Name | Meaning |
| --- | --- |
| `pos` | `(x, y)` in tile units |
| `tile` | Current tile `(col, row)` |
| `dir` | Current direction |
| `next` | Buffered direction, applied at the next tile center if open |
| `speed` | Tiles per frame right now |
| `moving` | False when blocked by a wall |
| `lives`, `score`, `alive` | Lives left, score, alive flag |

**Each ghost** (`blinky`, `pinky`, `inky`, `clyde`)

| Name | Meaning |
| --- | --- |
| `pos`, `tile` | Position in tile units and current tile |
| `dir` | Current direction |
| `mode` | `house`, `leaving`, `scatter`, `chase`, `frightened`, `eaten`, `entering` |
| `target` | Target tile used for steering |
| `speed` | Tiles per frame right now |

## Rules implemented

**Scoring:** dot 10, power pellet 50, ghosts 200 / 400 / 800 / 1600 per power pellet, fruit by level (100, 300, 500, 700, 1000, 2000, 3000, 5000).

**Targeting**

| Ghost | Chase target | Scatter corner |
| --- | --- | --- |
| Blinky | Pac-Man's tile | (25, -3) |
| Pinky | 4 tiles ahead of Pac-Man (with the original upward-facing overflow bug) | (2, -3) |
| Inky | Doubled vector from Blinky through the tile 2 ahead of Pac-Man (same bug) | (27, 31) |
| Clyde | Pac-Man's tile if farther than 8 tiles, else his scatter corner | (0, 31) |

**Steering:** at each tile center a ghost picks the exit whose neighbor tile is closest (straight-line distance) to its target. No reversing, ties broken in the order up, left, down, right, and no upward turns at `(12,11)`, `(15,11)`, `(12,23)`, `(15,23)` outside frightened mode. Ghosts reverse when the global mode switches.

**Frightened:** ghosts turn at random, move slowly, and flash near the end. The scatter/chase timer is paused. Eaten ghosts return to the house as eyes, then leave again.

**House exit:** Pinky leaves immediately, Inky and Clyde after a dot count that depends on the level (30 and 60 on level 1). A ghost is also released if no dot is eaten for 240 frames (180 from level 5).

**Eating stall:** Pac-Man pauses 1 frame per dot and 3 frames per power pellet, as in the original.

## Level parameters

Returned by `levels.params(level)`:

| Key | Meaning |
| --- | --- |
| `pac`, `pac_fr` | Pac-Man speed fraction, normal and during fright |
| `gh`, `gh_tun`, `gh_fr` | Ghost speed fraction: normal, tunnel, frightened |
| `elroy`, `el_sp` | Dots left for Elroy 1 (half for Elroy 2) and Blinky's speed fractions |
| `fright`, `flash` | Frightened seconds and number of end-of-fright flashes |
| `sched` | Scatter/chase schedule as `(frames, mode)` pairs |
| `fruit` | `(name, points)` for the level |
| `inky`, `clyde` | Dots needed before each leaves the house |
| `idle` | Frames without a dot before a forced release |

## Using the game as a library

`env.py` has no window or drawing, so it runs headless:

```python
from env import Game

g=Game(fx=False)        # fx=False skips READY!, death and ghost-eaten pauses
g.reset(seed=0)         # seeds Python's random, making runs reproducible
g.pacman.set_dir('left')
for _ in range(60):
    g.update()          # one frame
print(g.info())         # level, fruit, fright, phase, dots_left, elroy, game_over, freeze
print(g.pacman.state())
print([gh.state() for gh in g.ghosts])
```

| Member | Description |
| --- | --- |
| `Game(fx=True)` | Create a game. `fx=False` removes all pauses |
| `reset(seed=None)` | Start a new game at level 1 |
| `update()` | Advance one frame |
| `info()` | Game-level state dict (the panel's top block) |
| `pacman`, `ghosts` | The entity objects, with a `state()` method each |
| `level`, `fruit`, `fright`, `game_over`, `P` | Current game variables and level parameters |

Randomness is used in two places only: frightened ghost turns and the fruit's time on screen.

## Known limitations

- Fruit is drawn as a placeholder circle, no sprites yet
- No sound and no extra life at 10,000 points
- Cruise Elroy is not disabled after a death, and the global dot counter used after a death is not implemented
- Ghosts waiting in the house are not frightened and do not bounce
- Frightened turns use Python's `random`, not the original's pseudo-random sequence
- Wall and pellet state live at module level in `map.py`, so use one `Game` per process

## Roadmap

- Gymnasium wrapper (observation, action, reward)
- Optional DQN upgrades: prioritised replay and a distributional head
- Optional PPO/A2C comparison using the same environment and evaluation code

## Credits and disclaimer

Ghost behavior and timing follow the Pac-Man Dossier by Jamey Pittman. Pac-Man is a trademark of Bandai Namco. This is a non-commercial fan and educational project and contains no original game assets.

## License

Add a license before publishing.
