# Reinforcement learning layer for PacmanRL

The original game files (`env.py`, `pacman.py`, `ghost.py`, `map.py`,
`levels.py`, `game.py`) are **not modified**.  Everything here is a new
`rl/` package that reads the live game state and drives Pac-Man through
`pacman.set_dir()` and `game.update()`.

## Algorithm note

The final system uses **DQN only**.  Q-learning was tried and discarded:

- its hand-designed local state cannot see the full maze or all ghosts;
- a Python-dict Q-table scales poorly to millions of states;
- it scored lower than DQN in the earlier experiments, despite being
  much cheaper to train.

See `../REPORT.md` for the full explanation.  The Q-learning code and
command-line branches have been removed so there is one working path.

## Files

| File | Purpose |
| --- | --- |
| `config.py` | actions, observation sizes, reward and episode settings |
| `obs.py` | turns a live `Game` into an 8x31x28 grid + 18-vector + 16 ghost-direction bits |
| `pacman_env.py` | Gym-style `reset()` / `step()` wrapper; stay/no-dot/danger rewards and endgame dots |
| `evaluate.py` | fixed-seed episode runner and metrics |
| `agents/dqn.py` | Double DQN with replay, target network and optional CUDA AMP |
| `networks.py` | convolutional dueling Q-network |
| `buffer.py` | int8 replay buffer + n-step accumulator |
| `train_dqn.py` | low-level DQN training loop with CSV logs |
| `eval_agent.py` | evaluate a saved checkpoint |
| `report.py` | build `metrics.csv`, `stats.json`, `summary.md` and the showcase |
| `plot.py` | single-page six-panel training/evaluation showcase |
| `aggregate.py` | multi-seed mean/std table and bar chart |
| `watch.py` | watch a trained agent in pygame |
| `record.py` | record a trained agent to an animated GIF |
| `smoke_test.py` | shapes, determinism, reward, stay/no-dot penalties, benchmark |
| `run_experiments.sh` | three-seed DQN sweep + final evaluations |

The single-file entry point is one directory up:

| File | Purpose |
| --- | --- |
| `../train.py` | from-scratch DQN training, generations/curriculum, 100k-frame GIF, checkpoint selection, fixed-seed evaluation, then a live game until death |

Run it from the project root:

```bash
PY=/home/deathslash/venvs/mlasgn/bin/python

# DQN, 1M decisions, 100,000-frame training GIF, live game at the end
$PY train.py

# No display / server: still trains, records and evaluates, no window
$PY train.py --headless --no-watch

# Fast smoke test (not a useful policy)
$PY train.py --total-decisions 2000 --generations 2 --curriculum \
  --gif-frames 0 --eval-episodes 2 --no-watch
```

The GIF is a training timelapse: with the defaults it captures one frame
every 10 decisions, so 1M decisions produce exactly 100,000 frames.  Frames
are streamed to ffmpeg instead of being kept in RAM.  The palette pass and
the palette-application pass are separate, which keeps memory bounded for
a 100,000-frame clip.  The native 224x288 GIF is about 300 MiB; use
`--gif-width 140 --gif-height 180` for a smaller one.

## Observation

`obs = {"grid": float32[8,31,28], "vec": float32[18],
"ghost_dir": float32[16]}`.

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

`ghost_dir` is four one-hot directions for each ghost, in the fixed order
Blinky, Pinky, Inky, Clyde.  Position alone cannot tell the network whether
a ghost is about to enter Pac-Man's corridor; direction can.  New agents
use this branch by default.  Old checkpoints load without it, and
`--partial-load` transfers the old weights into the new architecture while
zeroing the new input columns so the starting policy is unchanged.

## Actions and time

Four actions: `up`, `left`, `down`, `right`.  The action is a *requested*
direction; Pac-Man's existing buffered-turn logic applies it at the next
tile centre.

`frame_skip = 4`: one agent decision every four 60 Hz game frames, so
the agent acts 15 times per second.  Rewards are summed over the skipped
frames.  This is the standard Atari-style action repeat and keeps the
environment fast without giving the agent super-human reaction speed.

## Reward

Defaults in `config.RewardConfig`:

| Event | Reward |
| --- | --- |
| dot | `+10 * 0.02 = +0.2` |
| power pellet | `+50 * 0.02 = +1.0` |
| ghost | `+200/400/800/1600 * 0.02 * ghost_score_scale` (`scale=0.5` by default) |
| fruit | score value `* 0.02` |
| death | `-15` |
| level clear | `+20` |
| stays still for one decision | `-0.02` |
| no dot for 45 decisions | `-0.02` per decision until the next dot |
| near a dangerous ghost | up to `-0.02` per decision within 5 tiles |
| per decision | `-0.005` (`step_penalty`) |

The game never subtracts score on death, so the death penalty is added
explicitly.  The per-decision step penalty is `-0.005`: a dot is +0.2, so
normal travel remains profitable, but idling is never free.  The stay
penalty is separate from the step penalty: it fires only when Pac-Man's
position is unchanged over the whole decision, so it stops corner-camping
without punishing normal movement or a legitimate pellet pause.

The no-dot penalty is separate progress pressure.  It starts after
`dot_not_eaten_grace=45` decisions without a pellet (3 seconds at the
default frame skip) and then charges `dot_not_eaten_penalty=-0.02` per
decision until the next dot.  It is not charged on terminal frames and
resets when a dot or power pellet is eaten.  This stops the agent from
surviving forever in an already-cleared corridor instead of finishing the
remaining pellets.

`ghost_score_scale` changes only the 200/400/800/1600 ghost-combo points,
not dots, power pellets or fruit.  It exists because a single power pellet
can otherwise be worth +60 reward, four times a death, which teaches the
agent to farm ghost chains instead of running away.

`danger_penalty`/`danger_radius` add a small per-decision cost when a
scatter/chase ghost is within the radius.  The cost is scaled linearly by
closeness and skipped on terminal frames.  It is a shaping signal, not a
replacement for the death penalty.

`dots_removed_min/max` implement the endgame curriculum.  Each training
episode can start with a random number of ordinary dots already removed;
power pellets are kept.  Evaluation always sets both to zero.  `info`
includes `dots_removed`, `dots_eaten`, `ghost_score`, `danger` and
`danger_penalty` so the new signals can be audited.

## Episodes

Some subtleties of this game matter for a correct wrapper:

- With `Game(fx=False)` there are no READY/death/eaten pauses, so every
  `update()` is exactly one game frame.
- A death is **not** visible through `pacman.alive` after `update()`:
  `collide()` sets `alive=False`, then `finish_death()` immediately
  respawns Pac-Man.  The wrapper therefore detects death by
  `pacman.lives` decreasing.
- After a death the game resets Pac-Man's direction to `left`, so the
  wrapper stops the current `frame_skip` immediately.
- `level_cap=1` ends the episode when level 1 is cleared; `level_cap=None`
  lets training continue into later levels.
- `max_frames` is an episode time limit (default 18,000 frames = 5
  minutes of game time).  This is a truncation, not a terminal state, and
  DQN still bootstraps from the last observation.

## DQN

- Input: the grid + vector + optional ghost-direction vector above.
- Network: 3 convolutions (32, `width`, `width`; two stride-2 layers),
  concatenated with the 18-vector (and the 64-unit ghost-direction branch
  when enabled), then a 512-unit trunk and a **dueling** head (state value
  + action advantages).
- Replay buffer: 100k transitions by default, observations stored as int8
  and converted to float on the GPU.  Ghost-direction bits add only 1.6 MB
  per 100k transitions.
- Double DQN: the online net selects the next action, the target net
  evaluates it.
- Target network copied every 2,000 gradient steps.
- Huber loss, Adam (`lr=1e-4`), gradient clipping at 10.
- Epsilon-greedy: 1.0 -> 0.02 over 500k decisions by default.
- CUDA mixed precision is on by default (`bf16` on the RTX 4060, `fp16`
  with a gradient scaler on older cards); `--no-amp` disables it.
- `--n-step N` stores N-step returns.  The default is 1 (classic one-step
  DQN); 3 is recommended for fine-tuning because deaths and clears are
  sparse.
- One gradient step every 2 decisions by default.  For a fast long run use
  `--train-every 2 --batch-size 256`, which keeps 128 samples per decision
  but makes each GPU update larger.  `--train-every 1` is the slowest,
  most sample-efficient option.
- `--warmstart-buffer 100000` fills the replay buffer from the loaded
  checkpoint before any gradient step.  This is the recommended way to
  continue a good policy without catastrophic forgetting.

## Generations and curriculum

`train.py` can split a long run into generations without resetting the
network or replay buffer:

```bash
$PY train.py --resume-from runs/train_dqn_curriculum/best.pt --partial-load \
  --warmstart-buffer 100000 --warmstart-epsilon 0.05 \
  --generations 10 --total-decisions 10000000 \
  --curriculum --curriculum-max-level 4 --max-frames 60000 \
  --width 64 --n-step 3 --train-every 2 --batch-size 256 \
  --target-update 1000 --gamma 0.995 --replay-size 150000 \
  --epsilon-start 0.1 --reset-epsilon --epsilon-decay 3000000 \
  --epsilon-end 0.01 --lr 3e-5 --lr-final 1e-6 --eval-episodes 30 \
  --eval-level-cap 0 --ghost-score-scale 0.5 --danger-penalty 0.02 \
  --danger-radius 5 --death-penalty -20 --level-clear-bonus 50 \
  --stay-penalty -0.03 --step-penalty -0.005 \
  --dot-not-eaten-penalty -0.02 --dot-not-eaten-grace 45 \
  --dots-removed-min 0 --dots-removed-max 120 \
  --gif-frames 0 --out runs/train_dqn_finetune
```

For multiple candidates, `scripts/train_parallel.sh 3 10000000` starts
three of these runs with different reward settings and writes each to
`runs/parallel/seedN`; compare their `eval.json` files.

- `--curriculum` makes generation 1 cap at level 1, generation 2 at level
  2, and so on.  After `--curriculum-max-level`, later generations have no
  level cap.
- `--dots-removed-min/max` gives the agent endgame transitions without
  throwing away full-level experience; the range is sampled per episode.
- `--partial-load` transfers a checkpoint from the old observation
  architecture.  The new ghost-direction columns start at zero, so the
  resumed policy is exactly preserved before training changes it.
- A resumed checkpoint is evaluated first as generation 0.  `best.pt` is
  only replaced if a later generation beats it on
  `(clears, pellets, score)`.
- After each generation, `eval_genN.json` is written and the agent is
  saved as `dqn_genN.pt`.
- `best.pt` is selected by `(clears, pellets, score)` on the fixed
  evaluation seeds.  Evaluation always disables the endgame dot removal and
  uses the full maze.  The final live game uses `best.pt` unless
  `--no-best` is passed.
- `--resume` continues from `<out>/dqn.pt` and skips warmup.  `train.py`
  reads `progress.json` (or `run.json` for older runs) and starts at the
  next unfinished generation automatically; `--start-generation G`
  overrides that when you deliberately want to repeat a generation.  For
  example, `--resume --generations 10 --start-generation 5` continues a
  curriculum at level cap 5.
- `--resume-from PATH` loads an explicit checkpoint.  Combined with a new
  `--out` directory it preserves the original run's logs and GIF while the
  continuation writes its own report.
- `--reset-epsilon` restarts epsilon decay from `--epsilon-start` on
  resume (1.0 by default, but 0.1 is recommended for fine-tuning).  Without
  it, the checkpoint's low epsilon is kept.  A reset to 1.0 should only be
  used when deliberately throwing away the old policy.
- At the end of every run `train.py` writes `metrics.csv`, `stats.json`,
  `summary.md` and `training_showcase.png`.  Existing runs can be rebuilt
  with `python -m rl.report --run <directory>`; the same command also works
  on a live run before `run.json` exists.

## Commands

Use the existing CUDA environment (torch 2.14.1 + cu126, Python 3.10):

```bash
PY=/home/deathslash/venvs/mlasgn/bin/python

# verification
SDL_VIDEODRIVER=dummy $PY -m rl.smoke_test

# low-level DQN trainer (CSV logs every eval)
SDL_VIDEODRIVER=dummy $PY -m rl.train_dqn \
    --total-decisions 1000000 --width 64 --train-every 2 \
    --eval-every 50000 --out runs/dqn

# evaluate on 30 fixed seeds; level-cap 0 = full game
SDL_VIDEODRIVER=dummy $PY -m rl.eval_agent --agent dqn --width 64 \
    --level-cap 0 --checkpoint runs/dqn/best.pt --episodes 30

# plots and summary for a finished train.py run
$PY -m rl.report --run runs/train_dqn

# single-page six-panel showcase
$PY -m rl.plot --run runs/dqn
$PY -m rl.aggregate

# watch (needs a display)
$PY -m rl.watch --agent dqn --checkpoint runs/dqn/best.pt --level-cap 0

# record a GIF (no display needed)
SDL_VIDEODRIVER=dummy $PY -m rl.record --agent dqn \
    --checkpoint runs/dqn/best.pt --level-cap 0 \
    --frames 1800 --out runs/dqn_demo.gif
```

## Metrics

Evaluation is greedy (no epsilon) on seeds `10000..10000+N-1`.  Reported
per episode: return, score, pellets eaten, survival frames, deaths, and
levels cleared.  The same reward, actions, frame skip, episode limit and
evaluation seeds are used everywhere.

## Known limitations

- DQN sees the full static maze, which makes level-specific overfitting
  easy; the curriculum is the first defence against that.
- Only level 1 is trained in the default 1M run.
- Clearing the game consistently is still the hard research problem; a
  longer run and the curriculum improve the odds but do not guarantee it.
- No prioritised replay, n-step returns or distributional RL yet.
