# Pac-Man RL: DQN from scratch

**Goal.** Keep the hand-written Pac-Man game exactly as it is, wrap it as a
reinforcement-learning environment, and train an agent that starts from
random weights and can eventually clear the game consistently.

## Status

This report describes the final DQN-only system.  `train.py` prints a
fixed-seed evaluation after every generation; the same numbers are written
to `eval_genN.json`, `eval.json` and `run.json`, so the results appear
directly in the terminal and in the output directory.

The original `runs/train_dqn` run is the 1M-decision baseline.  On 30 fixed
evaluation seeds it scored **1981.7 +/- 368.6**, ate **169.4 of 244
pellets**, and cleared **0 levels**.

The 10M-decision curriculum continuation in `runs/train_dqn_curriculum/`
finished with **2129.0 +/- 368.9** score, **171.7 pellets**, **3655 survival
frames**, and **0 clears** on its 30 fixed seeds.  Its best training
episode scored 3940, and the final 100 training episodes averaged 2103.8.
That is real progress, but still not a clear.

The diagnosis is now specific.  The agent has learned to collect pellets
and power pellets, but ghost-combo points dominate the reward: one power
pellet can produce up to 3000 game points, or +60 reward, while a death is
only -15.  The policy therefore takes risks near ghosts and has almost no
experience finishing a level.  The next run attacks both problems directly
with ghost-direction observations, an endgame dot-removal curriculum, and
reward shaping that reduces ghost-chase dominance and penalises proximity
to dangerous ghosts before the death penalty arrives.

## Algorithm note: Q-learning was tried and discarded

Q-learning was implemented first and then removed from the working
system.  The reasons:

1. **It cannot see the whole game.**  A tabular Q-table needs a compact
   hand-designed state.  The Q agent saw Pac-Man's tile, four neighbouring
   pellet bits, the nearest dangerous ghost, a fright bucket and its
   current direction.  It could not see the rest of the maze, the other
   ghosts, the fruit, or the global ghost mode.
2. **The table scales badly.**  Even the compressed state has millions of
   combinations, so the Q-table was a Python dictionary with one entry per
   visited state.  New states get no useful value until they are visited,
   and the table grows without bound.
3. **It scored lower.**  In the earlier trials the tabular agent reached a
   lower mean score and ate fewer pellets than DQN, despite using a
   multi-million-decision budget.  Its only advantage was cheap compute.

The Q-learning files, CLI options, plots, result rows and documentation
have been deleted.  This notice is kept so the decision is not silently
lost.

## What is built

- `rl/pacman_env.py`: Gym-style wrapper around the unmodified game.
- `rl/obs.py`: full 8 x 31 x 28 grid, 18 scalar features, and 16
  ghost-direction features (four one-hot directions per ghost).
- `rl/networks.py`: convolutional **dueling** Q-network with an optional
  ghost-direction branch; old checkpoints load unchanged because the branch
  is omitted when it is not used.
- `rl/agents/dqn.py`: Double DQN with replay buffer and target network.
- `rl/train_dqn.py`: low-level training loop and CSV logs.
- `rl/evaluate.py`, `rl/eval_agent.py`: fixed-seed evaluation.
- `rl/report.py`: per-run CSV/JSON/Markdown summary and the six-panel showcase.
- `rl/plot.py`, `rl/aggregate.py`: reusable showcase and multi-seed summaries.
- `rl/watch.py`, `rl/record.py`: live viewer and GIF recorder.
- `train.py`: the single from-scratch entry point: generations,
  curriculum, training GIF, best-checkpoint selection, evaluation and a
  live game until death.
- `rl/WALKTHROUGH.md`: line-by-line explanation of the design choices.

The game files `env.py`, `pacman.py`, `ghost.py`, `map.py`, `levels.py`
and `game.py` are untouched.

## Environment and reward

- **Actions:** up, left, down, right; Pac-Man's own buffered-turn logic
  decides when a turn is legal.
- **Frame skip:** 4 game frames per decision (15 decisions/second).
- **Observation:** full maze grid + direction, timers, lives, level,
  pellet count, ghost-mode counts, and each ghost's current direction.
- **Episode end:** game over, level clear (when capped), or the time limit.
- **Endgame curriculum:** a training episode can start with a random number
  of ordinary dots already removed.  Power pellets are never removed, and
  fixed-seed evaluation always uses the full 244-pellet maze.

Reward defaults in `rl/config.py`:

| Event | Reward |
| --- | ---: |
| dot | +0.2 |
| power pellet | +1.0 |
| ghost | +4 / +8 / +16 / +32, scaled by `ghost_score_scale=0.5` |
| fruit | score value x 0.02 |
| death | -15 |
| level clear | +20 |
| no movement for one decision | -0.02 |
| no dot for 45 decisions | -0.02 per decision until the next dot |
| near a dangerous ghost | up to -0.02 per decision within 5 tiles |
| every decision | -0.001 (`step_penalty`) |

The no-dot penalty is a progress signal, not a survival signal.  The danger
penalty is the survival signal: a dangerous ghost within five tiles costs a
small amount, scaled by closeness, so the agent receives a gradient before
it is already dead.  It is deliberately weak compared with the death
penalty and is skipped on terminal frames.

The ghost-combo scale is separate from the ordinary score scale.  Only the
200/400/800/1600 ghost points are reduced; dots, power pellets, fruit and
the game score itself are unchanged.  This makes a risky four-ghost chain
less attractive without removing the reward for a good power-pellet play.

For the aggressive next run the command uses
`--ghost-score-scale 0.35 --danger-penalty 0.03 --danger-radius 5
--death-penalty -20 --level-clear-bonus 30`.  These are CLI overrides, so
the original 10M result remains reproducible from its recorded config.

The anti-stay penalty is applied once per decision only when Pac-Man's
position is unchanged after all skipped frames.  It is deliberately
smaller than one dot, so it discourages corner-camping without teaching
the agent that standing still is worse than dying.  The wrapper exposes
`info["stayed"]` so the behaviour can be audited.

The `dot_not_eaten_penalty` is separate progress pressure.  After 45
consecutive decisions without a dot or power pellet (3 seconds at
`frame_skip=4`), the agent pays -0.02 per decision until it eats the next
pellet.  This is not the same as the anti-stay penalty: Pac-Man can move
constantly around an empty corridor and still make no progress.  The
penalty is not charged on death, level-clear or time-limit frames, and the
counter resets as soon as a pellet is eaten.  `info["dot_eaten"]`,
`info["decisions_since_dot"]` and `info["dot_not_eaten_penalty"]` expose
the signal for tests and debugging.

## Why these training changes

The first 1M-decision agents learned to eat pellets but never cleared
level 1.  The final system adds the changes that matter for a long run:

1. **Generations.**  A single opaque 10M-decision run is hard to inspect.
   Splitting it into generations gives a checkpoint and a fixed-seed
   evaluation every N decisions.  The network and replay buffer carry
   over, so this is not restarting from scratch.
2. **Level curriculum.**  Generation 1 only has to clear level 1,
   generation 2 level 2, and so on.  This keeps early learning focused.
   After the last curriculum level the cap is removed and the agent can
   play the whole game.  Note that this is an *episode-termination*
   curriculum, not a start-level curriculum: every episode still begins on
   level 1.  If the agent cannot clear level 1, the later caps never come
   into play and every generation effectively trains level 1.  That is why
   the next improvement should target the endgame rather than just adding
   more caps.
3. **Dueling DQN.**  Corridor states often have nearly equal action
   values; a dueling head separates the state value from the action
   advantages and is more stable on this kind of maze.
4. **Longer exploration and learning-rate decay.**  A multi-million
   decision run should not spend 95% of its life at the final epsilon.
   `--epsilon-decay 1000000` and `--lr-final 1e-5` make the late phase
   more useful.
5. **Best-checkpoint selection with a resume baseline.**  The final
   training checkpoint is not automatically the best one.  Every
   generation is evaluated on the same fixed seeds, and `best.pt` is
   chosen by `(level clears, pellets, score)`.  A resumed checkpoint is
   evaluated before training and entered as generation 0, so continuing a
   strong policy cannot silently replace it with a weaker one.
6. **More updates.**  `--train-every 1` gives one gradient step per
   decision, about twice the compute of the default but more sample
   efficiency.
7. **Ghost-direction observations.**  Position alone does not say where a
   ghost will be next frame.  The observation now includes four one-hot
   directions per ghost.  The optional MLP branch is zero-initialised when
   transferring from an old checkpoint, so the starting policy is exactly
   preserved.
8. **Endgame dot curriculum.**  Training episodes can start with up to 160
   ordinary dots already removed.  The agent therefore sees the last 80
   pellets many times per generation instead of only when a full level is
   almost finished.  Evaluation remains full-level.
9. **Ghost-risk reward shaping.**  `ghost_score_scale` reduces the value of
   ghost chains, while `danger_penalty`/`danger_radius` charge a small
   amount for standing near a dangerous ghost.  This directly targets the
   observed failure mode: farming power pellets and ghost combos instead of
   running away.

## Recommended run

For an overnight attempt, from the project root:

```bash
PY=/home/deathslash/venvs/mlasgn/bin/python

$PY train.py \
  --resume-from runs/train_dqn_curriculum/best.pt --partial-load \
  --generations 10 --total-decisions 20000000 \
  --curriculum --curriculum-max-level 4 \
  --max-frames 60000 --width 64 --train-every 1 --gamma 0.997 \
  --batch-size 128 --replay-size 150000 \
  --epsilon-decay 3000000 --epsilon-end 0.01 --reset-epsilon \
  --lr-final 5e-6 --eval-episodes 30 --eval-level-cap 0 \
  --ghost-score-scale 0.35 --danger-penalty 0.03 --danger-radius 5 \
  --death-penalty -20 --level-clear-bonus 30 \
  --stay-penalty -0.03 --dot-not-eaten-penalty -0.03 \
  --dot-not-eaten-grace 30 \
  --dots-removed-min 0 --dots-removed-max 160 \
  --gif-frames 100000 --out runs/train_dqn_final
```

`--partial-load` copies the convolutions, the dueling heads and the old
18-column part of the trunk from the 10M checkpoint.  The new
ghost-direction columns are zeroed, so generation 0 is exactly the old
policy; the branch learns from there.  The run is expected to take roughly
a day on the RTX 4060 because of the larger batch and 20M decisions.

If the run is interrupted, resume with the same output directory.  The
next unfinished generation is detected from `progress.json` (or
`run.json` for an older run); `--start-generation` only needs to be given
to repeat a generation deliberately:

```bash
$PY train.py --resume --generations 10 \
  --total-decisions 4000000 --curriculum --curriculum-max-level 4 \
  --dots-removed-min 0 --dots-removed-max 160 \
  --out runs/train_dqn_final
```

`--total-decisions` is the budget for the remaining generations.  The
10-generation split gives one fixed-seed evaluation and checkpoint per 2M
decisions, so the continuation gets its own learning curve.  Both the
resume baseline and every generation are evaluated with the endgame
curriculum disabled and on the full game, so the comparison is honest.

If the GPU is shared or time is short, start with 10M decisions and
`--train-every 2`; the generation loop, endgame curriculum and evaluation
still work the same.  The previous 10M run with `--train-every 1` took
about 12.4 hours on the RTX 4060; the new run is longer because of 20M
decisions and batch 128.

The output directory contains:

| File | Meaning |
| --- | --- |
| `dqn.pt` | final training checkpoint |
| `best.pt` | best evaluated checkpoint; used for the live game |
| `dqn_genN.pt` | checkpoint at the end of generation N |
| `eval_genN.json` | fixed-seed evaluation for generation N |
| `eval.json` | final evaluation of the selected agent |
| `episodes.csv` | one row per training episode |
| `metrics.csv` | one row per generation evaluation; plot input |
| `progress.json` | last completed generation; used by `--resume` |
| `run.json` | full configuration, timings and generation results |
| `stats.json` | machine-readable run + training-episode summary |
| `summary.md` | human-readable results table and artifact list |
| `training_showcase.png` | single-page six-panel training/evaluation figure |
| `training.gif` | 100,000-frame training timelapse |

## What is still likely missing

The new run now has endgame experience, ghost directions and survival
shaping, but none of these guarantees a clear.  The remaining risks are:

1. **The endgame is only half the problem.**  Starting with 160 dots gone
   teaches the last 80 pellets, but the agent still has to survive the
   first 160 in a full evaluation episode.  The mixed 0..160 curriculum
   keeps full-level experience in the buffer, which is why the long command
   does not remove dots from every episode.
2. **Reward shaping can be overdone.**  A danger penalty that is too strong
   makes the agent timid; the no-dot penalty then pushes it back into
   danger.  The command uses a small penalty (0.03 at one tile) and starts
   from a policy that already eats 170 pellets, so the trade-off is
   visible in the generation table.
3. **Exploration.**  A reset to epsilon 1.0 is necessary to escape the old
   policy, but 3M decisions of decay on a 20M run leaves a long low-epsilon
   phase.  If the first 3-4 generations do not beat generation 0, raise the
   decay or use `--epsilon-end 0.02`.
4. **More decisions.**  A from-scratch DQN on full Pac-Man is a large
   problem; 20M decisions is a serious attempt, not a guarantee.  A 50M
   budget with the same features is the next step if clears are still zero.
5. **Evaluation noise.**  Thirty episodes gives a useful estimate, but the
   score standard deviation is large.  Use 50-100 fixed seeds for a final
   claim, and report clears first, then pellets, then score.

A successful run should show generation 0 at about 2129 score / 171.7
pellets, then later generations with **nonzero clears**.  If no generation
beats generation 0, the resume baseline prevents a regression and the
correct decision is to inspect the training episodes before spending more
compute.

## What the evaluation means

Evaluation is greedy (no epsilon) on seeds `10000..10000+N-1`.  The
reported fields are return, score, pellets eaten, survival frames, deaths
and levels cleared.  `--eval-level-cap 0` evaluates the full game, which
is the right comparison once the agent is allowed past level 1.  With the
default `--eval-level-cap -1`, each generation is evaluated on its
training cap, which is more useful for watching the curriculum.

No agent has a guaranteed clear yet.  The honest success criterion is:
run the long command, then look at `eval.json` and `best.json`.  A
consistent clear should show `clears_mean >= 1` and low variance across the
30 fixed seeds, not one lucky live game.

## Compute reference

| Run | Wall-clock | Hardware |
| --- | ---: | --- |
| DQN 1M, width 32, update every 2 | ~33 min | RTX 4060 + 1 CPU core |
| DQN 5M, width 64, update every 2 | ~4.2 h | RTX 4060 + 1 CPU core |
| DQN 10M, width 64, update every 1 | 12.4 h | RTX 4060 + 1 CPU core |

These are training-only numbers; GIF rendering adds time, especially for
100,000 frames.

## Reproduce and inspect

```bash
PY=/home/deathslash/venvs/mlasgn/bin/python

# verification, including the anti-stay penalty
SDL_VIDEODRIVER=dummy $PY -m rl.smoke_test

# full three-seed DQN sweep (1M each) and final evaluations
bash rl/run_experiments.sh

# rebuild a run's summary and plots
$PY -m rl.report --run runs/train_dqn

# single-page showcase and multi-seed table
$PY -m rl.plot --run runs/dqn_seed0
$PY -m rl.aggregate

# watch the best checkpoint on the full game
$PY -m rl.watch --agent dqn --checkpoint runs/train_dqn_final/best.pt \
  --width 64 --level-cap 0
```

## Limitations

- The static full-maze observation can overfit to level 1; the curriculum
  is the first defence, but not a proof of generalisation.
- No prioritised replay, n-step returns or distributional RL yet.
- A full game clear is a long-horizon sparse-reward problem; the reward
  scale and curriculum are starting points, not a guarantee.
- `runs/` is intentionally git-ignored because checkpoints and the
  100,000-frame GIF are large.
