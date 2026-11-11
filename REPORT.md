# Pac-Man RL: Q-learning vs DQN

**Goal.** Keep the hand-written Pac-Man game exactly as it is, wrap it as a
reinforcement-learning environment, train a tabular Q-learning agent and a
Deep Q-Network agent, and compare them on identical evaluation seeds.

## TL;DR

| Agent | Eval score | Pellets eaten | Survival (frames) | Deaths | Level clears |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | 342 ± 0 | 34.2 ± 0 | 2057 ± 0 | 3.0 | 0 |
| Q-learning (3M decisions, 3 seeds) | 1791 ± 247 | 141.2 ± 27.5 | 3995 ± 181 | 3.0 | 0 |
| DQN (1M decisions, 3 seeds) | 2285 ± 375 | 170.3 ± 10.3 | 3068 ± 267 | 3.0 | 0 |

(Mean over 30 fixed evaluation seeds. Full numbers and standard
deviations are in the results section below.)

## What was built

- `rl/pacman_env.py`: Gym-style wrapper, no changes to the game.
- `rl/obs.py`: 8x31x28 grid + 18 scalars.
- `rl/agents/qlearning.py`: tabular Q-learning with a compact state.
- `rl/agents/dqn.py`: Double DQN with replay and target network.
- `rl/train_q.py`, `rl/train_dqn.py`: training loops and CSV logs.
- `rl/evaluate.py`, `rl/eval_agent.py`: fixed-seed evaluation.
- `rl/plot.py`: learning-curve comparison.
- `rl/watch.py`: watch a trained agent in pygame.
- `train.py`: one-file from-scratch training, 100,000-frame training GIF,
  final evaluation and a live game that plays until all lives are gone.
- `rl/WALKTHROUGH.md`: line-by-line explanation of every design choice.

The original files `env.py`, `pacman.py`, `ghost.py`, `map.py`,
`levels.py` and `game.py` are untouched.

## Experimental setup

- **Actions:** up, left, down, right (Pac-Man's own buffered-turn logic).
- **Frame skip:** 4 game frames per decision (15 decisions/second).
- **Observation:** full 28x31 maze, 8 channels, plus 18 scalar features.
- **Reward:** score delta / 100, death -5, level clear +10, no step
  penalty initially.
- **Episode:** ends on game over, level-1 clear, or 18,000 frames.
- **Training seeds:** 0, 1, 2.
- **Evaluation:** 30 fixed seeds (10000..10029), greedy policy.
- **Hardware:** RTX 4060 Laptop 8 GB, 32 CPU cores, 31 GB RAM.
- **Software:** Python 3.10, PyTorch 2.14.1+cu126, pygame-ce 2.5.8.

## Hyperparameters

### Q-learning

| Parameter | Value |
| --- | --- |
| alpha | 0.10 |
| gamma | 0.99 |
| epsilon | 1.0 -> 0.05 over 200k decisions |
| training decisions | 3,000,000 per seed |

### DQN

| Parameter | Value |
| --- | --- |
| network | 3 conv layers (32/32/32, `width=32`) + 512-unit head |
| replay | 100,000 transitions, int8 observations |
| batch | 64 |
| optimizer | Adam, lr 1e-4 |
| gamma | 0.99 |
| target update | every 2,000 gradient steps |
| epsilon | 1.0 -> 0.05 over 200k decisions |
| train every | 2 decisions |
| training decisions | 1,000,000 per seed |

## Results

### Equal-budget comparison (1M environment decisions)

| Agent | Score | Pellets | Survival (frames) |
| --- | ---: | ---: | ---: |
| Q-learning, 1M (3 seeds) | 1528 ± 101 | 125.7 ± 12.1 | 3437 ± 277 |
| DQN, 1M (3 seeds) | 2285 ± 375 | 170.3 ± 10.3 | 3068 ± 267 |

### Final-budget comparison

| Agent | Score | Pellets | Survival (frames) |
| --- | ---: | ---: | ---: |
| Q-learning, 3M (3 seeds) | 1791 ± 247 | 141.2 ± 27.5 | 3995 ± 181 |
| DQN, 1M (3 seeds) | 2285 ± 375 | 170.3 ± 10.3 | 3068 ± 267 |

Q-learning improves from 1528 to 1791 score and 125.7 to 141.2 pellets
between 1M and 3M decisions, but its seed spread widens (seed 0 reaches
2076, while seeds 1 and 2 sit near 1650).  DQN has seen only one third as
many environment interactions at 1M, so the equal-budget row is the fair
sample-efficiency comparison; the final-budget row shows what each
algorithm does with the budget it was given.

### Learning-curve shape (important)

DQN seed 0 was flat at roughly 1000-1200 score until about 450k
decisions, then improved sharply to 1900-2500 score and 160-175 pellets
between 500k and 950k.  Its loss fell from ~0.014 to ~0.009 over the same
period.  Q-learning's evaluation curve rises quickly through 1M decisions
and then oscillates around 1800-2100 score; its final greedy policy is
much better than its 1M checkpoint, but it does not show DQN's sharp late
takeoff.

This is the classic sample-efficiency vs asymptotic-performance
trade-off: the tabular agent extracts useful behaviour from far fewer
environment steps, while the neural agent needs more data but then
represents a stronger policy.  A 1M-decision DQN beats a 3M-decision
tabular agent on average by about 28% score, although the weakest DQN
seed (1853) is close to the Q-learning mean (1791).

Final DQN checkpoints over 30 evaluation seeds: seed 0 **2481 score /
173.0 pellets**, seed 1 **2523 / 178.9**, seed 2 **1853 / 158.9**.  The
mean is **2285 ± 375 score**.  The large standard deviation is driven by
seed 2, which improved later and less smoothly than seeds 0 and 1; it is
a real seed-variance warning, not an evaluation-seed artefact (all three
checkpoints use the same 30 evaluation seeds).

The learning curves are in `runs/comparison.png`; the 3-seed final bar
chart is `runs/seed_summary.png`.

### Single-file end-to-end demo

The standalone `train.py` was also run from scratch with the report
reward (`--score-scale 0.01 --death-penalty -5`), the same DQN width and
1M decisions.  It is a separate seed-0 trajectory because it does not run
the mid-training evaluations that `train_dqn.py` performs.  Its final
checkpoint scores **2359 ± 290** over 30 fixed evaluation seeds (177.5
pellets, 3127 survival frames).  The in-script 10-episode check reported
2383 ± 266, and the final live game scored 2330 before the third death.

That run produced `runs/train_demo/training.gif`: **100,000 frames,
224x288, 304.6 MiB**, a 30 fps training timelapse captured every tenth
decision.  The intermediate MP4 is 120 MiB.  Both live under `runs/`,
which is intentionally git-ignored because of their size.  The full
end-to-end run took 4235 s while sharing the GPU with another experiment;
most of the extra time versus the 1M training runs is frame rendering and
the two-pass GIF encode.

## Discussion

### Which one performs better?

**DQN wins on final performance; Q-learning wins on speed and sample
efficiency.**  The exact final numbers are in the tables above, but the
pattern is clear:

- Q-learning learns a useful policy in well under a million decisions
  and reaches 1791 score / 141 pellets after 3M decisions (1528 / 126 at
  1M).
- DQN is weaker for the first ~400k decisions, then improves sharply and
  reaches a 3-seed mean of 2285 score / 170 pellets by 1M decisions.  Its
  seed-to-seed spread is large (1853 to 2523), so DQN's advantage is
  reliable in the mean but not guaranteed for every seed.
- Neither agent clears level 1 in this first study, but DQN eats about
  70% of the pellets versus about 58% for Q-learning.

If the question is "which trained agent plays better?", the answer is
DQN.  If the question is "which method gives a usable agent for the least
compute?", the answer is Q-learning: 3M tabular decisions train in about
5 minutes on one CPU core, while 1M DQN decisions take about 33 minutes
on the RTX 4060.

### Why does DQN eventually win?

1. **More information.**  DQN sees the full maze, all four ghosts and all
   timers.  Q-learning sees only Pac-Man's tile, four neighbouring pellet
   bits, the nearest dangerous ghost and a frightened flag.
2. **Generalisation.**  A Q-table stores one independent value per
   visited state.  The neural network shares weights across similar
   states, so experience in one corridor helps in another.
3. **Better ghost handling.**  The full grid lets DQN learn Blinky's
   targeting behaviour and the scatter/chase phase, which the compressed
   Q-state cannot represent well.

### Why is DQN slow early?

DQN must learn the value function from scratch with random exploration,
a replay buffer and a target network.  Early Q-values are noisy, epsilon
is still high, and the network has not yet discovered that pellets are
worth moving toward.  Around 400k-500k decisions, those pieces line up
and the policy improves quickly.

### Failure modes

Both agents still lose all three lives.  Typical failures are:

- getting trapped in a corridor by a chasing ghost;
- turning into a ghost while committed to a pellet;
- wandering after the nearby pellets are gone (Q-learning especially,
  because it cannot see the rest of the maze).

A small step penalty, potential-based reward shaping, or a longer
training run are the obvious next experiments.

### Compute cost

| Run | Wall-clock | Hardware |
| --- | ---: | --- |
| Q-learning, 1M decisions | ~105 s | 1 CPU core |
| Q-learning, 3M decisions | ~320 s | 1 CPU core |
| DQN, 1M decisions | ~1950 s (3085 s while sharing the GPU) | RTX 4060 + 1 CPU core |

The DQN is roughly 6x more expensive than the 3M Q-learning run and
about 18x more expensive than the 1M Q-learning run.  That is the price
of the higher final score.

## Limitations

- Level 1 only in this first study.
- Q-learning sees a hand-designed local state, not the full maze.
- DQN sees the full static maze, so it can overfit to level 1.
- No prioritised replay, dueling heads, n-step returns or distributional
  RL yet.
- No PPO/A2C yet; the environment and evaluation code are ready for them.

## Reproduce

```bash
PY=/home/deathslash/venvs/mlasgn/bin/python
SDL_VIDEODRIVER=dummy $PY -m rl.smoke_test
bash rl/run_experiments.sh
$PY -m rl.plot
```

Single-file from-scratch run with the 100,000-frame training GIF and the
live game at the end:

```bash
$PY train.py
# server / no display:
$PY train.py --headless --no-watch
# reproduce this report's reward values explicitly:
$PY train.py --score-scale 0.01 --death-penalty -5
```

Watch a trained agent:

```bash
$PY -m rl.watch --agent q --checkpoint runs/q_seed0/q.pkl
$PY -m rl.watch --agent dqn --checkpoint runs/dqn_seed0/dqn.pt --width 32
```
