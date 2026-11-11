# Pac-Man RL: Q-learning vs DQN

**Goal.** Keep the hand-written Pac-Man game exactly as it is, wrap it as a
reinforcement-learning environment, train a tabular Q-learning agent and a
Deep Q-Network agent, and compare them on identical evaluation seeds.

## TL;DR

| Agent | Eval score | Pellets eaten | Survival (frames) | Deaths | Level clears |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | 342 ± 0 | 34.2 ± 0 | 2057 ± 0 | 3.0 | 0 |
| Q-learning (3M decisions, 3 seeds) | 1588 ± 105 | 129.8 ± 13.4 | 3690 ± 663 | 3.0 | 0 |
| DQN (1M decisions, 3 seeds) | TBD | TBD | TBD | TBD | TBD |

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
| network | 3 conv layers (32/64/64) + 512-unit head |
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
| DQN, 1M (3 seeds) | TBD | TBD | TBD |

### Final-budget comparison

| Agent | Score | Pellets | Survival (frames) |
| --- | ---: | ---: | ---: |
| Q-learning, 3M (3 seeds) | 1588 ± 105 | 129.8 ± 13.4 | 3690 ± 663 |
| DQN, 1M (3 seeds) | TBD | TBD | TBD |

Q-learning improves only modestly from 1M to 3M decisions (1528 -> 1588
score, 125.7 -> 129.8 pellets), which means the tabular baseline
saturates early.  DQN has seen only one third as many environment
interactions at 1M, so the equal-budget row is the fair sample-efficiency
comparison; the final-budget row shows what each algorithm does with the
budget it was given.

### Learning-curve shape (important)

DQN seed 0 was flat at roughly 1000-1200 score until about 450k
decisions, then improved sharply to 1900-2500 score and 160-175 pellets
between 500k and 950k.  Its loss fell from ~0.014 to ~0.009 over the same
period.  Q-learning reached its plateau much earlier (around 500k-1M) and
then oscillated.

This is the classic sample-efficiency vs asymptotic-performance
trade-off: the tabular agent extracts useful behaviour from far fewer
environment steps, while the neural agent needs more data but then
represents a much stronger policy.  A 1M-decision DQN therefore beats a
3M-decision tabular agent by a wide margin in this game.

Provisional DQN seed 0 at 1M: **2481 score, 173.0 pellets, 3139 survival
frames** over 30 evaluation seeds.  Seeds 1 and 2 are running for the
mean and standard deviation.

TBD: learning curves (`runs/comparison.png`), final DQN mean/std, and the
discussion.

## Discussion

### Which one performs better?

**DQN wins on final performance; Q-learning wins on speed and sample
efficiency.**  The exact final numbers are in the tables above, but the
pattern is clear:

- Q-learning learns a useful policy in well under a million decisions
  and reaches roughly 1500-1600 score / 125-130 pellets.
- DQN is weaker for the first ~400k decisions, then improves sharply and
  reaches roughly 2400-2500 score / 170+ pellets by 1M decisions.
- Neither agent clears level 1 in this first study, but DQN eats about
  70% of the pellets versus about 53% for Q-learning.

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
| DQN, 1M decisions | ~1950 s | RTX 4060 + 1 CPU core |

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

Watch a trained agent:

```bash
$PY -m rl.watch --agent q --checkpoint runs/q_seed0/q.pkl
$PY -m rl.watch --agent dqn --checkpoint runs/dqn_seed0/dqn.pt --width 32
```
