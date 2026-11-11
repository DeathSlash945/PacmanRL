# Pac-Man RL: Q-learning vs DQN

**Goal.** Keep the hand-written Pac-Man game exactly as it is, wrap it as a
reinforcement-learning environment, train a tabular Q-learning agent and a
Deep Q-Network agent, and compare them on identical evaluation seeds.

## TL;DR

| Agent | Eval score | Pellets eaten | Survival (frames) | Deaths | Level clears |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | TBD | TBD | TBD | TBD | TBD |
| Q-learning (3M decisions) | TBD | TBD | TBD | TBD | TBD |
| DQN (1M decisions) | TBD | TBD | TBD | TBD | TBD |

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

TBD: learning curves (`runs/comparison.png`), final table with mean and
standard deviation over seeds, and a discussion of the result.

## Discussion

TBD: which agent wins, on which metric, why, and what the failure modes
are (corner camping, ghost collisions, pellet loops, seed variance).

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
