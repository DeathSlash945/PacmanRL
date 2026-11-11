# Code walkthrough: every design choice in the RL layer

This document explains the RL code line by line.  It is written to be
read next to the source files.  The original game files are untouched;
everything below lives in `rl/`.

---

## 0. How the RL layer attaches to the game

The game already separates logic from rendering:

- `env.Game` owns all state (Pac-Man, ghosts, pellets, timers) and has
  `update()` but no drawing.
- `Game(fx=False)` removes READY/death/eaten pauses, so one `update()`
  is exactly one 60 Hz game frame.
- `Pacman.set_dir(d)` is the same input a human gives with an arrow key.
- `maze.pellets` is a module-level dict, so there can be only one `Game`
  per process.  We therefore train with one environment per process and
  do not try to run several `Game` objects in the same process.

The wrapper never edits those files.  It reads public attributes such as
`game.pacman`, `game.ghosts`, `game.level`, `game.fright` and
`maze.pellets`.

---

## 1. `config.py`

```python
ACTIONS = ("up", "left", "down", "right")
NUM_ACTIONS = len(ACTIONS)
```

The order is fixed for the whole project.  Action index `0` always means
"request up", and the network's output index maps to this tuple.  The
order itself is arbitrary for Pac-Man; it is chosen to match the
up/left/down/right order the ghost AI already uses, which makes debugging
easier.

```python
GRID_H, GRID_W = 31, 28
N_CHANNELS = 8
N_PELLETS = 244
VEC_DIM = 18
```

These are the exact maze dimensions from `map.MAZE`.  They are repeated
here so observation code does not need to import pygame just to learn
the maze size.  `N_PELLETS` is used to normalise "pellets left" and to
compute pellets eaten.

```python
@dataclass
class RewardConfig:
    score_scale: float = 0.01
    death_penalty: float = -5.0
    step_penalty: float = 0.0
    level_clear_bonus: float = 10.0
```

- `score_scale=0.01` turns game points into small RL numbers: a dot
  gives `+0.1`, a power pellet `+0.5`, ghosts `+2..+16`, fruit
  `+1..+50`.  This keeps Q-values in a comfortable range for Adam.
- `death_penalty=-5` is the only negative event.  The game subtracts no
  score on death, so without this the agent could learn that suicide is
  free.  `-5` is 50 dots' worth; it is deliberately larger than a few
  pellets but not so large that the agent refuses to move.
- `step_penalty=0` is off initially.  If the agent learns to wander, a
  small value such as `-0.01` per decision encourages it to collect
  pellets faster.  It is off now because the user asked to start simple.
- `level_clear_bonus=10` rewards finishing level 1, which is a sparse
  event worth 1000 game points.

```python
@dataclass
class EnvConfig:
    frame_skip: int = 4
    max_frames: int = 18000
    level_cap: int | None = 1
    seed: int | None = None
    reward: RewardConfig = field(default_factory=RewardConfig)
```

- `frame_skip=4`: one agent decision every four game frames.  The agent
  acts 15 times per second, while the game still runs at 60 Hz.  This is
  the standard Atari-style action repeat.  It also makes the environment
  about four times faster than deciding every frame.
- `max_frames=18000`: five minutes of game time per episode.  Without a
  limit, a policy that learns to avoid ghosts could run forever and the
  episode would never end.
- `level_cap=1`: end the episode after clearing level 1.  This is the
  first training stage.  Set it to `None` (or `--level-cap 0` on the
  command line) to let the agent continue into level 2 and beyond.
- `seed`: seed for the environment's episode-seed generator.  Each
  episode still gets its own seed, so training sees different ghost
  randomness.
- `reward`: nested config so reward tuning never changes the wrapper.

---

## 2. `obs.py`

### Grid layout

```python
grid = np.zeros((8, 31, 28), dtype=np.float32)
```

PyTorch convolutions expect `(channels, height, width)`.  Height is rows
(31) and width is columns (28).

Channels:

| index | meaning |
| --- | --- |
| 0 | wall |
| 1 | dot |
| 2 | power pellet |
| 3 | Pac-Man |
| 4 | Blinky |
| 5 | Pinky |
| 6 | Inky |
| 7 | Clyde |

### Ghost values

```python
GHOST_VALUE = {
    "scatter": 1.0,
    "chase": 1.0,
    "frightened": -1.0,
    "house": 0.5,
    "leaving": 0.5,
    "eaten": 0.5,
    "entering": 0.5,
}
```

This follows the game's `collide()` exactly:

- `scatter`/`chase`: touching the ghost kills Pac-Man -> `1.0`.
- `frightened`: Pac-Man can eat it -> `-1.0`.
- `house`/`leaving`/`eaten`/`entering`: `collide()` ignores them, so
  they are harmless -> `0.5`, not `1.0`.

The network can therefore distinguish "dangerous", "edible" and
"harmless" at a glance.

### `wrap_tile`

```python
def wrap_tile(tile):
    c, r = tile
    return c % GRID_W, r
```

Pac-Man and ghosts can have `x=-1` or `x=28` in the warp tunnel.  The
maze is 28 tiles wide, so modulo 28 maps those positions back onto the
board.  `r` is always 0..30.

### `build_wall_grid`

```python
for r in range(GRID_H):
    for c in range(GRID_W):
        if maze.is_wall(c, r):
            walls[r, c] = 1.0
```

`maze.is_wall(c,r)` returns `True` for `#`, for the ghost door `-` when
`door=False`, and for out-of-bounds rows.  For Pac-Man the door is a
wall, so this is the correct mask.  It is built once per
`ObservationBuilder`, not every frame.

### `build`

```python
for (c, r), kind in maze.pellets.items():
    channel = CH_DOT if kind == "." else CH_POWER
    grid[channel, r, c] = 1.0
```

At most 244 iterations.  This is simpler than maintaining an incremental
pellet grid, and it cannot go stale after level resets.

```python
pc, pr = wrap_tile(game.pacman.tile)
grid[CH_PAC, pr, pc] = 1.0
```

`pacman.tile` is `(floor(x+0.5), floor(y+0.5))` from the game, i.e. the
tile the game itself uses for collisions.  Using the same definition is
important: the observation and the death rule must agree.

```python
for i, ghost in enumerate(game.ghosts):
    gc, gr = wrap_tile(ghost.tile)
    grid[CH_GHOST0 + i, gr, gc] = GHOST_VALUE[ghost.mode]
```

The ghost order is fixed by `Game.__init__`: Blinky, Pinky, Inky, Clyde.
Each gets its own channel, so the network can learn that Blinky targets
Pac-Man directly while Clyde sometimes runs away.

### `_vector`

The 18 scalars are:

1-4. current direction one-hot (`up`, `left`, `down`, `right`).
5-8. buffered direction one-hot (`next_dir`).  This matters because
Pac-Man may be waiting for a tile centre before turning.
9. `fright / 360` clipped to 1: how much frightened time is left.
10. `1.0` if the global mode is chase, else `0.0`.
11. `pellets_left / 244`.
12. `elroy / 2`: Blinky's Cruise Elroy stage.
13. `lives / 3`.
14. `level / 21` (capped).
15. `1.0` if Pac-Man is currently moving, else `0.0`.
16. fraction of ghosts dangerous.
17. fraction of ghosts frightened.
18. fraction of ghosts harmless.

The grid cannot express timers, modes, lives or level; the vector does.
The split is deliberate: convolutions handle space, the MLP head handles
global context.

---

## 3. `pacman_env.py`

### Constructor

```python
self.game = Game(fx=False)
```

`fx=False` is the single most important choice for training.  It removes
all cosmetic pauses, so `update()` advances exactly one frame.  The game
is therefore deterministic given the seed and actions, and fast.

```python
self.rng = np.random.default_rng(self.cfg.seed)
```

This RNG only chooses episode seeds.  It is separate from Python's
`random`, which the game uses for frightened ghost turns and fruit
timing.  `Game.reset(seed)` seeds that global RNG.

### `reset`

```python
if seed is None:
    seed = int(self.rng.integers(0, 2**31 - 1))
self.episode_seed = seed
self.game.reset(seed=seed)
```

Every episode gets an integer seed.  Passing the same seed and the same
actions reproduces the episode exactly; the smoke test verifies this.

`Game.reset` also resets the level to 1, restores 244 pellets, resets
lives to 3 and score to 0.  That is what an RL episode needs.

### `step`: action validation

```python
if not isinstance(action, (int, np.integer)) or not 0 <= int(action) < NUM_ACTIONS:
    raise ValueError(...)
```

A wrong action is a programming bug, not something to silently clamp.
This catches shape/index mistakes early.

### `step`: frame loop and reward

```python
reward = float(self.cfg.reward.step_penalty)
for _ in range(self.cfg.frame_skip):
    lives_before = self.game.pacman.lives
    score_before = self.game.pacman.score
    self.game.update()
    self.frames += 1
    reward += (self.game.pacman.score - score_before) * self.cfg.reward.score_scale
```

The score delta covers dots, power pellets, eaten ghosts and fruit in one
line, because the game already adds all of those to `pacman.score`.
`lives_before` is captured per frame, not per decision, because a death
can happen on any of the four skipped frames.

### `step`: death detection

```python
if self.game.pacman.lives < lives_before:
    died = True
    reward += self.cfg.reward.death_penalty
    if self.game.game_over:
        done = True
    break
```

This is the subtle part.  With `fx=False`, `collide()` sets
`pac.alive=False`, decrements lives, then `finish_death()` immediately
respawns Pac-Man and sets `alive=True` again.  So checking `alive` after
`update()` would miss the death.  The lives counter is the reliable
signal.

The `break` is also deliberate.  `respawn()` sets `dir='left'` and
`next_dir='left'`.  If the wrapper continued the remaining skip frames,
Pac-Man would move left with an action the agent did not choose.  Stopping
lets the agent make a fresh decision at the spawn point.

### `step`: game over, level clear, time limit

```python
if self.game.game_over:
    done = True
    break

if self.cfg.level_cap is not None and self.game.level > self.cfg.level_cap:
    level_cleared = True
    reward += self.cfg.reward.level_clear_bonus
    done = True
    break

if self.frames >= self.cfg.max_frames:
    truncated = True
    done = True
    break
```

- `game_over` means all lives are gone.  This is a true terminal state.
- `level > level_cap` means `next_level()` just ran.  We treat clearing
  the target level as a successful terminal state and add the bonus.
- hitting `max_frames` is a **truncation**, not a terminal state.  DQN
  still bootstraps from the last observation; Q-learning can also
  bootstrap.  The `info["truncated"]` flag is what tells the training
  loops the difference.

### `info`

```python
info = {
    "episode_seed": ..., "frames": ..., "level": ..., "score": ...,
    "lives": ..., "dots_left": ..., "died": ..., "level_cleared": ...,
    "truncated": ..., "episode_return": ...,
}
```

Every training script logs from this dict.  Keeping it in one place means
the plots and the CSV files agree.

---

## 4. Tabular Q-learning (`agents/qlearning.py`)

### Why a compressed state

A real Q-table over the raw game would need one entry per
`(position, pellet configuration, ghost configuration, timers)`.  There
are 244 pellets, so the pellet part alone has `2**244` configurations.
That is not a table; it is the universe.

The baseline therefore uses seven features:

```text
(col, row, 4-neighbour pellet bits, nearest-ghost sector,
 nearest-ghost distance bucket, fright bucket, current direction)
```

- `(col,row)` is 868 tiles; it lets the agent learn the maze layout.
- The four pellet bits say whether the tile above/left/right/below still
  has a pellet.  This is the local "where should I go next" signal.
- The nearest dangerous ghost is compressed to an 8-way sector and a
  distance bucket (`0..2`, `3..5`, `6..9`, `10+`).  This is enough to
  learn "run away from the red thing".
- The fright bucket (none, <=2 s, >2 s) lets it switch between fleeing
  ghosts and chasing them.
- The direction is needed because Pac-Man's legal turn behaviour
  depends on where he is already facing.

Even this is millions of states, so `self.q` is a Python dict and rows
are created only when visited.  That is still tabular Q-learning: each
discrete state has its own four numbers.

### Update rule

```python
target = reward
if not terminal:
    next_row = self.q.get(next_state)
    if next_row is not None:
        target += self.gamma * float(next_row.max())
row[action] += self.alpha * (target - row[action])
```

This is the standard TD(0) update:
`Q(s,a) <- Q(s,a) + alpha * (r + gamma * max Q(s',a') - Q(s,a))`.

If `next_state` has never been seen, its value is implicitly zero.  That
is the usual optimistic initialisation, and it encourages the agent to
visit new states.

### Exploration

```python
if not greedy and self.rng.random() < self.epsilon:
    return self.rng.randrange(NUM_ACTIONS)
```

During training the agent explores with probability epsilon.  Epsilon
decays linearly from 1.0 to 0.05 over 200k decisions.  During evaluation
`greedy=True` disables exploration entirely; ties are broken
deterministically with `np.argmax`, so the evaluation is reproducible.

### Hyperparameters

- `alpha=0.1`: a moderate learning rate.  Tabular Q-learning can use
  larger rates than neural networks, but 0.1 keeps the values from
  oscillating when a state is revisited.
- `gamma=0.99`: with a decision every 4 frames, the effective horizon is
  about 100 decisions = 6.7 seconds.  That is long enough to value
  survival and short enough to learn.
- `epsilon 1.0 -> 0.05`: early decisions are almost entirely random,
  which fills the table; late decisions are mostly greedy.

---

## 5. DQN

### 5.1 Network (`networks.py`)

```python
nn.Conv2d(8, 32, kernel_size=3, padding=1)
nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
```

- The first convolution looks at 3x3 patches and learns local patterns
  such as "wall next to a pellet" or "ghost two tiles away".
- The two stride-2 layers halve the spatial size twice: 31x28 -> 16x14
  -> 8x7.  This keeps the fully connected layer small.
- 32/64/64 channels are enough for a maze this small.  A wider network
  was measured at 1.4x slower with no clear benefit at this stage.

```python
features = self.conv(grid)
features = torch.cat([features, vec], dim=1)
return self.head(features)
```

The convolutional features are flattened to `64*8*7 = 3584` numbers and
concatenated with the 18 scalars.  The 512-unit hidden layer mixes
"what is where" with "what mode am I in", and the final layer outputs one
Q-value per action.

The output size is computed with a dummy tensor in `__init__` rather than
hard-coded arithmetic.  If the maze size ever changes, the network still
works.

### 5.2 Replay buffer (`buffer.py`)

```python
self.grid = np.zeros((capacity, 8, 31, 28), dtype=np.int8)
self.next_grid = np.zeros_like(self.grid)
```

A transition stores both `obs` and `next_obs`, because the DQN target
needs the next state.  Storing them as int8 codes is important:

```python
def encode_grid(grid):
    return np.rint(grid * 2.0).astype(np.int8)
```

The observation values are `0`, `0.5`, `-1`, `1`.  Multiplying by 2
gives `0`, `1`, `-2`, `2`, which fit in int8 exactly.  The agent divides
by 2 on the GPU.  A 100k buffer therefore uses about 1.4 GB instead of
5.6 GB.

The buffer is a ring buffer:

```python
self.pos = (self.pos + 1) % self.capacity
self.size = min(self.size + 1, self.capacity)
```

When it is full, the oldest transition is overwritten.  Sampling is
uniform:

```python
idx = self.rng.integers(0, self.size, size=batch_size)
```

Uniform sampling is the original DQN choice.  Prioritised replay is a
later upgrade.

### 5.3 Agent (`agents/dqn.py`)

```python
self.online = QNetwork(width=width).to(self.device)
self.target = QNetwork(width=width).to(self.device)
self.target.load_state_dict(self.online.state_dict())
```

Two identical networks.  The online network is trained; the target
network is a slowly changing copy used to compute stable TD targets.
Without it, the target moves every time the online network updates and
training can diverge.

```python
def act(self, obs, greedy=False):
    if not greedy and self.rng.random() < self.epsilon:
        return self.rng.randrange(NUM_ACTIONS)
    ...
    return int(q.argmax(dim=1).item())
```

Epsilon-greedy action selection.  `greedy=True` is used only for
evaluation and for the live viewer.

```python
q = self.online(grid, vec).gather(1, action.unsqueeze(1)).squeeze(1)
```

`gather` picks the Q-value of the action that was actually taken.  The
result has shape `(batch,)`.

Double DQN target:

```python
with torch.no_grad():
    next_actions = self.online(next_grid, next_vec).argmax(dim=1, keepdim=True)
    next_q = self.target(next_grid, next_vec).gather(1, next_actions).squeeze(1)
    target = reward + self.gamma * next_q * (1.0 - terminal)
```

The online network chooses the next action; the target network evaluates
it.  This reduces the over-estimation that plain DQN gets from
`max` over noisy Q-values.  `(1.0 - terminal)` zeroes the bootstrap at
real terminal states, but keeps it at time-limit truncations.

```python
loss = F.smooth_l1_loss(q, target)
```

Huber loss behaves like squared error for small TD errors and like
absolute error for large ones, which makes it robust to occasional
outliers (for example, eating a ghost for +16).

```python
nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
```

Gradient clipping at 10 prevents a single large TD error from destroying
the network.

```python
if self.updates % self.target_update == 0:
    self.target.load_state_dict(self.online.state_dict())
```

The target network is refreshed every 2000 gradient steps.  Too frequent
and it behaves like plain DQN; too slow and the targets become stale.

---

## 6. Training loops

### Q-learning (`train_q.py`)

The loop is deliberately simple:

```python
state = agent.state_from_game(env.game)
action = agent.act(state)
obs, reward, done, info = env.step(action)
next_state = agent.state_from_game(env.game)
terminal = done and not info["truncated"]
agent.learn(state, action, reward, next_state, terminal)
state = next_state
```

`terminal` is true only for game over or a real episode end.  A time-limit
truncation still bootstraps, because the game did not actually end.

Evaluation happens only at episode boundaries.  Evaluating in the middle
of an episode would call `env.reset()` and destroy the training episode.
The cost is that an evaluation can be delayed by at most one episode.

### DQN (`train_dqn.py`)

```python
if decision <= args.warmup:
    action = rng.randrange(4)
else:
    action = agent.act(obs)
```

The first 5000 decisions are random.  This fills the replay buffer with
diverse transitions before the first gradient step.  Training on a buffer
that contains only a handful of near-identical states is unstable.

```python
agent.remember(obs, action, reward, next_obs, terminal)
agent.update_epsilon()
if decision > args.warmup and decision % args.train_every == 0:
    loss = agent.learn()
```

One environment step, one stored transition, one epsilon decay, and
optionally one gradient step.  `train_every=2` means one update per two
decisions (one per eight game frames).  This was chosen after measuring
the GPU: it roughly halves wall-clock time with only a small sample
efficiency cost.  Use `--train-every 1` for the standard one-update-per-
decision setting if you have more time.

---

## 7. Evaluation (`evaluate.py`)

```python
levels_cleared = info["level"] - 1
dots_eaten = levels_cleared * N_PELLETS + (N_PELLETS - info["dots_left"])
deaths = 3 - info["lives"]
```

- `info["level"] - 1` counts cleared levels, even if the agent continued
  into the next one.
- `dots_eaten` adds 244 for every cleared level plus the pellets eaten on
  the current level.
- `deaths` is derived from lives, not from the last step's `died` flag,
  so it is correct even if the final step was not the death step.

Evaluation uses fixed seeds `10000 + i`.  Both algorithms see exactly the
same ghost randomness, which makes the comparison paired and fair.

---

## 8. Plotting and viewing

`plot.py` reads the `metrics.csv` files, plots mean +/- standard
deviation for return, score, pellets, survival frames, deaths and clears,
and draws a dashed random-play baseline.  The final-checkpoint bar chart
is a quick visual summary.

`watch.py` deliberately uses `frame_skip=1` and asks the agent for a new
action every four rendered frames.  That keeps the game at 60 FPS and
makes the animation smooth, while the agent still behaves like it did
during training (one decision per four frames).  It duplicates a small
amount of drawing code from `game.py` because `game.py` runs its main
loop at import time; importing it would start the human game.

---

## 9. Design choices and trade-offs

**Full grid vs egocentric crop.**  The full 28x31 grid is simple and
lets the network see the whole maze, which is good for level 1.  It also
makes it easy to overfit to one level.  An egocentric crop (for example
15x15 around Pac-Man) would generalise better across levels and be
cheaper, but it is a later ablation, not the first version.

**Frame skip 4.**  A human reacts in roughly 100-200 ms.  Four frames is
67 ms, which is fast but not super-human.  It also keeps the episode
short enough to train in minutes.  Frame skip 2 would react faster but
quadruple the number of decisions and slow training.

**Reward scale.**  A dot is +0.1 and a death is -5.  This makes survival
worth about 50 dots.  If the agent camps in a corner, increase the step
penalty or the pellet reward; if it suicide-runs, make the death penalty
more negative.  These are the first knobs to tune.

**Q-learning state.**  The hand-designed state is intentionally compact
so that the baseline is genuinely tabular and trains in seconds.  It
cannot see the whole maze or all four ghosts, so its ceiling is lower
than DQN's.  That is exactly the comparison the project is testing.

**Seeds.**  A single training run is an anecdote.  The experiment script
runs three seeds per algorithm and evaluates each on 30 fixed evaluation
seeds.  Report mean and standard deviation, not one lucky run.

---

## 10. Extending to PPO / actor-critic

The environment, observation, reward and evaluation code are
algorithm-agnostic.  To add PPO later:

1. Add a policy network with two heads: action logits and state value.
2. Collect a rollout of `N` decisions, storing `obs`, `action`,
   `log_prob`, `value`, `reward`, `done`, `truncated`.
3. Compute returns and advantages with GAE(lambda).
4. Optimise the clipped surrogate objective plus a value loss and an
   entropy bonus.
5. Keep the same `evaluate()` and `watch.py` interfaces.

A2C is the same idea without the clipped update and with a shorter
rollout.  DPO is not a standard online RL algorithm for this setting; it
is preference optimisation for generative models, so it is not the next
step for Pac-Man.
