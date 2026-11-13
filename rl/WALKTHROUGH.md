# Code walkthrough: the DQN Pac-Man system

This document explains the final RL layer next to the source files.  The
original game files (`env.py`, `pacman.py`, `ghost.py`, `map.py`,
`levels.py`, `game.py`) are untouched.  Only the new `rl/` package and
`train.py` are described here.

---

## 0. How the RL layer attaches to the game

- `env.Game` owns all state (Pac-Man, ghosts, pellets, timers) and has
  `update()` but no drawing.
- `Game(fx=False)` removes READY/death/eaten pauses, so one `update()` is
  exactly one 60 Hz game frame.
- `Pacman.set_dir(d)` is the same input a human gives with an arrow key.
- `maze.pellets` is a module-level dict, so there can be only one `Game`
  per process.  The training script therefore uses one environment and
  resets it for evaluation instead of creating a second game.
- The wrapper never edits the game source.  It reads public attributes
  such as `game.pacman`, `game.ghosts`, `game.level`, `game.fright` and
  `maze.pellets`.

---

## 1. `config.py`

```python
ACTIONS = ("up", "left", "down", "right")
NUM_ACTIONS = len(ACTIONS)
GRID_H, GRID_W = 31, 28
N_CHANNELS = 8
N_GHOSTS = 4
GHOST_DIR_DIM = 16
N_PELLETS = 244
VEC_DIM = 18
```

The action order is fixed for the whole project.  The network's output
index maps to this tuple.  The maze constants are repeated here so the
observation code does not need pygame just to learn the board size.

```python
@dataclass
class RewardConfig:
    score_scale: float = 0.02
    death_penalty: float = -15.0
    step_penalty: float = -0.005
    stay_penalty: float = -0.02
    dot_not_eaten_penalty: float = -0.02
    dot_not_eaten_grace: int = 45
    level_clear_bonus: float = 20.0
    ghost_score_scale: float = 0.5
    danger_penalty: float = 0.02
    danger_radius: int = 5
```

- `score_scale=0.02`: a dot is +0.2, a power pellet +1.0, ghosts
  +4..+32 and fruit +2..+100.  This keeps Q-values in a comfortable
  range for Adam while making pellets worth chasing.
- `death_penalty=-15`: the game subtracts no score on death, so this is
  the explicit survival signal.  It is large enough to discourage
  suicide but much smaller than the value of a cleared level.
- `step_penalty=-0.005`: a small cost for every decision.  A dot is
  +0.2, so normal travel remains profitable (20 decisions cost 0.1), but
  idling or pressing into a wall is never free.  It is deliberately paired
  with the larger stay penalty below.
- `stay_penalty=-0.02`: charged once per decision only when Pac-Man's
  position did not change at all.  This is the anti-camping term the
  project asked for.  It is one tenth of a dot, so it prevents "park in a
  corner forever" without making normal movement or a pellet pause
  expensive.
- `dot_not_eaten_penalty=-0.02`, `dot_not_eaten_grace=45`: after 45
  consecutive decisions without a dot (3 seconds at the default frame
  skip), charge -0.02 per decision until the next dot.  This is the
  progress-pressure term.  It is distinct from the anti-stay penalty:
  circling an empty corridor still makes no progress even while moving.
  The counter resets on a pellet and no penalty is applied on a terminal
  frame.
- `level_clear_bonus=20`: clearing a level is worth 1000 base game
  points, which is 20 reward units at `score_scale=0.02`; the same amount
  again makes the sparse event easier to learn.
- `ghost_score_scale=0.5`: halves only the 200/400/800/1600 ghost combo
  points.  Without it a four-ghost chain is +60 reward, four times a death,
  which encourages risky ghost chasing.  Dots, power pellets, fruit and the
  displayed game score are unchanged.
- `danger_penalty=0.02`, `danger_radius=5`: a per-decision cost when a
  scatter/chase ghost is within five tiles, scaled linearly by closeness.
  It is a survival gradient before the sparse death penalty, not a
  replacement for it.

```python
@dataclass
class EnvConfig:
    frame_skip: int = 4
    max_frames: int = 18000
    level_cap: int | None = 1
    seed: int | None = None
    dots_removed_min: int = 0
    dots_removed_max: int = 0
    reward: RewardConfig = field(default_factory=RewardConfig)
```

- `frame_skip=4`: one agent decision every four game frames; the agent
  acts 15 times per second.  This is the standard Atari-style action
  repeat and makes the environment about four times cheaper.
- `max_frames=18000`: five minutes of game time per episode.  A policy
  that learns to avoid ghosts could otherwise run forever.
- `level_cap=1`: end the episode after clearing level 1.  In `train.py`
  this is changed per generation by the curriculum.  `None`/`0` means no
  cap.
- `seed`: seed for the environment's episode-seed generator.  Each
  episode still gets its own seed, so training sees different ghost
  randomness.
- `dots_removed_min/max`: endgame curriculum.  At reset, a random number of
  ordinary dots in that range is removed before the first observation.
  Power pellets are never removed.  `Game.total` is updated so fruit timing
  is correct.  Evaluation sets both to zero.

---

## 2. `obs.py`

The observation is a dict:

```python
obs = {
    "grid": float32[8, 31, 28],
    "vec": float32[18],
    "ghost_dir": float32[16],
}
```

PyTorch convolutions expect `(channels, height, width)`, so height is rows
(31) and width is columns (28).

Grid channels:

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

Ghost values:

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

This follows the game's `collide()` exactly: scatter/chase ghosts kill
Pac-Man, frightened ghosts are edible, and the remaining modes are
harmless.  The network can distinguish danger, food and irrelevant
ghosts at a glance.

The 18 scalars are:

1-4. current direction one-hot;
5-8. buffered direction one-hot (Pac-Man may be waiting for a tile
centre before turning);
9. fright timer;
10. chase/scatter flag;
11. pellets left;
12. Cruise Elroy stage;
13. lives;
14. level;
15. moving flag;
16-18. fraction of dangerous/frightened/harmless ghosts.

The grid handles space; the vector handles timers, modes, lives and level.

`ghost_dir` is 16 bits: four one-hot directions for each of Blinky, Pinky,
Inky and Clyde.  A single grid frame says where a ghost is, but not where
it is about to go.  The direction is the missing signal for learning to
run away.  The vector is flattened as `(ghost, direction)`, using the same
up/left/down/right order as `ACTIONS`.

`wrap_tile` handles the warp tunnel (`x=-1` and `x=28`), and the wall grid
is built once per `ObservationBuilder` because walls never change.

---

## 3. `pacman_env.py`

### Constructor

```python
self.game = Game(fx=False)
self.rng = np.random.default_rng(self.cfg.seed)
```

`fx=False` is the most important choice for training: no cosmetic pauses,
so `update()` advances exactly one frame.  The separate NumPy RNG only
chooses episode seeds; Python's `random` is what the game uses for
frightened turns and fruit timing.

### `reset`

```python
seed = int(self.rng.integers(0, 2**31 - 1))
self.game.reset(seed=seed)
```

`Game.reset` restores level 1, 244 pellets, three lives and score 0.
Passing the same seed and actions reproduces an episode exactly; the
smoke test verifies this.

After the reset, the endgame curriculum may remove ordinary dots:

```python
self._apply_dot_curriculum(seed)
self.game.total = len(maze.pellets)
self.dots_at_start = len(maze.pellets)
```

The removal uses a private RNG seeded from the episode seed, so it is
deterministic.  Only `"."` tiles are eligible; the four power pellets stay.
`Game.total` is updated because fruit timing uses `total - remaining`.
`reset_round()` after a death does not refill pellets, so `total` stays
correct; a level change refills the maze and the step loop resets
`game.total` for the new level.

### `step`: action validation

```python
if not isinstance(action, (int, np.integer)) or not 0 <= int(action) < NUM_ACTIONS:
    raise ValueError(...)
```

A wrong action is a programming bug, not something to silently clamp.

### `step`: reward, death, level clear, time limit

```python
reward = float(self.cfg.reward.step_penalty)
for _ in range(self.cfg.frame_skip):
    lives_before = self.game.pacman.lives
    score_before = self.game.pacman.score
    combo_before = self.game.combo
    self.game.update()
    score_delta = self.game.pacman.score - score_before
    ghost_points = 0
    if self.game.combo > combo_before:
        ghost_points = 200 * (2 ** combo_before)
        ghost_points = min(ghost_points, max(0, score_delta))
    base_points = score_delta - ghost_points
    reward += base_points * self.cfg.reward.score_scale
    reward += (ghost_points * self.cfg.reward.score_scale
               * self.cfg.reward.ghost_score_scale)
```

`combo` increments only when a frightened ghost is eaten, so the wrapper
can separate ghost-combo points from dots, power pellets and fruit.  The
base score keeps `score_scale`; the ghost part is multiplied again by
`ghost_score_scale`.  This is why `ghost_score_scale` can reduce risky
ghost farming without changing pellet rewards or the displayed game score.

A death is detected by `pacman.lives` decreasing, not by `pacman.alive`.
With `fx=False`, `collide()` sets `alive=False`, decrements lives, and
then `finish_death()` immediately respawns Pac-Man.  The wrapper stops the
current skip immediately because respawn resets the direction to `left`;
continuing would move Pac-Man with an action the agent did not choose.

A level clear happens when `game.level > level_cap`; the wrapper adds the
clear bonus and ends the episode.  Hitting `max_frames` is a
**truncation**, not a terminal state, and is reported through
`info["truncated"]`.

### `step`: the anti-stay penalty

```python
moved = False
last_pos = (self.game.pacman.x, self.game.pacman.y)
...
if (self.game.pacman.x, self.game.pacman.y) != last_pos:
    moved = True
    last_pos = (self.game.pacman.x, self.game.pacman.y)
...
stayed = not moved and not died and not level_cleared and not truncated
if stayed:
    reward += self.cfg.reward.stay_penalty
```

Position is used instead of `pacman.moving` because `moving` is only
refreshed at tile centres and can be stale during a pellet pause.  The
penalty is applied once per decision, not once per game frame, and never
on a terminal frame.  `info["stayed"]` exposes the flag for tests and
debugging.

### `step`: the no-dot progress penalty

`len(maze.pellets)` is sampled before and after each game frame.  A
decrease means a dot or power pellet was eaten; a level change also sets
`dot_eaten=True` because `next_level()` refills the maze immediately.

```python
if dot_eaten or died or level_cleared:
    self.decisions_since_dot = 0
else:
    self.decisions_since_dot += 1

dot_not_eaten = (
    not done and not dot_eaten
    and self.decisions_since_dot >= reward.dot_not_eaten_grace
)
if dot_not_eaten:
    reward += reward.dot_not_eaten_penalty
```

This is a progress signal, not another movement signal.  Pac-Man can move
every frame through an empty corridor and still get the penalty.  It
starts after 45 decisions without a pellet by default and is skipped on
death, clear and timeout frames.  `info["dot_eaten"]`,
`info["decisions_since_dot"]` and `info["dot_not_eaten_penalty"]` expose
the state for tests.

### `step`: danger shaping

After the frame loop, the wrapper measures the closest scatter/chase ghost
using wrapped Manhattan distance:

```python
danger = self._danger_level()
if danger:
    reward -= danger * self.cfg.reward.danger_penalty
```

The danger level is 1 at distance 1 and falls linearly to 0 at
`danger_radius`.  It is skipped on terminal frames, where the death
penalty and the respawn already dominate.  This is a small gradient toward
running away before a death happens; it is not a hard rule.

### `info`

```python
info = {
    "episode_seed": ..., "frames": ..., "level": ..., "score": ...,
    "lives": ..., "dots_left": ..., "dots_eaten": ...,
    "dots_removed": ..., "dots_at_start": ..., "died": ...,
    "level_cleared": ..., "truncated": ..., "stayed": ...,
    "dot_eaten": ..., "decisions_since_dot": ...,
    "dot_not_eaten_penalty": ..., "ghost_score": ...,
    "danger": ..., "danger_penalty": ..., "episode_return": ...,
}
```

Every script logs from this dict, so plots, CSV files and evaluation agree.

---

## 4. `networks.py`: a dueling Q-network

```python
self.conv = nn.Sequential(
    nn.Conv2d(8, 32, 3, padding=1), nn.ReLU(),
    nn.Conv2d(32, width, 3, stride=2, padding=1), nn.ReLU(),
    nn.Conv2d(width, width, 3, stride=2, padding=1), nn.ReLU(),
    nn.Flatten(),
)
self.ghost_mlp = nn.Sequential(
    nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
)
self.trunk = nn.Sequential(nn.Linear(conv_out + 18 + 64, 512), nn.ReLU())
self.value = nn.Linear(512, 1)
self.advantage = nn.Linear(512, NUM_ACTIONS)
```

The `ghost_mlp`/extra 64 inputs exist only when `use_ghost_dir=True`.
An old checkpoint therefore has exactly the old parameter names and
shapes.  `--partial-load` copies the old columns of `trunk.0.weight` and
zeros the new ones, so the old policy is preserved at the start of the new
run.

The first convolution learns local patterns such as "wall next to a
pellet" or "ghost two tiles away".  The stride-2 layers reduce 31x28 to
8x7, keeping the fully connected part small.  The 18 scalars and the
64-unit ghost-direction embedding are concatenated after the convolution
so the head can mix "what is where", "where is each ghost going" and "what
mode am I in".

The dueling head computes:

```python
value = self.value(hidden)
advantage = self.advantage(hidden)
return value + advantage - advantage.mean(dim=1, keepdim=True)
```

Many Pac-Man states have nearly identical action values (an empty
corridor), while a few need a very different escape action.  Splitting
value and advantage makes that structure easier to learn and stabilises
the update.

The output size is computed with a dummy tensor, so changing the maze size
does not silently break the head.

---

## 5. `agents/dqn.py`: Double DQN

Two identical networks are created:

```python
self.online = QNetwork(width=width, use_ghost_dir=use_ghost_dir)
self.target = QNetwork(width=width, use_ghost_dir=use_ghost_dir)
self.target.load_state_dict(self.online.state_dict())
```

The online network is trained; the target network is a slowly changing
copy used for stable TD targets.

Action selection:

```python
def act(self, obs, greedy=False):
    if not greedy and self.rng.random() < self.epsilon:
        return self.rng.randrange(NUM_ACTIONS)
    ...
    return int(q.argmax(dim=1).item())
```

`greedy=True` is used for evaluation and live play.

The update:

```python
q = self.online(grid, vec, ghost_dir).gather(
    1, action.unsqueeze(1)
).squeeze(1)
with torch.no_grad():
    next_actions = self.online(
        next_grid, next_vec, next_ghost_dir
    ).argmax(dim=1, keepdim=True)
    next_q = self.target(
        next_grid, next_vec, next_ghost_dir
    ).gather(1, next_actions).squeeze(1)
    target = reward + self.gamma * next_q * (1.0 - terminal)
loss = F.smooth_l1_loss(q, target)
```

The online network chooses the next action; the target network evaluates
it.  This is Double DQN and reduces the over-estimation of plain DQN.
`(1.0 - terminal)` zeroes the bootstrap at real terminal states but keeps
it at time-limit truncations.

Huber loss behaves like squared error for small errors and absolute error
for large ones, which is robust to rare large rewards such as eating a
ghost.  Gradients are clipped at 10.  The target network is copied every
2,000 gradient steps.

`save()` stores the online and target networks, optimizer, decision count,
update count, epsilon and the architecture flags (`use_ghost_dir`, `width`,
`lr`), so `--resume` restores the policy, optimiser state and exploration
schedule.  The replay buffer is deliberately not saved (it is large); it
starts empty and refills from new experience.

`load()` rebuilds the network automatically when an old checkpoint uses a
different observation architecture, so `rl.eval_agent`, `rl.watch` and
`rl.record` can open both old and new checkpoints without extra flags.  With
`allow_partial=True` (`train.py --partial-load`), it instead keeps the new
architecture and copies every shape-compatible tensor.  The old trunk
columns load and the new ghost-direction columns start at zero, which makes
the transferred policy exactly equal to the old one before training.

---

## 6. `buffer.py`: the replay buffer

```python
self.grid = np.zeros((capacity, 8, 31, 28), dtype=np.int8)
self.next_grid = np.zeros_like(self.grid)
self.ghost_dir = np.zeros((capacity, 16), dtype=np.int8)
self.next_ghost_dir = np.zeros_like(self.ghost_dir)
```

A transition stores both `obs` and `next_obs` because the DQN target needs
the next state.  Observations are multiplied by 2 and stored as int8:
values are `0`, `0.5`, `-1`, `1`, so the encoded values `0`, `1`, `-2`,
`2` fit exactly.  A 100k buffer uses about 1.4 GB instead of 5.6 GB; the
ghost-direction bits add only 1.6 MB.  They are stored even when the current
agent ignores them, so one buffer layout works for both architectures.

The buffer is a ring buffer; when full, the oldest transition is
overwritten.  Sampling is uniform.  Prioritised replay is a possible
future upgrade, not part of the final working system.

---

## 7. `train.py`: the single entry point

### 7.1 SDL setup

```python
_HEADLESS_FLAG = "--headless" in sys.argv
_NO_DISPLAY = not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
if _HEADLESS_FLAG or _NO_DISPLAY:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
```

This happens before pygame is imported.  On a desktop the final game opens
a window; on a server, or with `--headless`, pygame still creates
surfaces and fonts but needs no display.

### 7.2 Drawing

`draw_game()` mirrors `game.py`'s `draw()` but writes to an arbitrary
surface.  Training renders a 224x288 off-screen frame; the live game uses
the display surface after `set_maze_scale()`.  It calls the original
`map.build_walls`, `map.draw_pellets`, `Pacman.draw` and `Ghost.draw`, so
the recorded visual is the real game.

### 7.3 Streamed GIF

A list of 100,000 frames at 224x288 would need about 19 GB of raw RGB.
`FFmpegGIFRecorder` instead pipes one frame at a time into ffmpeg:

```text
PyGame Surface -> RGB bytes -> rawvideo -> MP4
MP4 -> palettegen -> paletteuse -> GIF
```

The palette pass and the palette-application pass are separate.  A single
`palettegen+paletteuse` graph makes `paletteuse` buffer every frame while
it waits for the palette, which is OOM-killed on 100,000 frames.  Two
passes keep memory bounded.  The recorder checks free disk space every
2,000 frames and stops capturing rather than filling the filesystem; the
checkpoint is never lost.

### 7.4 Generations and curriculum

The decision budget is split evenly:

```python
base_decisions, extra = divmod(args.total_decisions, args.generations)
```

For each generation the script chooses a level cap:

```python
if args.curriculum:
    cap = generation if generation <= args.curriculum_max_level else None
else:
    cap = args.level_cap if args.level_cap > 0 else None
```

The network and replay buffer are created once, outside the generation
loop.  Only `env.cfg.level_cap` changes.  This is one continuous training
run, not five independent runs.

The endgame curriculum is a separate, per-episode axis:

```python
EnvConfig(
    ...,
    dots_removed_min=args.dots_removed_min,
    dots_removed_max=args.dots_removed_max,
)
```

Every training episode samples a number of ordinary dots to remove before
the first observation.  This is what gives the agent repeated practice on
the last 80 pellets.  It is not a start-level curriculum: every episode
still begins on level 1.  Evaluation temporarily sets both bounds to zero,
so a full-level score is always comparable.

### 7.5 Warmup and epsilon

```python
if global_decision <= warmup:
    action = rng.randrange(4)
else:
    action = agent.act(obs)
```

The first 5,000 decisions are random and only fill the replay buffer.
Training on a buffer containing only a handful of near-identical states is
unstable.  For a short smoke run the warmup is reduced to at most one
quarter of the budget.  `--resume` sets warmup to zero so a learned policy
is not destroyed.  On resume, `detect_last_generation()` reads
`progress.json` (written after every generation) or `run.json` and starts
at the next unfinished generation.  `--start-generation` overrides that
when a generation should deliberately be repeated.  `--total-decisions` is
the remaining decision budget, not the original total.  `--resume-from
PATH` loads an explicit checkpoint instead of `<out>/dqn.pt`; pairing it
with a new `--out` preserves the original run's logs and GIF.
`--reset-epsilon` sets the loaded agent's decision counter to zero and
epsilon back to `--epsilon-start` (1.0 by default).  For a fine-tune, pass
`--epsilon-start 0.1`: a full reset to 1.0 explores randomly and can erase
the checkpoint before the new reward signal has any effect.
`--warmstart-buffer N` fills the replay buffer with N transitions from the
loaded policy before the first gradient step, so early updates are not
dominated by random exploration.  `--partial-load` transfers an old
architecture into the new ghost-direction network; the old policy is
preserved exactly because the new input columns start at zero.

The training loop also wraps transitions in an `NStepAccumulator`.  With
`--n-step 3`, the stored target includes three decisions of reward, which
shortens the path from a death or level clear back to the responsible
actions.  The accumulator flushes on terminal states, time-limit episode
ends and generation boundaries, so a return never crosses a reset.

`agent.update_epsilon()` decays linearly from 1.0 to `--epsilon-end`
(0.02) over `--epsilon-decay` decisions (500k by default).  A long run can
pass `--epsilon-decay 3000000` so exploration lasts longer.

### 7.6 Learning-rate decay

```python
if args.lr_final is not None:
    frac = min(1.0, global_decision / args.total_decisions)
    set_learning_rate(agent, args.lr + frac * (args.lr_final - args.lr))
```

This is optional.  A final rate of `1e-5` lets the network settle instead
of oscillating around a good policy at the end of a multi-million-decision
run.

### 7.6b Mixed precision and update scheduling

`DQNAgent` runs the forward pass under `torch.autocast` when the device is
CUDA.  The RTX 4060 supports bf16, which has the same exponent range as
fp32 and therefore needs no gradient scaler; older cards fall back to fp16
with `torch.amp.GradScaler`.  The optimizer and saved weights stay in
fp32, so checkpoints are unaffected.  `--no-amp` disables the feature if a
driver or custom kernel misbehaves.

The old loop took one batch-128 gradient step after every decision.  That
leaves the GPU with many tiny kernels and a lot of launch overhead.  The
recommended long-run setting is `--train-every 2 --batch-size 256`: the
same 128 samples per decision, but half as many, twice-as-large updates.
Action selection is still one batched forward per decision, so the policy
does not change between the two collected transitions in any meaningful
way.  Trace the exact flags into `run.json` when comparing throughput.

### 7.7 Episode bookkeeping

On `done`, the script writes one row to `episodes.csv`:

```text
generation, episode, decision, return, score, dots, dots_removed,
frames, deaths, clears, epsilon, loss_mean
```

`dots` is the true number of pellets eaten in the episode (from
`info["dots_eaten"]`), not a full-level equivalent.  `dots_removed` records
the endgame-curriculum head start, so a low pellet count in a curriculum
episode is explainable.  `deaths` is `3 - lives`.  The CSV is flushed after
every episode so a long run can be inspected while it is still going.

### 7.8 Evaluation and best checkpoint

After each generation:

```python
result = run_eval(eval_cap)
```

`run_eval` temporarily changes `env.cfg.level_cap` and sets
`dots_removed_min = dots_removed_max = 0`, so evaluation always uses the
full maze.  With `--eval-level-cap 0`, every generation is measured on the
full game; with the default `-1`, it follows the training cap.

A resumed checkpoint is evaluated before any new training and is entered
as **generation 0**.  This baseline is important: without it, a
continuation could finish below the policy it started from and still be
called "best".  If the output directory already has a `best.json`, the
resume keeps that saved best and does not overwrite it with the latest
checkpoint.  `load_previous_generation_rows()` also re-reads the old
`run.json`/`eval_genN.json` files so the final table and the six-panel
showcase stay continuous across an interrupted run.

The comparison metric is lexicographic:

```python
metric = (result["clears_mean"], result["dots_mean"], result["score_mean"])
```

Level clears dominate, then pellets, then score.  A new best saves
`best.pt` and `best.json`; `best_source` records whether it came from
generation 0 or a later generation.  The final live game loads `best.pt`
unless `--no-best` is passed.

### 7.9 Report and plots

`train.py` writes `run.json` first, then calls
`rl.report.build_report(out)`.  The report module rebuilds `metrics.csv`
from the generation evaluations in `run.json`, summarises `episodes.csv`,
and writes:

- `stats.json`: machine-readable run + episode summary;
- `summary.md`: a human-readable generation table and artifact list;
- `training_showcase.png`: one presentation-ready six-panel figure with
  training score/pellets/survival, episode return, DQN loss, and greedy
  evaluation by generation.

The plotting code lives in `rl/plot.py`.  It deliberately replaces the old
separate `training_curves*.png` and `training_episodes.png` files: the
evaluation-only curve could contain a single point, and the final-bars
chart became meaningless once Q-learning was removed.  The six panels are
score, pellets and survival (training plus greedy-evaluation markers),
training return, DQN loss, and greedy evaluation score/pellets by
generation.  The figure also includes a subtitle with the episode/decision
counts and a text box with the latest evaluation, so it can be dropped
straight into a presentation.

The same report can be rebuilt for a run created before this feature
existed, or while a run is still training:

```bash
python -m rl.report --run runs/train_dqn
```

`--no-report` skips the report when only checkpoints are wanted.

### 7.10 Final live game

The live game uses a fresh `Game(fx=True)`, restoring READY!, death pauses
and score popups.  It does not use `PacmanEnv`, so it is not limited by
`level_cap` and continues into level 2+ after a clear.  It asks the greedy
DQN for a direction every `--frame-skip` frames, calls
`pacman.set_dir(...)` and `game.update()`, and stops when `game.game_over`
is true.  `--live-max-frames` is only a safety cap for automated tests.

---

## 8. Evaluation and logs

`evaluate.py` runs greedy episodes and aggregates:

```python
levels_cleared = info["level"] - 1
dots_eaten = info.get(
    "dots_eaten",
    levels_cleared * N_PELLETS + (N_PELLETS - info["dots_left"]),
)
deaths = 3 - info["lives"]
```

`info["dots_eaten"]` is the true cumulative count, including later levels
and excluding dots that the endgame curriculum pre-removed.  The fallback
keeps old records readable.

The fixed evaluation seeds are `10000 + i`, so every generation and every
agent sees the same ghost randomness.  That makes the numbers paired and
comparable.

`train_dqn.py` writes `metrics.csv` with one row per evaluation, including
return, score, pellets, survival, deaths, clears and wall time.  `train.py`
builds the same schema through `rl.report` from its per-generation
evaluations.  `plot.py` reads it and draws mean +/- standard deviation with
a random baseline.  `aggregate.py` reads the three `eval_dqn_seed*.json`
files and produces a multi-seed table and bar chart.

Always compare on the greedy evaluation, not on the noisy exploration
episodes in `episodes.csv`.

---

## 9. Design choices and trade-offs

**Full grid vs egocentric crop.**  The full 28x31 grid is simple and lets
the network see the whole maze, which is good for level 1.  It also makes
it easy to overfit to one level.  An egocentric crop is a reasonable later
ablation, but the curriculum is the first defence used here.

**Frame skip 4.**  A human reacts in roughly 100-200 ms; four frames is
67 ms.  It keeps training fast.  Frame skip 2 would react faster but
double the decisions and slow training.

**Ghost directions vs frame stacking.**  Four direction one-hots per ghost
are cheap and directly encode the next movement.  Frame stacking would
also expose velocity but multiplies the grid channels by the stack depth
and the buffer memory.  The direction branch is the smaller first step and
can be transferred into an old checkpoint without changing the old policy.

**Endgame curriculum vs full episodes.**  Pre-removing dots gives many
more finishing transitions per million decisions.  Removing them from
every episode would distort the training distribution and make the
full-level evaluation harder to compare, so the command mixes 0..160.
Evaluation always disables removal.

**Reward scale.**  A dot is +0.2, a death is -15, a level clear is +20.
Survival is worth about 75 dots, which is strong but not overwhelming.
If the agent camps, increase `stay_penalty`; if it moves forever without
eating, increase the magnitude of `dot_not_eaten_penalty` or lower
`dot_not_eaten_grace`; if it suicide-runs, make `death_penalty` more
negative; if it farms ghost combos, lower `ghost_score_scale` or raise
`danger_penalty`; if it refuses to finish the last pellets, raise
`level_clear_bonus` or increase `dots_removed_max`.

**Stay penalty.**  Position displacement is the right test because the
game's `moving` flag is only refreshed at tile centres.  The penalty is
per decision and terminal frames are exempt.

**Seeds.**  A single run is an anecdote.  `run_experiments.sh` trains
three seeds and evaluates each on 30 fixed seeds.  Report mean and
standard deviation, not one lucky live game.

---

## 10. Why Q-learning is not in the code

The Q-learning implementation was removed rather than kept as a dormant
branch.  It used a hand-designed local state, could not see the full maze
or all four ghosts, and its Python-dict table scaled poorly.  In the
earlier experiments it also scored lower than DQN despite much cheaper
training.  The final system deliberately exposes one algorithm so there is
one code path to test and document.  The short notice is repeated in
`REPORT.md` and `README.md`.

---

## 11. Extending to PPO / actor-critic

The environment, observation, reward and evaluation code are
algorithm-agnostic.  To add PPO later:

1. Add a policy network with two heads: action logits and state value.
2. Collect a rollout of N decisions, storing `obs`, `action`, `log_prob`,
   `value`, `reward`, `done`, `truncated`.
3. Compute returns and advantages with GAE(lambda).
4. Optimise the clipped surrogate objective plus value loss and an
   entropy bonus.
5. Keep the same `evaluate()` and `watch.py` interfaces.

A2C is the same idea with a shorter rollout and no clipped update.
