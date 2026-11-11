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

## Project layout

| File | Purpose |
| --- | --- |
| `game.py` | Viewer: window, input, drawing, debug panel |
| `env.py` | `Game` class: all game logic, no drawing |
| `pacman.py` | `Pacman` class: movement, wall collision, speed, drawing |
| `ghost.py` | `Ghost` class: modes, targeting, movement, house walks, drawing |
| `levels.py` | Per-level parameter table and the `params(level)` lookup |
| `map.py` | Maze tile map, wall outlines, pellets, wall and tunnel checks |

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
- Reinforcement learning agents written from scratch: Q-learning, DQN, PPO

## Credits and disclaimer

Ghost behavior and timing follow the Pac-Man Dossier by Jamey Pittman. Pac-Man is a trademark of Bandai Namco. This is a non-commercial fan and educational project and contains no original game assets.

## License

Add a license before publishing.
