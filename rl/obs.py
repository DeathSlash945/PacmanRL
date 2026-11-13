"""Turn a live Game object into a fixed-size observation.

Observation = a (8, 31, 28) float grid, a (18,) float vector, and a
(16,) ghost-direction vector.

Grid channels
-------------
0 wall
1 dot
2 power pellet
3 Pac-Man
4 Blinky
5 Pinky
6 Inky
7 Clyde

Ghost channel values
--------------------
 1.0  dangerous  (scatter / chase)  -> touching it kills Pac-Man
-1.0  frightened (edible)           -> touching it eats the ghost
 0.5  harmless   (house / leaving / eaten / entering)
 0.0  absent

The vector carries things a spatial grid cannot express well: current
and buffered direction, frightened timer, scatter/chase phase, pellets
left, Cruise Elroy stage, lives, level, and how many ghosts are in each
state.

The ghost-direction vector is four one-hot directions per ghost:
[Blinky up/left/down/right, Pinky ..., Inky ..., Clyde ...].  A single
frame shows where a ghost *is*; the direction tells the network where it
is about to go, which is the missing piece for learning to run away.
"""

import numpy as np

import map as maze
from .config import (
    ACTIONS,
    GHOST_DIR_DIM,
    GRID_H,
    GRID_W,
    N_CHANNELS,
    N_GHOSTS,
    N_PELLETS,
    VEC_DIM,
)

CH_WALL = 0
CH_DOT = 1
CH_POWER = 2
CH_PAC = 3
CH_GHOST0 = 4  # Blinky, Pinky, Inky, Clyde occupy channels 4..7

# The game's collide() only kills Pac-Man in scatter/chase and only lets
# him eat ghosts in frightened mode.  Every other mode is harmless.
GHOST_VALUE = {
    "scatter": 1.0,
    "chase": 1.0,
    "frightened": -1.0,
    "house": 0.5,
    "leaving": 0.5,
    "eaten": 0.5,
    "entering": 0.5,
}


def wrap_tile(tile):
    """Wrap tunnel tiles into the 0..27 grid.

    Pac-Man's x can be -1 or 28 while inside the warp tunnel.  The maze
    is 28 tiles wide, so modulo 28 maps those back onto the board.
    """
    c, r = tile
    return c % GRID_W, r


def one_hot_dir(direction):
    """One-hot encode a direction string; unknown directions stay all zero."""
    v = np.zeros(len(ACTIONS), dtype=np.float32)
    if direction in ACTIONS:
        v[ACTIONS.index(direction)] = 1.0
    return v


def build_wall_grid():
    """Static 31x28 wall mask, built once per observation builder.

    maze.is_wall() counts the ghost door '-' as a wall unless door=True,
    which is exactly what Pac-Man experiences, so we use the default.
    """
    walls = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    for r in range(GRID_H):
        for c in range(GRID_W):
            if maze.is_wall(c, r):
                walls[r, c] = 1.0
    return walls


class ObservationBuilder:
    """Builds observations from the live, module-level game state."""

    def __init__(self):
        self.walls = build_wall_grid()

    def build(self, game):
        grid = np.zeros((N_CHANNELS, GRID_H, GRID_W), dtype=np.float32)
        grid[CH_WALL] = self.walls

        # Pellets are a dict {(col, row): '.' or 'o'}.  Iterating all
        # remaining pellets is cheap (at most 244) and always correct.
        for (c, r), kind in maze.pellets.items():
            channel = CH_DOT if kind == "." else CH_POWER
            grid[channel, r, c] = 1.0

        pc, pr = wrap_tile(game.pacman.tile)
        grid[CH_PAC, pr, pc] = 1.0

        ghost_dir = np.zeros((N_GHOSTS, 4), dtype=np.float32)
        for i, ghost in enumerate(game.ghosts):
            gc, gr = wrap_tile(ghost.tile)
            grid[CH_GHOST0 + i, gr, gc] = GHOST_VALUE[ghost.mode]
            if ghost.dir in ACTIONS:
                ghost_dir[i, ACTIONS.index(ghost.dir)] = 1.0

        return {
            "grid": grid,
            "vec": self._vector(game),
            "ghost_dir": ghost_dir.reshape(GHOST_DIR_DIM),
        }

    def _vector(self, game):
        pac = game.pacman
        dangerous = sum(g.mode in ("scatter", "chase") for g in game.ghosts)
        frightened = sum(g.mode == "frightened" for g in game.ghosts)
        harmless = 4 - dangerous - frightened

        # game.sched[game.phase] is (frames, mode); mode is 'scatter'
        # or 'chase'.  The final schedule entry never advances further.
        phase_is_chase = 1.0 if game.sched[game.phase][1] == "chase" else 0.0

        vec = np.array(
            [
                *one_hot_dir(pac.dir),
                *one_hot_dir(pac.next_dir),
                min(game.fright / 360.0, 1.0),
                phase_is_chase,
                len(maze.pellets) / N_PELLETS,
                game.ghosts[0].elroy / 2.0,
                pac.lives / 3.0,
                min(game.level, 21) / 21.0,
                1.0 if pac.moving else 0.0,
                dangerous / 4.0,
                frightened / 4.0,
                harmless / 4.0,
            ],
            dtype=np.float32,
        )
        assert vec.shape == (VEC_DIM,)
        return vec
