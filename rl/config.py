"""Central configuration for the Pac-Man RL experiments.

All tunables live here so the training script, evaluation script and
environment wrapper can never silently disagree about important numbers
such as frame skip or reward scale.
"""

from dataclasses import dataclass, field

# Action index -> direction string understood by Pacman.set_dir().
# Order matches the original ghost tie-break (up, left, down, right),
# but for Pac-Man the order is arbitrary; it only has to stay fixed.
ACTIONS = ("up", "left", "down", "right")
NUM_ACTIONS = len(ACTIONS)

# Maze / observation constants.  These mirror map.py but are repeated as
# plain numbers so the RL code can be imported without pygame.
GRID_H, GRID_W = 31, 28
N_CHANNELS = 8          # wall, dot, power, pacman, 4 ghost channels
N_GHOSTS = 4
GHOST_DIR_DIM = N_GHOSTS * 4   # one-hot direction for each ghost
N_PELLETS = 244
VEC_DIM = 18            # size of the scalar feature vector


@dataclass
class RewardConfig:
    """How game events become RL reward.

    score_scale converts game points to RL units.  With 0.02, a dot is
    +0.2, a power pellet +1.0, a ghost +4/+8/+16/+32 and fruit +2..+100.
    A death is explicitly negative because the game itself never
    subtracts score when Pac-Man dies.

    stay_penalty is applied once per decision when Pac-Man's position did
    not change at all during the skipped frames.  This is different from
    step_penalty (which is charged for every decision): it specifically
    discourages parking in a corner or pressing into a wall forever.
    The default step penalty is -0.005: small enough that eating a dot
    (+0.2) easily pays for the moves needed to reach it, but large enough
    that idling is never free.

    dot_not_eaten_penalty is charged once per decision after
    dot_not_eaten_grace consecutive decisions without eating a dot or
    power pellet.  It adds a progress/time-pressure signal so that simply
    surviving in a safe corner is not as good as eating the next pellet.
    The default grace is 45 decisions, which is 3 seconds at frame_skip=4.
    It is not charged on terminal frames and resets as soon as a pellet is
    eaten.

    ghost_score_scale changes only the *ghost-combo* part of the score
    reward.  The game awards 200, 400, 800, 1600 points for a ghost chain;
    at score_scale=0.02 that is +4..+32, which can dominate the +0.2 dot
    and even the death penalty.  Reducing it makes risky ghost chasing less
    attractive while still rewarding a good power-pellet play.

    danger_penalty is a small per-decision shaping term for being close to
    a dangerous (scatter/chase) ghost.  It is scaled linearly by distance
    within danger_radius tiles.  It is not a replacement for the death
    penalty; it gives the agent a gradient before it is already dead.
    """

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


@dataclass
class EnvConfig:
    """Episode and environment settings.

    frame_skip: number of 60 Hz game frames per agent decision.
    max_frames: time limit for one episode, in game frames.
    level_cap: 1 means "stop and reset after clearing level 1";
               None means "let the agent continue into the next level".
    seed: optional seed for the environment's episode-seed generator.
    dots_removed_min/max: endgame curriculum.  Each new episode starts
               with a random number of ordinary dots already removed.
               Power pellets are never removed.  Evaluation must set both
               to zero so the score is comparable with full-level runs.
    """

    frame_skip: int = 4
    max_frames: int = 18000
    level_cap: int | None = 1
    seed: int | None = None
    dots_removed_min: int = 0
    dots_removed_max: int = 0
    reward: RewardConfig = field(default_factory=RewardConfig)
