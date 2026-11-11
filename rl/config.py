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
N_PELLETS = 244
VEC_DIM = 18            # size of the scalar feature vector


@dataclass
class RewardConfig:
    """How game events become RL reward.

    score_scale converts game points to RL units.  With 0.01, a dot is
    +0.1, a power pellet +0.5, a ghost +2/+4/+8/+16 and fruit +1..+50.
    A death is explicitly negative because the game itself never
    subtracts score when Pac-Man dies.
    """

    score_scale: float = 0.01
    death_penalty: float = -5.0
    step_penalty: float = 0.0
    level_clear_bonus: float = 10.0


@dataclass
class EnvConfig:
    """Episode and environment settings.

    frame_skip: number of 60 Hz game frames per agent decision.
    max_frames: time limit for one episode, in game frames.
    level_cap: 1 means "stop and reset after clearing level 1";
               None means "let the agent continue into the next level".
    seed: optional seed for the environment's episode-seed generator.
    """

    frame_skip: int = 4
    max_frames: int = 18000
    level_cap: int | None = 1
    seed: int | None = None
    reward: RewardConfig = field(default_factory=RewardConfig)
