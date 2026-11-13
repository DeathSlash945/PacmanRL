#!/usr/bin/env bash
# Reproduce a three-seed DQN sweep and its fixed-seed evaluation.
# Run from the project root or from rl/.  Override PY if needed.
set -euo pipefail

PY=${PY:-/home/deathslash/venvs/mlasgn/bin/python}
cd "$(dirname "$0")/.."
export SDL_VIDEODRIVER=dummy

# One million decisions per seed is a quick comparison run.  For a serious
# attempt at clearing the game, use the longer command in README.md.
for seed in 0 1 2; do
  "$PY" -u -m rl.train_dqn \
    --total-decisions 1000000 --width 64 --train-every 2 \
    --eval-every 50000 --eval-episodes 10 \
    --seed "$seed" --out "runs/dqn_seed${seed}"
done

# Final 30-episode evaluations on identical seeds.  level-cap 0 means the
# evaluation continues through the whole game instead of stopping at level 1.
for seed in 0 1 2; do
  "$PY" -m rl.eval_agent --agent dqn --width 64 --level-cap 0 \
    --checkpoint "runs/dqn_seed${seed}/dqn.pt" --episodes 30 \
    --out "runs/eval_dqn_seed${seed}.json"
done

"$PY" -m rl.eval_agent --agent random --level-cap 0 --episodes 30 \
  --out runs/eval_random.json
"$PY" -m rl.plot --run runs/dqn_seed0
"$PY" -m rl.aggregate
