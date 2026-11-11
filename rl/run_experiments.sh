#!/usr/bin/env bash
# Reproduce the full comparison.  Run from the project root or from rl/.
# Override PY if your torch environment lives somewhere else.
set -euo pipefail

PY=${PY:-/home/deathslash/venvs/mlasgn/bin/python}
cd "$(dirname "$0")/.."
export SDL_VIDEODRIVER=dummy

# Q-learning is cheap: 3M decisions per seed takes a few minutes.
for seed in 0 1 2; do
  "$PY" -u -m rl.train_q \
    --total-decisions 3000000 --eval-every 250000 --eval-episodes 10 \
    --seed "$seed" --out "runs/q_seed${seed}"
done

# DQN: 1M decisions per seed, one gradient step every 2 decisions.
for seed in 0 1 2; do
  "$PY" -u -m rl.train_dqn \
    --total-decisions 1000000 --width 32 --train-every 2 \
    --eval-every 50000 --eval-episodes 10 \
    --seed "$seed" --out "runs/dqn_seed${seed}"
done

# Final 30-episode evaluations on identical seeds.
for seed in 0 1 2; do
  "$PY" -m rl.eval_agent --agent q \
    --checkpoint "runs/q_seed${seed}/q.pkl" --episodes 30 \
    --out "runs/eval_q_seed${seed}.json"
  "$PY" -m rl.eval_agent --agent dqn \
    --checkpoint "runs/dqn_seed${seed}/dqn.pt" --width 32 --episodes 30 \
    --out "runs/eval_dqn_seed${seed}.json"
done

"$PY" -m rl.eval_agent --agent random --episodes 30 --out runs/eval_random.json
"$PY" -m rl.plot --q runs/q_seed0/metrics.csv --dqn runs/dqn_seed0/metrics.csv
