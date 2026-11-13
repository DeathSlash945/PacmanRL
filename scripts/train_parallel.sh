#!/usr/bin/env bash
# Launch independent fine-tuning candidates in parallel.
#
# The game simulation and the tiny DQN updates use different resources, and
# this machine has plenty of CPU cores.  Running two or three candidates at
# once is therefore a useful way to explore reward settings without making a
# single run longer.  Each process writes its own run directory and picks its
# own best checkpoint; compare their eval.json files afterwards.
#
# Usage:
#   scripts/train_parallel.sh              # 2 candidates, 10M decisions each
#   scripts/train_parallel.sh 3 10000000   # 3 candidates, 10M decisions each
#
# Stop everything with Ctrl-C.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$ROOT/../../venvs/mlasgn/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3)"
fi

N="${1:-2}"
DECISIONS="${2:-10000000}"
BASE="$ROOT/runs/parallel"
START="$ROOT/runs/train_dqn_curriculum/best.pt"

if [[ ! -f "$START" ]]; then
  echo "missing start checkpoint: $START" >&2
  exit 1
fi
if ! [[ "$N" =~ ^[1-9][0-9]*$ ]] || ! [[ "$DECISIONS" =~ ^[1-9][0-9]*$ ]]; then
  echo "usage: $0 [candidates] [decisions]" >&2
  exit 1
fi

mkdir -p "$BASE"
pids=()
cleanup() {
  echo
  echo "stopping candidates..."
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup INT TERM

for ((i=0; i<N; i++)); do
  case $i in
    0) GHOST=0.50; DANGER=0.020; DOTS=120; CLEAR=50 ;;
    1) GHOST=0.35; DANGER=0.030; DOTS=160; CLEAR=80 ;;
    2) GHOST=0.65; DANGER=0.010; DOTS=80;  CLEAR=40 ;;
    *) GHOST=0.50; DANGER=0.020; DOTS=120; CLEAR=50 ;;
  esac

  out="$BASE/seed$i"
  mkdir -p "$out"
  echo "candidate $i: ghost=$GHOST danger=$DANGER dots_removed=$DOTS clear=$CLEAR -> $out"
  "$PY" "$ROOT/train.py" \
    --resume-from "$START" --partial-load \
    --warmstart-buffer 100000 --warmstart-epsilon 0.05 \
    --n-step 3 --gamma 0.995 \
    --epsilon-start 0.1 --reset-epsilon \
    --epsilon-decay 3000000 --epsilon-end 0.01 \
    --lr 3e-5 --lr-final 1e-6 \
    --batch-size 256 --train-every 2 --target-update 1000 \
    --replay-size 150000 \
    --generations 10 --total-decisions "$DECISIONS" \
    --curriculum --curriculum-max-level 4 --max-frames 60000 --width 64 \
    --death-penalty -20 --level-clear-bonus "$CLEAR" \
    --stay-penalty -0.03 --step-penalty -0.005 \
    --dot-not-eaten-penalty -0.02 --dot-not-eaten-grace 45 \
    --ghost-score-scale "$GHOST" --danger-penalty "$DANGER" --danger-radius 5 \
    --dots-removed-min 0 --dots-removed-max "$DOTS" \
    --eval-episodes 30 --eval-level-cap 0 --eval-seed0 10000 \
    --seed "$i" --out "$out" \
    --gif-frames 0 --save-every 500000 --log-every 50000 \
    --headless --no-watch --no-live \
    > "$out/train.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
echo "all candidates finished (status $status)"
for ((i=0; i<N; i++)); do
  echo "candidate $i: $BASE/seed$i/eval.json"
done
exit "$status"
