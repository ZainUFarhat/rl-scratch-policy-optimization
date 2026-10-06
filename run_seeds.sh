#!/usr/bin/env bash
# Run a training script with seeds 0..4.
# Usage: ./run_seeds.sh reinforce.py --env CartPole-v1
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <script.py> [extra args passed through to the script]" >&2
  exit 1
fi

SCRIPT="$1"
shift

for seed in 0 1 2 3 4; do
  echo "=== seed $seed ==="
  python "$SCRIPT" --seed "$seed" "$@"
done
