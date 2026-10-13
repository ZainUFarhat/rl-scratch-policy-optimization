#!/usr/bin/env bash
set -euo pipefail

SCRIPT="${1:-reinforce.py}"
shift || true

for seed in 0 1 2 3 4; do
  echo "=== $SCRIPT | seed $seed ==="
  python "$SCRIPT" --seed "$seed" "$@"
done

echo "=== plotting ==="
python utils/plotting.py