#!/bin/zsh
set -euo pipefail
cd "${0:A:h}"
if [[ -f checkpoints/b1-v3.npz ]]; then
  exec python3 -m kelus.train --resume checkpoints/b1-v3.npz --steps 1000
else
  exec python3 -m kelus.train
fi
