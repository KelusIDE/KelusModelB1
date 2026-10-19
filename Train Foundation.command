#!/bin/zsh
set -euo pipefail
cd "${0:A:h}"
if [[ -f training-runs/b1-foundation/latest.npz ]]; then
  exec /usr/bin/caffeinate -i /usr/bin/nice -n 10 python3 -m kelus.pretrain --resume
else
  exec /usr/bin/caffeinate -i /usr/bin/nice -n 10 python3 -m kelus.pretrain
fi
