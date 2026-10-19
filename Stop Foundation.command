#!/bin/zsh
set -euo pipefail
cd "${0:A:h}"
exec python3 -m kelus.foundation_status --stop
