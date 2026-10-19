#!/bin/zsh
set -euo pipefail
cd "${0:A:h}"
python3 -m kelus.auto_deploy --start
printf '\nPress Enter to close.\n'
read
