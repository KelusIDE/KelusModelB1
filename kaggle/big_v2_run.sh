#!/bin/bash




set -euo pipefail
cd "$(dirname "$0")"



stop_pod() {
  echo "run ended ($1) at $(date -u)"
  if [[ -n "${RUNPOD_POD_ID:-}" ]] && command -v runpodctl > /dev/null; then
    sleep "${STOP_DELAY:-1800}"
    runpodctl stop pod "$RUNPOD_POD_ID"
  fi
}
trap 'stop_pod "exit code $?"' EXIT



rm -f TIME_UP
if [[ -n "${MAX_HOURS:-}" ]]; then
  (sleep "$(python3 -c "print(int(float('$MAX_HOURS') * 3600))")"; touch TIME_UP; pkill -f 'kani_advanced.py (prepare|train)') &
fi


if command -v cargo > /dev/null; then
  python3 -m pip install -q ./kani_tokenize || echo "kani_tokenize did not build; tokenizing in Python"
fi

reached() {
  python3 -c "import json; p = json.load(open('run/kani-advanced/progress.json')); s = json.load(open('settings.json'))
raise SystemExit(0 if p['tokens_seen'] >= s['target_tokens'] else 1)"
}

for ((session = 0; session < ${SESSIONS:-1}; session++)); do
  reached && break
  python3 kani_advanced.py prepare --settings settings.json || [[ -e TIME_UP ]]
  [[ -e TIME_UP ]] && break
  python3 kani_advanced.py train --settings settings.json || [[ -e TIME_UP ]]
  [[ -e TIME_UP ]] && break
done
python3 kani_advanced.py export --settings settings.json
echo "run complete"
