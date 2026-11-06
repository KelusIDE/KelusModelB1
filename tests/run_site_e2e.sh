#!/bin/zsh


set -euo pipefail
root="${0:A:h:h}"
rules="${KELUS_RULES:-$root/../Kelus/firestore.rules}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cp "$rules" "$work/firestore.rules"
cat > "$work/firebase.json" <<'JSON'
{"firestore": {"rules": "firestore.rules"},
 "emulators": {"auth": {"port": 9099}, "firestore": {"port": 8080}, "ui": {"enabled": false}, "singleProjectMode": true}}
JSON
cd "$work"
firebase emulators:exec --only auth,firestore --project kelus-ide \
  "python3 -m http.server 8791 --bind 127.0.0.1 --directory '$root/site' >/dev/null 2>&1 & server=\$!; sleep 1; node '$root/tests/site_accounts_e2e.mjs'; status=\$?; kill \$server; exit \$status"
