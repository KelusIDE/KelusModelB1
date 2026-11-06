"""instruct-v11: instruct-v10 with Kani's own name. Every row is kept; only the fixed greeting and identity replies change.

Since instruct-v1 the curricula answered "hi" and "who are you" with the 2025 name "Kelus Kani B2:2F", so Kani Otter 4.2
and Kani Seal 3 still introduced themselves that way. The user chose "Kani Otter 4.3" (2026-10-26), the name of the model
this data trains. Prompts are unchanged, so no held-out probe wording enters training.

    python3 data/make_instruct_v11.py   # writes data/instruct-v11.jsonl
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RENAMES = {
    "Hi! I'm Kelus Kani B2:2F. What would you like to build?": "Hi! I'm Kani Otter 4.3 by Kelus Corp. What would you like to build?",
    "I'm Kelus Kani B2:2F, a small model trained locally from scratch for basic chat and code.":
        "I'm Kani Otter 4.3 by Kelus Corp., a small model trained from scratch for everyday chat and code.",
}


def rename(row):
    """The row with its fixed greeting and identity replies renamed, and how many replies changed."""
    changed = 0
    for message in row['messages']:
        if message['role'] == 'assistant' and message['content'] in RENAMES:
            message['content'] = RENAMES[message['content']]
            changed += 1
    return row, changed


def build():
    rows, changed = [], 0
    for line in (ROOT / 'data/instruct-v10.jsonl').read_text(encoding='utf-8').splitlines():
        row, n = rename(json.loads(line))
        rows.append(row); changed += n
    leftover = sum('B2:2F' in m['content'] for r in rows for m in r['messages'] if m['role'] == 'assistant')
    if leftover:
        raise SystemExit(f'{leftover} replies still use the old name in another wording: rename them too')
    return rows, changed


if __name__ == '__main__':
    rows, changed = build()
    destination = ROOT / 'data/instruct-v11.jsonl'
    destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    print(f'{destination}: {len(rows):,} rows, {changed:,} greeting and identity replies renamed')
