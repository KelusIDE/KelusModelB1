"""instruct-v8: Kani Otter 4's chat data (instruct-v7) plus deciding to look up places (data/places-v1.jsonl).

A fine-tuning mix for Kani Otter 4, which was trained on instruct-v7: every places example, and a replay sample
of v7 so its chat, code, reading and search skills stay. v7's "best restaurants in <city>" searches are dropped,
because those questions are now places calls. Validation keeps all of v7's held-out rows and the places ones.

    python3 data/make_instruct_v8.py   # writes data/instruct-v8.jsonl
"""
import json
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parent.parent
PLACE_SEARCH = re.compile(r'^SEARCH: best restaurants in ', re.I)


def build(seed=2406, replay=30000):
    rng = random.Random(seed)
    load = lambda name: [json.loads(line) for line in open(ROOT / 'data' / name, encoding='utf-8')]
    v7 = [r for r in load('instruct-v7.jsonl') if not PLACE_SEARCH.match(r['messages'][-1]['content'])]
    rows = rng.sample([r for r in v7 if r['split'] == 'train'], replay) + [r for r in v7 if r['split'] == 'validation']
    rows += load('places-v1.jsonl')
    rng.shuffle(rows)
    return rows


if __name__ == '__main__':
    rows = build()
    destination = ROOT / 'data/instruct-v8.jsonl'
    destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    counts = {}
    for row in rows:
        key = f"{row['split']}/{row.get('family')}"
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({'records': len(rows), 'counts': dict(sorted(counts.items()))}))
