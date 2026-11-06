"""instruct-v9: instruct-v8 with the places data's v2 (data/places-v2.jsonl) and more of Kani's chat replayed.

v8's model learned places well but called places on feelings ("I need a pep talk", "stuck on this bug for hours").
v9 replays chat and feelings at about three times v8's share and adds v2's "I need ..." chat contrasts.
v7's "best restaurants in <city>" searches stay dropped. Validation: all of v7's held-out rows and places v2's.

    python3 data/make_instruct_v9.py   # writes data/instruct-v9.jsonl
"""
import json
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parent.parent
PLACE_SEARCH = re.compile(r'^SEARCH: best restaurants in ', re.I)
REPLAY = {'feelings': 12000, 'chat_pairs': 12000, 'practiced': 8000, 'reading': 5000, 'decide': 3000, 'python_compose': 2000,
          'python_spec': 1000, 'javascript_spec': 400, 'website': 300, 'python_power': 300}


def build(seed=2407):
    rng = random.Random(seed)
    load = lambda name: [json.loads(line) for line in open(ROOT / 'data' / name, encoding='utf-8')]
    v7 = [r for r in load('instruct-v7.jsonl') if not PLACE_SEARCH.match(r['messages'][-1]['content'])]
    rows = [r for r in v7 if r['split'] == 'validation']
    for family, count in REPLAY.items():
        rows += rng.sample([r for r in v7 if r['split'] == 'train' and r.get('family') == family], count)
    rows += load('places-v2.jsonl')
    rng.shuffle(rows)
    return rows


if __name__ == '__main__':
    rows = build()
    destination = ROOT / 'data/instruct-v9.jsonl'
    destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    counts = {}
    for row in rows:
        key = f"{row['split']}/{row.get('family')}"
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({'records': len(rows), 'counts': dict(sorted(counts.items()))}))
