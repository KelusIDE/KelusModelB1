"""instruct-v7: v6's search and reading skills with chat restored.

v6 (data/make_instruct_v6.py) learned to search and read, but replaying only 70,000 v4 records
weakened multi-turn chat: a short follow-up ("hi!") after another exchange got the earlier reply
again. v7 continues from v6 with every chat, practiced and feelings record of instruct-v4 (declined
fact questions still become searches), a code sample, and smaller reading and decide samples.

    python3 data/make_instruct_v7.py   # writes data/instruct-v7.jsonl
"""
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'data'))
from make_instruct_v6 import searchable  # noqa: E402

CHAT = ('chat_pairs', 'practiced', 'feelings')


def build(seed=2037, code=15000, reading=15000, decide=8000):
    rng = random.Random(seed)
    load = lambda name: [json.loads(line) for line in open(ROOT / 'data' / name, encoding='utf-8')]
    v4, read, deciding = load('instruct-v4.jsonl'), load('reading-v1.jsonl'), load('decide-v1.jsonl')
    rows = [searchable(r) for r in v4 if r.get('family') in CHAT]
    rows += rng.sample([r for r in v4 if r['split'] == 'train' and r.get('family') not in CHAT], code)
    rows += rng.sample([r for r in v4 if r['split'] == 'validation' and r.get('family') not in CHAT], 600)
    rows += rng.sample([r for r in read if r['split'] == 'train'], reading) + [r for r in read if r['split'] == 'validation']
    rows += rng.sample([r for r in deciding if r['split'] == 'train'], decide) + [r for r in deciding if r['split'] == 'validation']
    rng.shuffle(rows)
    return rows


if __name__ == '__main__':
    rows = build()
    destination = ROOT / 'data/instruct-v7.jsonl'
    destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    counts = {}
    for row in rows:
        key = f"{row['split']}/{row.get('family')}"
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({'records': len(rows), 'counts': dict(sorted(counts.items()))}))
