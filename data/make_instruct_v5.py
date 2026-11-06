"""instruct-v5: reading practice (data/reading-v1.jsonl) mixed with a replay of instruct-v4.

Kani v5 continues from the released v4 model, so a 70,000-record sample of v4's training
records (every family, same proportions) keeps chat, feelings and code from fading while it
learns to answer from search results. v4 validation is sampled to 2,000; reading validation
(held-out articles only) is kept whole.

    python3 data/make_instruct_v5.py   # writes data/instruct-v5.jsonl
"""
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent.parent


def build(seed=2033, replay=70000, replay_validation=2000):
    rng = random.Random(seed)
    v4 = [json.loads(line) for line in open(ROOT / 'data/instruct-v4.jsonl', encoding='utf-8')]
    reading = [json.loads(line) for line in open(ROOT / 'data/reading-v1.jsonl', encoding='utf-8')]
    rows = rng.sample([r for r in v4 if r['split'] == 'train'], replay) + rng.sample([r for r in v4 if r['split'] == 'validation'], replay_validation)
    rows += reading
    rng.shuffle(rows)
    return rows


if __name__ == '__main__':
    rows = build()
    destination = ROOT / 'data/instruct-v5.jsonl'
    destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    counts = {}
    for row in rows:
        key = f"{row['split']}/{row.get('family')}"
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({'path': str(destination), 'records': len(rows), 'counts': dict(sorted(counts.items()))}))
