"""cide-v7: Cide 2's training data. cide-v6 (coding in 19 languages and Cide v5's data) plus websites-v1 (complete,
working single-file websites; Cide's main goal is building one).

    python3 data/make_websites.py && python3 data/make_cide_v6.py && python3 data/make_cide_v7.py   # writes data/cide-v7.jsonl
"""
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent.parent


def build(seed=2415):
    rows = [json.loads(line) for name in ('cide-v6.jsonl', 'websites-v1.jsonl')
            for line in (ROOT / 'data' / name).read_text(encoding='utf-8').splitlines()]
    train = [r for r in rows if r['split'] == 'train']
    random.Random(seed).shuffle(train)
    return train + [r for r in rows if r['split'] != 'train']


if __name__ == '__main__':
    rows = build()
    destination = ROOT / 'data/cide-v7.jsonl'
    destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    train = [r for r in rows if r['split'] == 'train']
    print(f'{destination}: {len(train):,} training rows ({sum(r.get("source") == "make_websites" for r in train):,} websites), '
          f'{len(rows) - len(train):,} held out')
