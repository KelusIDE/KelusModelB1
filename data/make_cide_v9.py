"""cide-v9: Cide 3's training data. cide-v7 (Cide 2's: coding in 19 languages, Cide v5's data and websites-v1) plus
websites-big-v1 (web apps with four to eight working features, up to 3,950 tokens, for Cide 3's 4,096-token window).

    .venv/bin/python data/make_big_websites.py && .venv/bin/python data/make_cide_v9.py   # writes data/cide-v9.jsonl
"""
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent.parent


def build(seed=2421):
    rows = [json.loads(line) for name in ('cide-v7.jsonl', 'websites-big-v1.jsonl')
            for line in (ROOT / 'data' / name).read_text(encoding='utf-8').splitlines()]
    train = [r for r in rows if r['split'] == 'train']
    random.Random(seed).shuffle(train)
    return train + [r for r in rows if r['split'] != 'train']


if __name__ == '__main__':
    rows = build()
    destination = ROOT / 'data/cide-v9.jsonl'
    destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    train = [r for r in rows if r['split'] == 'train']
    print(f'{destination}: {len(train):,} training rows ({sum(r.get("source") == "make_websites" for r in train):,} websites, '
          f'{sum(r.get("source") == "make_big_websites" for r in train):,} big websites), {len(rows) - len(train):,} held out')
