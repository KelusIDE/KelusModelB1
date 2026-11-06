"""cide-v10: Cide 4.1's training data (2026-11-02), a short follow-up stage on top of Cide 4. Half websites-big-v2
(the big web apps with a numbered plan and "2/5 reading list" markers), half an equal-sized random draw from cide-v7
(19 languages, Cide v5's data, small websites) so the rest of what Cide knows is kept.

    .venv/bin/python data/make_big_websites.py --plan && .venv/bin/python data/make_cide_v10.py   # writes data/cide-v10.jsonl
"""
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent.parent


def build(seed=2422):
    read = lambda name: [json.loads(line) for line in (ROOT / 'data' / name).read_text(encoding='utf-8').splitlines()]
    rng = random.Random(seed)
    big, rest = read('websites-big-v2.jsonl'), read('cide-v7.jsonl')
    big_train = [r for r in big if r['split'] == 'train']
    train = big_train + rng.sample([r for r in rest if r['split'] == 'train'], len(big_train))
    rng.shuffle(train)
    return train + [r for r in rest + big if r['split'] != 'train']


if __name__ == '__main__':
    rows = build()
    destination = ROOT / 'data/cide-v10.jsonl'
    destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    train = [r for r in rows if r['split'] == 'train']
    print(f'{destination}: {len(train):,} training rows ({sum(r.get("source") == "make_big_websites" for r in train):,} big websites with plans, '
          f'{sum(r.get("source") == "make_websites" for r in train):,} small websites), {len(rows) - len(train):,} held out')
