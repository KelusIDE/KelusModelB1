"""cide-v11: Cide 4.2's training data (2026-11-03), a short follow-up stage on Cide 4.1. Half web-basics-v1 (plain
requests for one common page, such as "Code a basic login html", which Cide 4.1 answered with one useless sentence),
and the other half split evenly between websites-big-v2 (big pages with a numbered plan) and cide-v7 (19 languages,
small websites), so what Cide 4.1 learned is kept.

    .venv/bin/python data/make_web_basics.py && .venv/bin/python data/make_cide_v11.py   # writes data/cide-v11.jsonl
"""
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent.parent


def build(seed=2432):
    read = lambda name: [json.loads(line) for line in (ROOT / 'data' / name).read_text(encoding='utf-8').splitlines()]
    rng = random.Random(seed)
    basics, big, rest = read('web-basics-v1.jsonl'), read('websites-big-v2.jsonl'), read('cide-v7.jsonl')
    basics_train = [r for r in basics if r['split'] == 'train']
    half = len(basics_train) // 2
    train = (basics_train + rng.sample([r for r in big if r['split'] == 'train'], half)
             + rng.sample([r for r in rest if r['split'] == 'train'], half))
    rng.shuffle(train)
    return train + [r for r in basics + rest + big if r['split'] != 'train']


if __name__ == '__main__':
    rows = build()
    destination = ROOT / 'data/cide-v11.jsonl'
    destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    train = [r for r in rows if r['split'] == 'train']
    count = lambda source: sum(r.get('source') == source for r in train)
    print(f'{destination}: {len(train):,} training rows ({count("make_web_basics"):,} plain requests, '
          f'{count("make_big_websites"):,} big websites with plans), {len(rows) - len(train):,} held out')
