"""cide-v12: Cide 4.3's training data (2026-11-03), a short follow-up stage on Cide 4.2. Half cide-followups-v1 (chats
where a page is followed by a new request or by an edit to it; Cide 4.2 lost the thread after an earlier page) and
half cide-v11 replay (plain requests, big websites with plans, the 19 languages), so what Cide 4.2 learned is kept.

    .venv/bin/python data/make_cide_followups.py && .venv/bin/python data/make_cide_v12.py   # writes data/cide-v12.jsonl
"""
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent.parent


def build(seed=2434):
    read = lambda name: [json.loads(line) for line in (ROOT / 'data' / name).read_text(encoding='utf-8').splitlines()]
    rng = random.Random(seed)
    followups, replay = read('cide-followups-v1.jsonl'), read('cide-v11.jsonl')
    followups_train = [r for r in followups if r['split'] == 'train']
    train = followups_train + rng.sample([r for r in replay if r['split'] == 'train'], len(followups_train))
    rng.shuffle(train)
    return train + [r for r in followups + replay if r['split'] != 'train']


if __name__ == '__main__':
    rows = build()
    destination = ROOT / 'data/cide-v12.jsonl'
    destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    train = [r for r in rows if r['split'] == 'train']
    count = lambda source: sum(r.get('source') == source for r in train)
    print(f'{destination}: {len(train):,} training rows ({count("make_cide_followups"):,} chats with a follow-up, '
          f'{count("make_web_basics"):,} plain requests, {count("make_big_websites"):,} big websites), {len(rows) - len(train):,} held out')
