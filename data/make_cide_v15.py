"""cide-v15: Cide 4.6's training data (2026-11-05): terminal commands for macOS, Linux and PowerShell (data/commands-v1.jsonl),
with a share of cide-v14 replayed so that Cide 4.5's skills stay. Every part of cide-v14 keeps some rows, as Cide 4.4 showed what
dropping a part does (make_cide_v14.py). Rare languages keep all of theirs. Rows that overlap HumanEval or MBPP, or any held-out
request, are dropped, as before.

    .venv/bin/python -I data/make_commands.py && .venv/bin/python -I data/make_cide_v15.py   # writes data/cide-v15.jsonl
"""
import collections
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'data'))
from kelus.commands_eval import requests  # noqa: E402
from make_cide_v13 import benchmark_grams, grams, held_out_requests, words  # noqa: E402

REPLAY = {'contracts': 4000, 'lus': 1500}  # the rest: REST of each part, and parts under RARE rows whole
REST, RARE = 0.28, 400


def read(name):
    with (ROOT / 'data' / name).open(encoding='utf-8') as lines:
        return [json.loads(line) for line in lines]


def build(seed=2473):
    rng = random.Random(seed)
    v14, commands = read('cide-v14.jsonl'), read('commands-v1.jsonl')
    held = held_out_requests() | requests() | {' '.join(words(r['messages'][0]['content'])) for r in v14 if r['split'] != 'train'}
    contaminated = benchmark_grams()
    dropped = collections.Counter()
    train = []
    for r in commands:
        if r['split'] != 'train':
            continue
        if ' '.join(words(r['messages'][0]['content'])) in held:
            dropped['held out'] += 1
        elif grams(' '.join(m['content'] for m in r['messages'])) & contaminated:
            dropped['HumanEval/MBPP overlap'] += 1
        else:
            train.append({**r, 'part': 'commands'})
    parts = collections.defaultdict(list)
    for r in v14:
        if r['split'] == 'train':
            parts[r['part']].append(r)
    for part, rows in parts.items():
        count = REPLAY.get(part, len(rows) if len(rows) < RARE else round(len(rows) * REST))
        train += rng.sample(rows, min(count, len(rows)))
    rng.shuffle(train)
    validation = [r for r in v14 if r['split'] != 'train'] + [r for r in commands if r['split'] != 'train']
    return train + validation, dropped


if __name__ == '__main__':
    rows, dropped = build()
    destination = ROOT / 'data/cide-v15.jsonl'
    line = lambda r: json.dumps(r, ensure_ascii=False).replace('\u2028', '\\u2028').replace('\u2029', '\\u2029') + '\n'
    destination.write_text(''.join(line(r) for r in rows), encoding='utf-8')
    train = [r for r in rows if r['split'] == 'train']
    print(f'{destination}: {len(train):,} training rows, {len(rows) - len(train):,} held out; dropped {dict(dropped)}')
    print(dict(collections.Counter(r['part'] for r in train).most_common()))
