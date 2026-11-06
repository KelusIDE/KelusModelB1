"""instruct-v10: instruct-v9 (Kani Otter 4.1's data: chat, code, reading, search, places) plus everyday skills.

From data/everyday-v1.jsonl (data/make_everyday.py), for a model with Kani's English tokenizer:
  - recipe questions answered by a recipe search (the app shows the recipe a page gives), learning questions by a
    search for the topic, and food chat kept as chat;
  - 2,000 phrasebook examples per language, all seven (short enough for any tokenizer);
  - Dolly's brainstorming, writing, classification and questions about a given passage. Its explanations from memory,
    and OpenAssistant's and Aya's answers, are left out: Kani looks facts up instead of recalling them.
Validation: v9's held-out rows and every held-out everyday row that the same filters keep.

    python3 data/make_instruct_v10.py   # writes data/instruct-v10.jsonl
"""
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent.parent
PHRASES_PER_LANGUAGE = 2000
SEARCHES = ('everyday_recipe_search', 'everyday_learn_search', 'everyday_food_chat')


def keep(row):
    family = row['family']
    if family in SEARCHES or family == 'everyday_phrases':
        return True
    return family == 'everyday_dolly' and row['source'] != 'dolly/general_qa'


def build(seed=2411):
    rng = random.Random(seed)
    load = lambda name: [json.loads(line) for line in open(ROOT / 'data' / name, encoding='utf-8')]
    rows = load('instruct-v9.jsonl')
    everyday = [r for r in load('everyday-v1.jsonl') if keep(r)]
    phrases = {}
    for r in everyday:
        if r['family'] == 'everyday_phrases' and r['split'] == 'train':
            phrases.setdefault(r['language'], []).append(r)
    rows += [r for r in everyday if r['family'] != 'everyday_phrases' or r['split'] == 'validation']
    for language, group in sorted(phrases.items()):
        rows += rng.sample(group, min(PHRASES_PER_LANGUAGE, len(group)))
    rng.shuffle(rows)
    return rows


if __name__ == '__main__':
    rows = build()
    destination = ROOT / 'data/instruct-v10.jsonl'
    destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    counts = {}
    for row in rows:
        key = f"{row['split']}/{row.get('family')}"
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({'records': len(rows), 'counts': dict(sorted(counts.items()))}))
