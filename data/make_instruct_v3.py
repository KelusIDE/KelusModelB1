"""Instruction curriculum v3: v2 plus more chat practice and repeated multiplication.

At step 8000 of v2 training, Kani B2:2F passed every code probe (31/31) but chat
lagged: 75/115 held-out chat wordings, and 145/184 on the context regression, where
a greeting or thanks after an earlier exchange was often answered as a request to
clarify. Chat was 40% of v2. The data loader drops exact duplicates, so v3 raises
chat weight with new combinations instead of copies:

  * chat_pairs: two practiced single-turn exchanges joined into one conversation.
    Both replies are targets, so the second turn must be answered on its own merits
    whatever came before. Validation pairs use only practiced validation exchanges.
  * python_power: "x multiplied by itself N times" for N in two, four, five and six.
    Three stays held out for the readiness cube check, as does the name cube.
  * feelings (make_feelings.py): supportive replies for breakups, bad days, stress,
    good news and asking to be cheered up, with a fixed crisis reply that points to a
    person and a crisis line. Validation uses held-out wordings.

No benchmark or readiness prompt is added; held-out wordings stay held out.

    python3 data/make_instruct_v3.py   # writes data/instruct-v3.jsonl
"""
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'data'))
from kelus.evaluate import code_result  # noqa: E402
import make_instruct as v1  # noqa: E402
import make_instruct_v2 as v2  # noqa: E402
import make_feelings  # noqa: E402

TIMES = {2: 'two', 4: 'four', 5: 'five', 6: 'six'}
POWER_TEMPLATES = [
    'Write a Python function {name}({param}) that returns {param} multiplied by itself {count} times.',
    'Write a Python function {name}({param}) that returns {param} multiplied by itself {count} times. Return only the function.',
    'Create a Python function called {name} that takes {param} and returns {param} multiplied by itself {count} times.',
    'I need a Python function named {name}. It takes {param} and returns {param} multiplied by itself {count} times.',
    'Define {name}({param}) in Python. It returns {param} multiplied by itself {count} times.',
]


def chat_pairs(practiced, count, split, rng):
    singles = [r['messages'] for r in practiced if r['split'] == split and len(r['messages']) == 2]
    return [{'split': split, 'family': 'chat_pairs', 'messages': rng.choice(singles) + rng.choice(singles)} for _ in range(count)]


def power(rng, words, split):
    times = rng.choice(list(TIMES))
    param = rng.choice(words) if rng.random() < 0.6 else rng.choice(['x', 'n', 'a', 'value', 'number'])
    name = '_'.join(rng.sample(words, rng.choice([1, 2, 2])))
    if name in v1.HELD_OUT_NAMES or name in v2.RESERVED or name == param:
        return None
    expr = ' * '.join([param] * times)
    answer = f"```python\ndef {name}({param}):\n    return {expr}\n```"
    trials = [((n,), n ** times) for n in (2, -3, 5)]
    if not code_result(answer, name, trials):
        return None
    count = TIMES[times] if rng.random() < 0.7 else str(times)
    prompt = rng.choice(POWER_TEMPLATES).format(name=name, param=param, count=count)
    return {'split': split, 'family': 'python_power', 'messages': [
        {'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': answer}]}


def build(seed=2028, feelings=None):
    rng = random.Random(seed)
    rows, _, _ = v2.build()
    feelings = make_feelings.build() if feelings is None else feelings
    practiced = [r for r in rows if r['family'] == 'practiced'] + feelings
    rows += feelings
    rows += chat_pairs(practiced, 30000, 'train', rng) + chat_pairs(practiced, 600, 'validation', rng)
    train_words, validation_words = v2.vocabulary()
    for count, split, words in ((3000, 'train', train_words), (100, 'validation', validation_words)):
        made = 0
        while made < count:
            record = power(rng, words, split)
            if record:
                rows.append(record)
                made += 1
    unique = {}
    for row in rows:
        unique.setdefault(json.dumps(row['messages'], sort_keys=True), row)
    rows = list(unique.values())
    rng.shuffle(rows)
    return rows


if __name__ == '__main__':
    rows = build()
    destination = ROOT / 'data/instruct-v3.jsonl'
    destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    counts = {}
    for row in rows:
        key = f"{row['family']}/{row['split']}"
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({'path': str(destination), 'records': len(rows), 'counts': dict(sorted(counts.items()))}, indent=1))
