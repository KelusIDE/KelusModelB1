"""instruct-v6: reading (data/reading-v1.jsonl), deciding to search (data/decide-v1.jsonl) and a
replay of instruct-v4 so chat, feelings and code stay.

In the replay, practiced declines ("I do not have enough training to answer that reliably ...")
to fact questions (who/what/when/where, how tall/many/far, news, prices) become search calls
("SEARCH: <question>"); gibberish and tasks it cannot do ("translate this", "predict the lottery") keep the decline.
Kani v6 continues from the released v4 model.

    python3 data/make_instruct_v6.py   # writes data/instruct-v6.jsonl
"""
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'data'))
from make_decide import query  # noqa: E402

DECLINE = 'I do not have enough training to answer that reliably.'


LOOKUP = re.compile(r"\b(?:who|what|whats|when|where|which|how (?:tall|many|much|far|old|big|long|deep|fast)|latest|news|population|capital|weather|price)\b")


def real_question(text):
    """A question worth searching: 'whats the capital of japan', not 'Zxqv' or tasks like 'translate this into french'."""
    words = [w for w in re.findall(r'[a-z]+', text.lower()) if len(w) >= 2 and re.search(r'[aeiouy]', w)]
    return len(words) >= 2 and bool(LOOKUP.search(text.lower()))


def searchable(row):
    """The replay record with declined real questions turned into search calls."""
    messages = [dict(m) for m in row['messages']]
    for i in range(1, len(messages), 2):
        if messages[i]['content'].startswith(DECLINE) and real_question(messages[i - 1]['content']):
            messages[i]['content'] = f"SEARCH: {query(messages[i - 1]['content'])}"
    return {**row, 'messages': messages}


def build(seed=2034, replay=70000, replay_validation=2000, decide=15000):
    rng = random.Random(seed)
    load = lambda name: [json.loads(line) for line in open(ROOT / 'data' / name, encoding='utf-8')]
    v4, reading, deciding = load('instruct-v4.jsonl'), load('reading-v1.jsonl'), load('decide-v1.jsonl')
    rows = [searchable(r) for r in rng.sample([r for r in v4 if r['split'] == 'train'], replay)]
    rows += [searchable(r) for r in rng.sample([r for r in v4 if r['split'] == 'validation'], replay_validation)]
    rows += reading + rng.sample([r for r in deciding if r['split'] == 'train'], decide) + [r for r in deciding if r['split'] == 'validation']
    rng.shuffle(rows)
    return rows


if __name__ == '__main__':
    rows = build()
    destination = ROOT / 'data/instruct-v6.jsonl'
    destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    counts = {}
    for row in rows:
        key = f"{row['split']}/{row.get('family')}"
        counts[key] = counts.get(key, 0) + 1
    converted = sum(m['content'].startswith('SEARCH: ') for r in rows if r.get('family') != 'decide' for m in r['messages'])
    print(json.dumps({'path': str(destination), 'records': len(rows), 'declines_now_searches': converted, 'counts': dict(sorted(counts.items()))}))
