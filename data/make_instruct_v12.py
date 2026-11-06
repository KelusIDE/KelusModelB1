"""instruct-v12: instruct-v11 with fewer canned replies, plus open conversation from SmolTalk.

Kani answered anything unfamiliar with its nearest canned reply ("AI will take over the world" got the support message
for someone in pain): in instruct-v11, 25 sentences were 29% of all assistant replies, and 45% of replies were copies
of a sentence already seen 60 times. Here:
  - every exact reply is kept at most CAP times in training rows (the model's own support reply keeps SUPPORT_CAP as a
    backup; kelus' fixed crisis message still answers real crisis messages first). A row stays while any of its replies
    is under its cap, so every skill keeps its examples;
  - about 45,000 conversations from HuggingFaceTB/smol-smoltalk (Apache-2.0; data/smol-smoltalk/, downloaded): writing,
    explaining, rewriting, summarizing, Python and everyday chat. Left out: conversations with a system message (Kani
    has none), long-context ones, replies that claim another name ("as an AI language model", ChatGPT, SmolLM), and short
    fact lookups ("who was ...", "when did ...") that Kani answers by searching. Turns are kept while the conversation
    fits Kani's 1,024-token window (about 3,000 characters).
Validation: instruct-v11's held-out rows, plus 1,000 SmolTalk test conversations filtered the same way.

    python3 data/make_instruct_v12.py   # writes data/instruct-v12.jsonl
"""
import collections
import json
from pathlib import Path
import random
import re

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
SMOLTALK = ROOT / 'data/smol-smoltalk/data'
CAP, SUPPORT_CAP = 60, 400
SUPPORT = "I'm really sorry you're in this much pain."
SMOLTALK_TRAIN, SMOLTALK_VALIDATION = 45_000, 1_000
MAX_CHARACTERS = 3_000
SKIP_SOURCES = {'longalign'}
OTHER_NAME = re.compile(r"\b(?:as an ai|i(?:'m| am) an ai|language model|openai|chatgpt|gpt-[345]|smollm|hugging ?face|"
                        r"anthropic|claude|llama|qwen|mistral)\b", re.I)
LOOKUP = re.compile(r"^(?:who (?:is|was|were|invented|discovered|wrote)|when (?:did|was|is|were)|where (?:is|was|are)|"
                    r"what (?:year|is the (?:capital|population|height|tallest|largest))|how (?:many|tall|old|far))\b", re.I)


def capped(rows, rng):
    """Training rows after capping each exact reply; held-out rows are untouched."""
    rows = rows[:]
    rng.shuffle(rows)
    seen, kept = collections.Counter(), []
    for row in rows:
        replies = [m['content'] for m in row['messages'] if m['role'] == 'assistant']
        limit = lambda text: SUPPORT_CAP if text.startswith(SUPPORT) else CAP
        if any(seen[text] < limit(text) for text in replies):
            kept.append(row)
            seen.update(replies)
    return kept


def conversation(messages):
    """The leading user/assistant turns that fit, ending on a reply; None when nothing fits or the content is unsuitable."""
    if not messages or any(m['role'] not in ('user', 'assistant') for m in messages) or messages[0]['role'] != 'user':
        return None
    first = messages[0]['content'].strip()
    if len(first) < 120 and LOOKUP.match(first):
        return None
    kept, size = [], 0
    for message in messages:
        size += len(message['content'])
        if size > MAX_CHARACTERS:
            break
        if message['role'] == 'assistant' and OTHER_NAME.search(message['content']):
            break
        kept.append({'role': message['role'], 'content': message['content'].strip()})
    while kept and kept[-1]['role'] != 'assistant':
        kept.pop()
    return kept if len(kept) >= 2 and all(m['content'] for m in kept) else None


def smoltalk(files, wanted, split, rng):
    rows = []
    for path in files:
        for record in pq.read_table(path, columns=['messages', 'source']).to_pylist():
            if record['source'] in SKIP_SOURCES:
                continue
            messages = conversation(record['messages'])
            if messages:
                rows.append({'split': split, 'family': 'smoltalk', 'source': f"smol-smoltalk/{record['source']}", 'messages': messages})
    rng.shuffle(rows)
    return rows[:wanted]


def build(seed=2412):
    rng = random.Random(seed)
    rows = [json.loads(line) for line in (ROOT / 'data/instruct-v11.jsonl').read_text(encoding='utf-8').splitlines()]
    train = capped([r for r in rows if r['split'] == 'train'], rng)
    validation = [r for r in rows if r['split'] != 'train']
    train += smoltalk(sorted(SMOLTALK.glob('train-*.parquet')), SMOLTALK_TRAIN, 'train', rng)
    validation += smoltalk(sorted(SMOLTALK.glob('test-*.parquet')), SMOLTALK_VALIDATION, 'validation', rng)
    rng.shuffle(train)
    return train + validation


if __name__ == '__main__':
    rows = build()
    destination = ROOT / 'data/instruct-v12.jsonl'
    destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    train = [r for r in rows if r['split'] == 'train']
    replies = collections.Counter(m['content'] for r in train for m in r['messages'] if m['role'] == 'assistant')
    top = sum(n for _, n in replies.most_common(25)) / sum(replies.values())
    print(f'{destination}: {len(rows):,} rows ({len(train):,} train, {sum(r["family"] == "smoltalk" for r in train):,} from SmolTalk); '
          f'the 25 most common replies are now {top:.1%} of replies')
