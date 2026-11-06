"""cide-v6: coding instructions in 19 languages for Cide 2, plus Cide v5's own data.

Cide v5 (8.7M) knew Python and a little JavaScript. Cide 2 starts from Kani Otter (88M) and learns from:
  - Magicoder OSS-Instruct (ise-uiuc/Magicoder-OSS-Instruct-75K, MIT): problem and solution, labeled by language;
  - Magicoder Evol-Instruct (ise-uiuc/Magicoder-Evol-Instruct-110K, Apache-2.0): language read from the reply's first
    code fence (```go, ```sql ...); replies without a recognised language are left out;
  - OpenCoder algorithmic problems (OpenCoder-LLM/opc-annealing-corpus, ODC-BY; the three shards downloaded hold Shell,
    C++ and Python): "Write a ... function to ..." then the code;
  - data/cide-contracts-v5/instructions.jsonl: Cide v5's training and held-out rows (Python data layers, SQL, chat replay).
Each language is capped (Python already dominates every source) so the rarer ones are not drowned out, and each keeps
HELD_OUT examples for validation. Only conversations that fit Kani's 1,024-token window (about 3,000 characters) stay.
Downloaded files live in data/cide-v6-sources/ (git-ignored).

    python3 data/make_cide_v6.py   # writes data/cide-v6.jsonl
"""
import collections
import json
from pathlib import Path
import random
import re

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / 'data/cide-v6-sources'
MAX_CHARACTERS = 3_000
HELD_OUT = 40
CAPS = {'python': 8_000}
DEFAULT_CAP = 3_000
ALIASES = {'c++': 'cpp', 'c#': 'csharp', 'cs': 'csharp', 'js': 'javascript', 'jsx': 'javascript', 'ts': 'typescript',
           'sh': 'shell', 'bash': 'shell', 'zsh': 'shell', 'py': 'python', 'python3': 'python', 'golang': 'go',
           'rb': 'ruby', 'rs': 'rust', 'mysql': 'sql', 'postgresql': 'sql', 'sqlite': 'sql'}
LANGUAGES = {'python', 'javascript', 'typescript', 'sql', 'html', 'css', 'java', 'c', 'cpp', 'csharp', 'shell', 'php',
             'go', 'swift', 'rust', 'ruby', 'perl', 'lua', 'r'}
FENCE = re.compile(r'```\s*([A-Za-z+#]+)')


def language(name):
    name = ALIASES.get(name.lower(), name.lower())
    return name if name in LANGUAGES else None


def example(prompt, reply, lang, source):
    prompt, reply = prompt.strip(), reply.strip()
    if not lang or not prompt or not reply or len(prompt) + len(reply) > MAX_CHARACTERS:
        return None
    return {'family': f'code_{lang}', 'source': source, 'language': lang,
            'messages': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': reply}]}


def magicoder_oss():
    for line in (SOURCES / 'magicoder-oss/data-oss_instruct-decontaminated.jsonl').open(encoding='utf-8'):
        row = json.loads(line)
        yield example(row['problem'], row['solution'], language(row['lang']), 'magicoder-oss-instruct')


def magicoder_evol():
    for line in (SOURCES / 'magicoder-evol/data-evol_instruct-decontaminated.jsonl').open(encoding='utf-8'):
        row = json.loads(line)
        fence = FENCE.search(row['response'])
        yield example(row['instruction'], row['response'], fence and language(fence.group(1)), 'magicoder-evol-instruct')


def opencoder():
    for path in sorted((SOURCES / 'opencoder/algorithmic_corpus/train').glob('*.parquet')):
        for row in pq.read_table(path, columns=['text', 'lang']).to_pylist():
            question, fence, code = row['text'].partition('```')
            if fence:
                yield example(question, fence + code, language(row['lang']), 'opencoder-algorithmic')


def build(seed=2413):
    rng = random.Random(seed)
    by_language = collections.defaultdict(list)
    for source in (magicoder_oss(), magicoder_evol(), opencoder()):
        for row in source:
            if row:
                by_language[row['language']].append(row)
    train, validation = [], []
    for lang, rows in sorted(by_language.items()):
        rng.shuffle(rows)
        validation += [{**r, 'split': 'validation'} for r in rows[:HELD_OUT]]
        train += [{**r, 'split': 'train'} for r in rows[HELD_OUT:HELD_OUT + CAPS.get(lang, DEFAULT_CAP)]]
    for line in (ROOT / 'data/cide-contracts-v5/instructions.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        (train if row['split'] == 'train' else validation).append(row)
    rng.shuffle(train)
    return train + validation


if __name__ == '__main__':
    rows = build()
    destination = ROOT / 'data/cide-v6.jsonl'
    destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    train = collections.Counter(r.get('language', 'cide-v5') for r in rows if r['split'] == 'train')
    print(f'{destination}: {len(rows):,} rows; training rows by language: '
          + ', '.join(f'{lang} {n:,}' for lang, n in train.most_common()))
