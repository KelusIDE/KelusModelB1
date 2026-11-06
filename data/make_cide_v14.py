"""cide-v14: Cide 4.5's training data (2026-11-04), a repair of Cide 4.4.

Cide 4.4 (cide-v13) gained Python and HTML but fell on CRUD (19/24 to 8/24), development tasks (3/5 to 1/5) and the 19
languages (118/152 to 112/152). Why:
  - make_cide_v13.py replayed other languages with `language not in (None, ...)`. CRUD, function signatures, code
    continuation and MDN rows have language None, so all 13,200 of them were dropped by mistake. Sixteen thousand steps
    without them was enough to forget them;
  - Python was 54% of cide-v13 and JavaScript 14%, while 14 other languages shared 7% (C 80 rows, Lua 29, R 36). Five
    of the eight C requests then got Python.
This stage puts back all those rows, every usable row of the rare languages and more of the rest, and keeps a share of
cide-v13 so that what Cide 4.4 gained stays. Language rows must name their language. It also teaches Kel, the user's own
language (.lus), from data/lus-v1.jsonl, whose programs were all compiled and run with the real compiler. Rows that overlap HumanEval or MBPP, or any held-out request, are dropped, as before.

    .venv/bin/python -I data/make_lus.py && .venv/bin/python -I data/make_cide_v14.py   # writes data/cide-v14.jsonl
"""
import collections
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'data'))
from kelus.cide_saas_eval import CASES  # noqa: E402
from make_cide_v13 import benchmark_grams, grams, held_out_requests, words  # noqa: E402

# Rows per language from cide-v7. The rare ones have fewer than this, and all of them are used.
LANGUAGES = {'c': 900, 'go': 900, 'ruby': 900, 'perl': 900, 'lua': 900, 'r': 900, 'css': 900, 'typescript': 1500, 'html': 800, 'shell': 800,
             'rust': 800, 'swift': 800, 'cpp': 800, 'csharp': 800, 'java': 800, 'sql': 800, 'php': 800}
# Rows per part of cide-v13, kept so that Python, JavaScript and web pages stay.
KEEP = {'python-self-oss': 1800, 'python-magicoder': 1800, 'javascript': 1200, 'typescript': 200, 'html': 700, 'css': 100,
        'web-basics': 1000, 'web-big': 900, 'web-followups': 1500}


# A request must say which language it wants, or the row teaches nothing about choosing one ("Can you make me a create
# account page?" answered in C is in cide-v7). Case matters for the one-letter names.
NAMES = {'c': r'\bC\b|\bansi c\b', 'cpp': r'C\+\+|\bcpp\b', 'csharp': r'C#|c-sharp|csharp|\.net\b', 'go': r'\bGo\b|golang', 'r': r'\bR\b',
         'shell': r'\bbash\b|\bshell\b|\bsh\b|\bzsh\b|linux command|command line', 'html': r'\bhtml\b', 'css': r'\bcss\b|\bscss\b|\bstyles?heet',
         'typescript': r'\btypescript\b|\bts\b|angular', 'java': r'\bjava\b', 'lua': r'\blua\b', 'perl': r'\bperl\b', 'php': r'\bphp\b|laravel|wordpress',
         'ruby': r'\bruby\b|\brails\b', 'rust': r'\brust\b', 'sql': r'\bsql\b|\bmysql\b|\bpostgres|\bsqlite\b|\bquery\b', 'swift': r'\bswift\b|\bios\b|swiftui'}


def names_language(language, text):
    return bool(re.search(NAMES[language], text, 0 if language in ('c', 'r', 'go') else re.I))


def read(name):
    with (ROOT / 'data' / name).open(encoding='utf-8') as lines:
        return [json.loads(line) for line in lines]


def build(seed=2437):
    rng = random.Random(seed)
    v7, v13 = read('cide-v7.jsonl'), read('cide-v13.jsonl')
    held = held_out_requests() | {' '.join(words(r['messages'][0]['content'])) for r in v13 if r['split'] != 'train'}
    held |= {' '.join(words(prompt)) for _, prompt in CASES}
    held |= {' '.join(words(case['prompt'])) for case in json.loads((ROOT / 'data/cide-crud-v3/test-cases.json').read_text()) if isinstance(case, dict) and 'prompt' in case}
    contaminated = benchmark_grams()
    dropped = collections.Counter()

    def clean(rows):
        kept = []
        for r in rows:
            if ' '.join(words(r['messages'][0]['content'])) in held:
                dropped['held out'] += 1
            elif grams(' '.join(m['content'] for m in r['messages'])) & contaminated:
                dropped['HumanEval/MBPP overlap'] += 1
            else:
                kept.append(r)
        return kept

    train = [{**r, 'part': 'contracts'} for r in clean([r for r in v7 if r['split'] == 'train' and r.get('language') is None])]
    for language, count in LANGUAGES.items():
        pool = clean([r for r in v7 if r['split'] == 'train' and r.get('language') == language and r.get('source') != 'make_websites'
                      and names_language(language, r['messages'][0]['content'])])
        train += [{**r, 'part': f'language-{language}'} for r in rng.sample(pool, min(count, len(pool)))]
    for part, count in KEEP.items():
        pool = [r for r in v13 if r['split'] == 'train' and r.get('part') == part]
        train += rng.sample(pool, min(count, len(pool)))
    lus = read('lus-v1.jsonl')  # Kel, the user's own language (data/make_lus.py)
    train += [{**r, 'part': 'lus'} for r in lus if r['split'] == 'train']
    rng.shuffle(train)
    validation = [r for r in v13 if r['split'] != 'train'] + [r for r in v7 if r['split'] != 'train' and r.get('language') is None] + [r for r in lus if r['split'] != 'train']
    return train + validation, dropped


if __name__ == '__main__':
    rows, dropped = build()
    destination = ROOT / 'data/cide-v14.jsonl'
    # U+2028 and U+2029 are written escaped: readers of these files use str.splitlines, which breaks a row at them
    line = lambda r: json.dumps(r, ensure_ascii=False).replace('\u2028', '\\u2028').replace('\u2029', '\\u2029') + '\n'
    destination.write_text(''.join(line(r) for r in rows), encoding='utf-8')
    train = [r for r in rows if r['split'] == 'train']
    print(f'{destination}: {len(train):,} training rows, {len(rows) - len(train):,} held out; dropped {dict(dropped)}')
    print(dict(collections.Counter(r['part'] for r in train).most_common()))
    print('language', dict(collections.Counter(r.get('language') for r in train).most_common()))
    crud = sum('crud' in (r.get('family') or '') for r in train)
    print(f'CRUD rows {crud:,}; signature rows {sum(r.get("family") == "exact_signature" for r in train):,}; saas rows {sum("saas" in (r.get("family") or "") for r in train):,}')
