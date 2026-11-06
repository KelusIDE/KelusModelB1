"""cide-v13: Cide 4.4's training data (2026-11-03): Python first, then JavaScript and HTML.

The user asked for Cide to be "expert at js and html and especially python". Cide 2-4.3 learned Python from 8,000
examples, all cut to 3,000 characters for Kani's old 1,024-token window; Cide now reads 4,096 tokens, and far more
Python is on disk. This stage adds:
  - Python: bigcode/self-oss-instruct-sc2-exec-filter-50k (ODC-BY; StarCoder2-Instruct's data, every response kept
    only after its own tests passed when its authors ran them), plus Magicoder OSS-Instruct and Evol-Instruct Python
    up to 10,000 characters;
  - JavaScript and TypeScript from Magicoder; HTML and CSS from Magicoder Evol-Instruct;
  - replay of Cide's own web data (plain requests, big pages with plans, follow-up chats) and the other 16 languages,
    so Cide 4.3's skills stay.
Checks are static, nothing is run: every ```python block must parse (ast.parse), every JavaScript block must pass
`node --check`. Rows sharing a 13-word run with HumanEval or MBPP (any split) are dropped, so those can measure Cide
later, and so are rows whose request matches any held-out request in Cide's earlier data (kelus.code_language_eval
reads cide-v6's). Downloads live in data/cide-v13-sources/ (git-ignored); run with -I, as the sources are untrusted.

    .venv/bin/python -I data/make_cide_v13.py   # writes data/cide-v13.jsonl
"""
import ast
import collections
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pyarrow.parquet as pq  # noqa: E402

from kelus.model import Model  # noqa: E402
from kelus.tokenizer import conversation  # noqa: E402

SOURCES = ROOT / 'data/cide-v13-sources'
OLD_SOURCES = ROOT / 'data/cide-v6-sources'
TOKENIZER = ROOT / 'checkpoints/cide-4-2-release.npz'
MAX_CHARACTERS, MAX_TOKENS = 10_000, 3_500
HELD_OUT = 40
# training rows per part; Python is about half
PARTS = {'python-self-oss': 12_000, 'python-magicoder': 8_000, 'javascript': 5_000, 'typescript': 1_000, 'html': 2_000,
         'css': 500, 'web-basics': 2_000, 'web-big': 1_000, 'web-followups': 3_000, 'other-languages': 2_500}
FENCE = re.compile(r'```\s*([A-Za-z+#]*)[^\n]*\n(.*?)(?:```|$)', re.S)
ALIASES = {'js': 'javascript', 'jsx': 'javascript', 'ts': 'typescript', 'tsx': 'typescript', 'py': 'python', 'python3': 'python'}


def words(text):
    return re.findall(r'[a-z0-9_]+', text.lower())


def grams(text, n=13):
    w = words(text)
    return {hash(' '.join(w[i:i + n])) for i in range(len(w) - n + 1)}


def benchmark_grams():
    """13-word runs of HumanEval and MBPP (prompts, solutions and tests, every split)."""
    found = set()
    for row in pq.read_table(SOURCES / 'decontam/humaneval.parquet').to_pylist():
        found |= grams(row['prompt']) | grams(row['canonical_solution']) | grams(row['test'])
    for split in ('train', 'test', 'validation', 'prompt'):
        for row in pq.read_table(SOURCES / f'decontam/mbpp-{split}.parquet').to_pylist():
            found |= grams(row['text'] + ' ' + row['code']) | grams(' '.join(row['test_list']))
    return found


def held_out_requests():
    """Requests held out anywhere in Cide's earlier data: never train on them."""
    found = set()
    for name in ('cide-v6.jsonl', 'cide-v7.jsonl', 'cide-v11.jsonl', 'cide-v12.jsonl'):
        for line in (ROOT / 'data' / name).read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if row['split'] != 'train':
                found.add(' '.join(words(row['messages'][0]['content'])))
    return found


def blocks(reply):
    return [(ALIASES.get(lang.lower(), lang.lower()), code) for lang, code in FENCE.findall(reply)]


def python_ok(reply):
    code = [c for lang, c in blocks(reply) if lang == 'python']
    if not code:
        return False
    try:
        for c in code:
            ast.parse(c)
    except (SyntaxError, ValueError):
        return False
    return True


def javascript_parses(code):
    """`node --check` parses without running; modules (import/export) are checked as .mjs."""
    module = re.search(r'^\s*(import|export)\b', code, re.M)
    with tempfile.NamedTemporaryFile('w', suffix='.mjs' if module else '.js', delete=False) as handle:
        handle.write(code)
    try:
        return subprocess.run(['node', '--check', handle.name], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False
    finally:
        Path(handle.name).unlink(missing_ok=True)


def row(prompt, reply, lang, source, part):
    prompt, reply = prompt.strip(), reply.strip()
    if not prompt or not reply or len(prompt) + len(reply) > MAX_CHARACTERS:
        return None
    return {'family': f'code_{lang}', 'source': source, 'language': lang, 'part': part,
            'messages': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': reply}]}


def external():
    """(part, row) from the downloaded sets, before checks."""
    for r in pq.read_table(SOURCES / 'self-oss/train.parquet', columns=['instruction', 'response']).to_pylist():
        yield 'python-self-oss', row(r['instruction'], r['response'], 'python', 'self-oss-instruct-sc2', 'python-self-oss')
    for line in (OLD_SOURCES / 'magicoder-oss/data-oss_instruct-decontaminated.jsonl').open(encoding='utf-8'):
        r = json.loads(line)
        lang = ALIASES.get(r['lang'].lower(), r['lang'].lower())
        part = 'python-magicoder' if lang == 'python' else lang
        if part in PARTS:
            yield part, row(r['problem'], r['solution'], lang, 'magicoder-oss-instruct', part)
    for line in (OLD_SOURCES / 'magicoder-evol/data-evol_instruct-decontaminated.jsonl').open(encoding='utf-8'):
        r = json.loads(line)
        found = blocks(r['response'])
        lang = found[0][0] if found else None
        part = 'python-magicoder' if lang == 'python' else lang
        if part in PARTS:
            yield part, row(r['instruction'], r['response'], lang, 'magicoder-evol-instruct', part)


def build(seed=2435):
    rng = random.Random(seed)
    tokenizer = Model.load(TOKENIZER)[0].tokenizer
    contaminated, held, seen = benchmark_grams(), held_out_requests(), set()
    dropped = collections.Counter()
    by_part = collections.defaultdict(list)
    for part, r in external():
        if not r:
            dropped['too long'] += 1
            continue
        key = ' '.join(words(r['messages'][0]['content']))
        if key in seen or key in held:
            dropped['repeat or held out elsewhere'] += 1
            continue
        seen.add(key)
        if grams(r['messages'][0]['content'] + ' ' + r['messages'][1]['content']) & contaminated:
            dropped['HumanEval/MBPP overlap'] += 1
            continue
        if r['language'] == 'python' and not python_ok(r['messages'][1]['content']):
            dropped['python does not parse'] += 1
            continue
        if r['language'] in ('html', 'css') and not blocks(r['messages'][1]['content']):
            dropped['no code block'] += 1
            continue
        by_part[part].append(r)
    # JavaScript: parse only the rows that can be used (node --check is the slow part)
    for part in ('javascript',):
        rng.shuffle(by_part[part])
        candidates = by_part[part][:int((PARTS[part] + HELD_OUT) * 1.4)]
        def ok(r):
            code = [c for lang, c in blocks(r['messages'][1]['content']) if lang == 'javascript']
            return bool(code) and all(javascript_parses(c) for c in code)
        with ThreadPoolExecutor(8) as pool:
            passed = list(pool.map(ok, candidates))
        dropped['javascript does not parse'] += passed.count(False)
        by_part[part] = [r for r, good in zip(candidates, passed) if good]
    train, validation = [], []
    for part, rows in sorted(by_part.items()):
        rng.shuffle(rows)
        rows = [r for r in rows[:(PARTS[part] + HELD_OUT) * 2]
                if len(conversation(r['messages'], tokenizer=tokenizer)[0]) <= MAX_TOKENS]
        validation += [{**r, 'split': 'validation'} for r in rows[:HELD_OUT]]
        train += [{**r, 'split': 'train'} for r in rows[HELD_OUT:HELD_OUT + PARTS[part]]]
    # replay of Cide's own data (already split; validation rows stay with their own files)
    replay = lambda name: [json.loads(line) for line in (ROOT / 'data' / name).read_text(encoding='utf-8').splitlines()]
    basics, big, followups, v7 = replay('web-basics-v1.jsonl'), replay('websites-big-v2.jsonl'), replay('cide-followups-v1.jsonl'), replay('cide-v7.jsonl')
    others = [r for r in v7 if r['split'] == 'train' and r.get('language') not in (None, 'python', 'javascript', 'typescript', 'html', 'css')]
    for part, rows in (('web-basics', basics), ('web-big', big), ('web-followups', followups), ('other-languages', others)):
        pool = [r for r in rows if r['split'] == 'train']
        train += [{**r, 'part': part} for r in rng.sample(pool, min(PARTS[part], len(pool)))]
    rng.shuffle(train)
    return train + validation, dropped


if __name__ == '__main__':
    rows, dropped = build()
    destination = ROOT / 'data/cide-v13.jsonl'
    # U+2028 and U+2029 are written escaped: readers of these files use str.splitlines, which breaks a row at them
    line = lambda r: json.dumps(r, ensure_ascii=False).replace('\u2028', '\\u2028').replace('\u2029', '\\u2029') + '\n'
    destination.write_text(''.join(line(r) for r in rows), encoding='utf-8')
    train = collections.Counter(r['part'] for r in rows if r['split'] == 'train')
    print(f'{destination}: {sum(train.values()):,} training rows ({", ".join(f"{p} {n:,}" for p, n in train.most_common())}), '
          f'{len(rows) - sum(train.values()):,} held out; dropped {dict(dropped)}')
