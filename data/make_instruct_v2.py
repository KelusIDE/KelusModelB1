"""Instruction curriculum v2: teach copying and composition instead of memorized names.

v1 built function names from 17 verbs x 20 nouns and expressions from ~35 templates;
at step 1000 the model replaced unseen names with memorized ones (0/60 held out).
v2 keeps all of v1 and adds procedurally generated specifications:

  * names and parameters drawn from ~1000 words found in this repository; validation
    uses only words that never appear in a training name, so passing requires copying
  * expressions are random trees of + - * / % over parameters and constants, each
    paired with a compositional English description ("the sum of x and the product
    of y and 4") or stated verbatim ("returns x + y * 4")

The readiness checks stay held out: no sum of exactly three parameters, no
x * x * x, no p * (1 + q), no multiplication by 3, and none of their names.

    python3 data/make_instruct_v2.py   # writes data/instruct-v2.jsonl
"""
import ast
import hashlib
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'data'))
from kelus.evaluate import code_result  # noqa: E402
import make_instruct as v1  # noqa: E402

RESERVED = set('and as assert async await break class continue def del elif else except finally for from global if import in is lambda '
               'nonlocal not or pass raise return try while with yield none true false self cls len sum max min print int str float list dict '
               'set tuple range type object open input file'.split())
HELD_OUT_WORDS = {'cube', 'triple', 'three', 'counter'}
OPS = {'+': ast.Add, '-': ast.Sub, '*': ast.Mult, '/': ast.Div, '%': ast.Mod}
TEMPLATES = [
    'Write a Python function {name}({params}) that returns {expr}.',
    'Write a Python function {name}({params}) that returns {expr}. Return only the function.',
    'Write a Python function {name}({params}) that returns {desc}.',
    'Write a Python function {name}({params}) that returns {desc}. Return only the function.',
    'Write a Python function {name}({params}) that returns {desc}. Use exactly that function name. Return only the function.',
    'Create a Python function called {name} that takes {words} and returns {desc}.',
    'Define {name}({params}) in Python. It returns {expr}.',
    'I need a Python function named {name}. It takes {words} and returns {desc}.',
    'python: {name}({params}) returns {expr}',
    'Write {name}({params}) in Python so it returns {desc}. Return only the function.',
]


def vocabulary():
    words = set()
    for path in list((ROOT / 'kelus').glob('*.py')) + list(ROOT.glob('*.md')) + list((ROOT / 'data').glob('make_*.py')):
        words.update(re.findall(r'\b[a-z]{3,9}\b', path.read_text(errors='replace').lower()))
    words = {w for w in words - RESERVED if not any(held in w for held in HELD_OUT_WORDS)}
    ordered = sorted(words)

    validation = {w for w in ordered if int(hashlib.sha256(w.encode()).hexdigest(), 16) % 6 == 0}
    return [w for w in ordered if w not in validation], sorted(validation)


def tree(rng, params, depth):
    if depth == 0 or rng.random() < 0.25:
        if rng.random() < 0.7:
            return ('var', rng.choice(params))
        return ('num', rng.randint(1, 12))
    return (rng.choice(list(OPS)), tree(rng, params, depth - 1), tree(rng, params, depth - 1))


def to_ast(node):
    if node[0] == 'var':
        return ast.Name(node[1])
    if node[0] == 'num':
        return ast.Constant(node[1])
    return ast.BinOp(to_ast(node[1]), OPS[node[0]](), to_ast(node[2]))


def describe(node):
    kind = node[0]
    if kind == 'var':
        return node[1]
    if kind == 'num':
        return str(node[1])
    a, b = describe(node[1]), describe(node[2])
    return {'+': f'the sum of {a} and {b}', '-': f'{a} minus {b}', '*': f'the product of {a} and {b}',
            '/': f'{a} divided by {b}', '%': f'the remainder of {a} divided by {b}'}[kind]


def variables(node):
    return {node[1]} if node[0] == 'var' else set() if node[0] == 'num' else variables(node[1]) | variables(node[2])


def probe_like(node, params):
    """Shapes the readiness checks ask for; never trained."""
    def flat(n, op):
        return flat(n[1], op) + flat(n[2], op) if n[0] == op else [n]
    sums, products = flat(node, '+'), flat(node, '*')
    if len(params) == 3 and len(sums) == 3 and all(n[0] == 'var' for n in sums):
        return True
    if len(products) == 3 and all(n == ('var', params[0]) for n in products):
        return True
    if any(n == ('num', 3) for n in products) and len(products) == 2:
        return True
    if node[0] == '*' and node[2][0] == '+' and ('num', 1) in (node[2][1], node[2][2]):
        return True
    return False


def spec(rng, words, split):
    arity = rng.choice([1, 1, 2, 2, 2, 3])
    params = rng.sample(words, arity) if rng.random() < 0.6 else rng.sample(['x', 'y', 'z', 'a', 'b', 'c', 'n', 'm', 'p', 'q'], arity)
    node = tree(rng, params, rng.choice([1, 1, 2, 2, 3]))
    if node[0] in ('var', 'num') or variables(node) != set(params) or probe_like(node, params):
        return None
    name = '_'.join(rng.sample(words, rng.choice([1, 2, 2, 3])))
    if name in v1.HELD_OUT_NAMES or name in RESERVED:
        return None
    expr = ast.unparse(to_ast(node))
    answer = f"```python\ndef {name}({', '.join(params)}):\n    return {expr}\n```"
    trials, attempts = [], 0
    while len(trials) < 3 and attempts < 20:
        attempts += 1
        inputs = tuple(rng.randint(-9, 12) for _ in params)
        try:
            value = eval(expr, {}, dict(zip(params, inputs)))
        except ZeroDivisionError:
            continue
        if abs(value) < 1e6:
            trials.append((inputs, value))
    if len(trials) < 3 or not code_result(answer, name, trials):
        return None
    prompt = rng.choice(TEMPLATES).format(name=name, params=', '.join(params), expr=expr, desc=describe(node),
                                          words=v1.words(params))
    return {'split': split, 'family': 'python_compose', 'messages': [
        {'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': answer}]}


def build(seed=2027):
    rng = random.Random(seed)
    rows = v1.build()
    train_words, validation_words = vocabulary()
    for count, split, words in ((30000, 'train', train_words), (700, 'validation', validation_words)):
        made = 0
        while made < count:
            record = spec(rng, words, split)
            if record:
                rows.append(record)
                made += 1
    unique = {}
    for row in rows:
        unique.setdefault(json.dumps(row['messages'], sort_keys=True), row)
    rows = list(unique.values())
    rng.shuffle(rows)
    return rows, len(train_words), len(validation_words)


if __name__ == '__main__':
    rows, train_words, validation_words = build()
    destination = ROOT / 'data/instruct-v2.jsonl'
    destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    counts = {}
    for row in rows:
        key = f"{row['family']}/{row['split']}"
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({'path': str(destination), 'records': len(rows), 'name_words': {'train': train_words, 'validation': validation_words},
                      'counts': dict(sorted(counts.items()))}, indent=1))
