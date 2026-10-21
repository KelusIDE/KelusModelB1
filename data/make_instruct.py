"""Instruction curriculum for Kelus Kani B2:2F, trained from the foundation checkpoint.

Families (each record carries "family" for provenance):
  practiced          curriculum v3 (conversations-v3.jsonl), identity replies renamed
  python_spec        Python functions written to a stated name, parameters, and behavior
  javascript_spec    the same specifications in JavaScript
  website            short single-file HTML pages with inline CSS and JavaScript

Validation holds out whole operation templates and function names, so its loss
measures specifications the model has not seen. The development readiness
checks in kelus/code_readiness.py stay out of training entirely: their prompts,
function names, and behaviors (adding three arguments, cubing, price * (1 + rate),
multiplying by three) and counter pages are never generated. Every Python answer
is confirmed by the same bounded AST checker the evaluations use.

    python3 data/make_instruct.py   # writes data/instruct-v1.jsonl
"""
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kelus.evaluate import code_result  # noqa: E402  (the checker, not the probes)

NAME = 'Kelus Kani B2:2F'
RENAMES = {
    "Hi! I'm Kelus Model B1. What would you like to build?": f"Hi! I'm {NAME}. What would you like to build?",
    "I'm Kelus Model B1, a small model trained locally from scratch for basic chat and code.":
        f"I'm {NAME}, a small model trained locally from scratch for basic chat and code.",
}
HELD_OUT_NAMES = {'add_three', 'cube', 'final_price', 'triple_score'}
NUMBER_WORDS = {2: 'two', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten'}

# (expression, description). {0} {1} {2} are parameters; {k} {c} are constants.
OPS = {
    1: [('{0} + {k}', 'adds {k} to {0}'), ('{0} - {k}', 'subtracts {k} from {0}'),
        ('{0} * {k}', 'multiplies {0} by {k}'), ('{0} / {k}', 'divides {0} by {k}'),
        ('{0} % {k}', 'returns the remainder when {0} is divided by {k}'), ('{0} * {0}', 'returns {0} multiplied by itself'),
        ('-{0}', 'returns the negative of {0}'), ('{0} * 2', 'doubles {0}'), ('{0} / 2', 'returns half of {0}'),
        ('{0} * {k} + {c}', 'multiplies {0} by {k} and then adds {c}'), ('({0} + {k}) * {c}', 'adds {k} to {0} and multiplies the result by {c}'),
        ('{0} == {k}', 'returns whether {0} equals {k}'), ('{0} % 2 == 0', 'returns whether {0} is even'),
        ('{0} - {k} * {c}', 'subtracts {k} times {c} from {0}'), ('{0} * {0} + {k}', 'squares {0} and adds {k}')],
    2: [('{0} + {1}', 'adds {0} and {1}'), ('{0} - {1}', 'subtracts {1} from {0}'), ('{0} * {1}', 'multiplies {0} by {1}'),
        ('{0} / {1}', 'divides {0} by {1}'), ('{0} % {1}', 'returns the remainder when {0} is divided by {1}'),
        ('({0} + {1}) / 2', 'returns the average of {0} and {1}'), ('{0} * {1} / 2', 'returns half of {0} times {1}'),
        ('{0} * {1} + {k}', 'multiplies {0} by {1} and adds {k}'), ('({0} - {1}) * {k}', 'subtracts {1} from {0} and multiplies by {k}'),
        ('{0} == {1}', 'returns whether {0} equals {1}'), ('{0} != {1}', 'returns whether {0} is different from {1}'),
        ('{0} * {k} + {1}', 'multiplies {0} by {k} and adds {1}'), ('{0} * {0} + {1} * {1}', 'adds the squares of {0} and {1}'),
        ('({0} + {1}) * ({0} - {1})', 'multiplies the sum of {0} and {1} by their difference')],
    3: [('{0} * {1} * {2}', 'multiplies all three arguments'), ('{0} + {1} - {2}', 'adds {0} and {1}, then subtracts {2}'),
        ('({0} + {1}) * {2}', 'adds {0} and {1}, then multiplies by {2}'), ('{0} * {1} + {2}', 'multiplies {0} by {1} and adds {2}'),
        ('{0} - {1} - {2}', 'subtracts {1} and {2} from {0}'), ('{0} + {1} * {2}', 'adds {0} to {1} times {2}'),
        ('({0} - {1}) * {2}', 'subtracts {1} from {0}, then multiplies by {2}')],
}
# Whole templates kept for validation only: specifications the model never trains on.
VALIDATION_OPS = {'{0} - {k} * {c}', '{0} * {0} + {k}', '({0} + {1}) * ({0} - {1})', '{0} + {1} * {2}', '({0} - {1}) * {2}'}
PARAMS = {
    1: [('x',), ('n',), ('value',), ('score',), ('points',), ('total',), ('size',), ('length',), ('amount',), ('count',),
        ('number',), ('speed',), ('distance',), ('age',), ('level',), ('temperature',), ('weight',), ('minutes',)],
    2: [('a', 'b'), ('x', 'y'), ('width', 'height'), ('base', 'bonus'), ('total', 'count'), ('left', 'right'), ('first', 'second'),
        ('start', 'step'), ('cost', 'quantity'), ('hours', 'rate'), ('score', 'weight'), ('m', 'n'), ('low', 'high'), ('old', 'new')],
    3: [('a', 'b', 'c'), ('x', 'y', 'z'), ('length', 'width', 'height'), ('base', 'bonus', 'penalty'), ('first', 'second', 'third'),
        ('p', 'q', 'r'), ('start', 'step', 'count'), ('cost', 'quantity', 'discount')],
}
VERBS = ['compute', 'calculate', 'get', 'make', 'find', 'apply', 'update', 'adjust', 'process', 'convert', 'scale', 'combine',
         'shift', 'measure', 'count', 'build', 'check', 'mix', 'boost', 'reduce']
NOUNS = ['value', 'result', 'total', 'score', 'points', 'amount', 'number', 'size', 'output', 'answer', 'level', 'area',
         'cost', 'speed', 'reading', 'balance', 'offset', 'ratio', 'sum', 'count']
VALIDATION_VERBS = {'measure', 'boost', 'reduce'}
PY_TEMPLATES = [
    'Write a Python function {name}({params}) that {desc}.',
    'Write a Python function {name}({params}) that returns {expr}.',
    'Write a Python function {name}({params}) that {desc}. Return only the function.',
    'Write a Python function {name}({params}) that returns {expr}. Return only the function.',
    'Write a Python function {name}({params}) that {desc}. Use exactly that function name. Return only the function.',
    'Create a Python function called {name} that takes {words} and {desc_s}.',
    'Define {name}({params}) in Python so that it returns {expr}.',
    'python function {name}({params}) that {desc}',
    'I need a Python function named {name}. It takes {words} and {desc_s}.',
    'Write {name}({params}) in Python. It should return {expr}.',
]
JS_TEMPLATES = [
    'Write a JavaScript function {name}({params}) that {desc}.',
    'Write a JavaScript function {name}({params}) that returns {expr}. Return only the function.',
    'Create a JavaScript function called {name} that takes {words} and {desc_s}.',
    'js function {name}({params}) that {desc}',
]


def words(params):
    return params[0] if len(params) == 1 else ', '.join(params[:-1]) + ' and ' + params[-1]


def spec(rng, arity, op, validation):
    expression_template, description_template = op
    params = rng.choice(PARAMS[arity])
    k = rng.choice([2, 4, 5, 6, 7, 8, 9, 10])  # 3 is held out with triple_score
    c = rng.choice([1, 2, 4, 5, 10, 100])
    expr = expression_template.replace('{k}', str(k)).replace('{c}', str(c)).format(*params)
    desc = description_template.replace('{k}', str(k)).replace('{c}', str(c)).format(*params)
    verbs = sorted(VALIDATION_VERBS) if validation else [v for v in VERBS if v not in VALIDATION_VERBS]
    name = f'{rng.choice(verbs)}_{rng.choice(NOUNS)}'
    if name in HELD_OUT_NAMES or name.startswith(('triple', 'cube')):
        return None
    return {'name': name, 'params': params, 'expr': expr, 'desc': desc}


def trials(rng, item):
    params = item['params']
    cases = []
    while len(cases) < 3:
        inputs = tuple(rng.randint(-9, 12) for _ in params)
        try:
            expected = eval(item['expr'], {}, dict(zip(params, inputs)))  # our own template, never model output
        except ZeroDivisionError:
            continue
        cases.append((inputs, expected))
    return cases


def python_record(rng, item, split):
    params = ', '.join(item['params'])
    answer = f"```python\ndef {item['name']}({params}):\n    return {item['expr']}\n```"
    if not code_result(answer, item['name'], trials(rng, item)):
        raise AssertionError(f'Generated answer fails the checker: {answer}')
    prompt = rng.choice(PY_TEMPLATES).format(name=item['name'], params=params, desc=item['desc'], expr=item['expr'],
                                             words=words(item['params']), desc_s=item['desc'])
    return {'split': split, 'family': 'python_spec', 'messages': [
        {'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': answer}]}


def javascript_record(rng, item, split):
    params = ', '.join(item['params'])
    expr = item['expr'].replace('==', '===').replace('!===', '!==')
    name = ''.join(part if i == 0 else part.capitalize() for i, part in enumerate(item['name'].split('_')))
    answer = f'```javascript\nfunction {name}({params}) {{\n  return {expr};\n}}\n```'
    prompt = rng.choice(JS_TEMPLATES).format(name=name, params=params, desc=item['desc'], expr=expr,
                                             words=words(item['params']), desc_s=item['desc'])
    return {'split': split, 'family': 'javascript_spec', 'messages': [
        {'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': answer}]}


TITLES = ['Color Switcher', 'Greeting Page', 'Mood Board', 'Quote Box', 'Day Planner', 'Theme Toggle', 'Message Board',
          'Weather Note', 'Fruit List', 'Hello Page', 'Secret Message', 'Light Switch', 'Shopping List', 'Book Notes',
          'Pet Corner', 'Study Plan', 'Travel Ideas', 'Snack Picker', 'Garden Log', 'Music Notes']
MESSAGES = ['Hello there!', 'Have a great day!', 'You did it!', 'Welcome back!', 'Keep going!', 'Nice work!', 'See you soon!']
COLORS = ['lightblue', 'lightgreen', 'lavender', 'peachpuff', 'khaki', 'mistyrose', 'honeydew', 'lightyellow']
ITEMS = ['Apple', 'Bread', 'Milk', 'Tea', 'Notebook', 'Pencil', 'Banana', 'Rice']
FONTS = ['sans-serif', 'Arial, sans-serif', 'Georgia, serif', 'system-ui, sans-serif', 'Verdana, sans-serif']
# (prompt phrase, extra CSS, body elements, script). Counters are deliberately absent.
FEATURES = [
    ('a paragraph and a button that changes the paragraph text to "{msg}"', '',
     '<p id="text">Click the button.</p>\n<button onclick="change()">{label}</button>',
     "function change() {{\n  document.getElementById('text').textContent = '{msg}';\n}}"),
    ('a button that changes the page background to {color}', '',
     '<button onclick="paint()">{label}</button>',
     "function paint() {{\n  document.body.style.background = '{color}';\n}}"),
    ('a hidden message and a button that shows it', '.hidden {{ display: none; }}\n',
     '<p id="secret" class="hidden">{msg}</p>\n<button onclick="reveal()">{label}</button>',
     "function reveal() {{\n  document.getElementById('secret').classList.remove('hidden');\n}}"),
    ('a list and a button that adds "{item}" to the list', '',
     '<ul id="list"></ul>\n<button onclick="add()">{label}</button>',
     "function add() {{\n  document.getElementById('list').innerHTML += '<li>{item}</li>';\n}}"),
    ('a text box and a button that greets the name typed in it', '',
     '<input id="name" placeholder="Your name">\n<button onclick="greet()">{label}</button>\n<p id="out"></p>',
     "function greet() {{\n  const name = document.getElementById('name').value;\n  document.getElementById('out').textContent = 'Hi, ' + name;\n}}"),
    ('a button that turns the heading {color}', '',
     '<button onclick="tint()">{label}</button>',
     "function tint() {{\n  document.querySelector('h1').style.color = '{color}';\n}}"),
    ('a button that toggles dark mode', '.dark {{ background: #222; color: #eee; }}\n',
     '<button onclick="toggle()">{label}</button>',
     "function toggle() {{\n  document.body.classList.toggle('dark');\n}}"),
    ('a button that shows an alert saying "{msg}"', '',
     '<button onclick="say()">{label}</button>',
     "function say() {{\n  alert('{msg}');\n}}"),
]
VALIDATION_FEATURES = {5}  # the heading-color page is only ever validation
LABELS = [['Change text', 'Click me'], ['Change color', 'Paint'], ['Show message', 'Reveal'], ['Add item', 'Add'],
          ['Greet', 'Say hi'], ['Color heading', 'Change color'], ['Dark mode', 'Toggle theme'], ['Show alert', 'Say it']]
WEB_TEMPLATES = [
    'Build a complete single-file HTML website with inline CSS and JavaScript. Include a heading and {feature}. Return the complete HTML document.',
    'Make a single HTML page with a heading and {feature}. Put the CSS and JavaScript inline.',
    'Create a simple web page titled "{title}" with {feature}.',
    'Write an HTML file with inline CSS and JS: a heading and {feature}.',
    'build a website with {feature}',
]


def website_record(rng, index, split):
    phrase, css, body, script = FEATURES[index]
    values = {'msg': rng.choice(MESSAGES), 'color': rng.choice(COLORS), 'item': rng.choice(ITEMS), 'label': rng.choice(LABELS[index])}
    title = rng.choice(TITLES)
    feature = phrase.format(**values)
    style = f"body {{ font-family: {rng.choice(FONTS)}; text-align: center; }}\n" + css.format()
    page = ('<!doctype html>\n<html>\n<head>\n<style>\n{style}</style>\n</head>\n<body>\n'
            '<h1>{title}</h1>\n{body}\n<script>\n{script}\n</script>\n</body>\n</html>').format(
                title=title, style=style, body=body.format(**values), script=script.format(**values))
    prompt = rng.choice(WEB_TEMPLATES).format(feature=feature, title=title)
    assert 'counter' not in (prompt + page).lower()
    return {'split': split, 'family': 'website', 'messages': [
        {'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': f'```html\n{page}\n```'}]}


def practiced():
    records = []
    for line in (ROOT / 'data/conversations-v3.jsonl').read_text(encoding='utf-8').splitlines():
        record = json.loads(line)
        for message in record['messages']:
            if message['role'] == 'assistant':
                message['content'] = RENAMES.get(message['content'], message['content'])
        records.append({'split': record['split'], 'family': 'practiced', 'messages': record['messages']})
    return records


def build(seed=2026):
    rng = random.Random(seed)
    records = practiced()
    for count, split in ((16000, 'train'), (900, 'validation')):
        made = 0
        while made < count:
            arity = rng.choice([1, 1, 2, 2, 3])
            ops = [op for op in OPS[arity] if (op[0] in VALIDATION_OPS) == (split == 'validation')]
            if not ops:
                continue
            item = spec(rng, arity, rng.choice(ops), split == 'validation')
            if item is None:
                continue
            records.append(python_record(rng, item, split))
            if made % 3 == 0:
                records.append(javascript_record(rng, item, split))
            made += 1
    for count, split in ((5000, 'train'), (250, 'validation')):
        indices = [i for i in range(len(FEATURES)) if (i in VALIDATION_FEATURES) == (split == 'validation')]
        for _ in range(count):
            records.append(website_record(rng, rng.choice(indices), split))
    # Drop repeated conversations, then keep new-family validation prompts out of training.
    # (Curriculum v3 was decontaminated by its own generator and is left as it is.)
    unique = {}
    for record in records:
        unique.setdefault(json.dumps(record['messages'], sort_keys=True), record)
    validation_prompts = {r['messages'][0]['content'] for r in unique.values() if r['split'] == 'validation' and r['family'] != 'practiced'}
    rows = [r for r in unique.values()
            if r['split'] == 'validation' or r['family'] == 'practiced' or r['messages'][0]['content'] not in validation_prompts]
    rng.shuffle(rows)
    return rows


if __name__ == '__main__':
    rows = build()
    destination = ROOT / 'data/instruct-v1.jsonl'
    destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    counts = {}
    for row in rows:
        key = f"{row['family']}/{row['split']}"
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({'path': str(destination), 'records': len(rows), 'counts': dict(sorted(counts.items()))}, indent=1))
