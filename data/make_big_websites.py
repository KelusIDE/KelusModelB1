"""websites-big-v1: long, complete single-file web apps for Cide 3 (2026-10-30). websites-big-v2 (2026-11-02, --plan):
the same pages with a numbered plan before the code ("It has 5 parts: 1. Reading list ...") and a "2/5 reading list"
comment before each part's HTML and script. Cide 3 lost track of which features it still owed: it wrote one
twice (a second stopwatch, the FAQ twice) and stopped before others. With the plan it can see what is done and what
is left. The prompts and held-out themes are unchanged, so websites-big-v1's held-out prompts still test both.

data/make_websites.py's pages have one or two features and fit Cide 2's 2,048-token window. These have four to eight
of the same seventeen features in one page (a navigation link for each), so a page runs to 2,500-3,900 tokens: what
Cide 3's 4,096-token window is for. The features come from make_websites.py unchanged, so each one is the code that
data/check_websites.mjs clicks through in a browser. The whole script must also parse (`node --check`, which runs
nothing), so two features never declare the same name. HELD_OUT_THEMES stay out of training, as in websites-v1.

    .venv/bin/python data/make_big_websites.py          # writes data/websites-big-v1.jsonl
    .venv/bin/python data/make_big_websites.py --plan   # writes data/websites-big-v2.jsonl
"""
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'data'))
from kelus.model import Model  # noqa: E402
from make_websites import (BASE_CSS, FEATURES, FONTS, HELD_OUT_THEMES, INTROS, OUTROS, THEMES, check)  # noqa: E402

TRAIN_EXAMPLES, VALIDATION_EXAMPLES = 9_000, 60
MAX_TOKENS = 3_950
PLAN_MAX_TOKENS = 4_080  # the plan adds ~150-250 tokens; the window is 4,096
TOKENIZER = ROOT / 'checkpoints/kani-otter-4-8-release.npz'

ASKS = [
    'Build a complete web app for {name}, a {kind}. It needs all of these: {features} Use one HTML file with inline CSS and JavaScript.',
    'Make me a big website for my {kind} called {name}, with a section for each of these. {features}',
    'Create a full single-page site for {name} ({kind}) with lots of features. {features} Put everything in one HTML file.',
    'I need a feature-packed homepage for {name}, a {kind}. {features} Give me the whole HTML file.',
    'Code a large website for {name} ({kind}). Include every one of these features: {features} No external libraries.',
    'write a full html site for a {kind} named {name} with all of this: {features} css and js inside the file pls',
    'Can you build {name} a complete website? They are a {kind}. Features: {features} Make sure all the JavaScript works.',
    'Write the entire website for {name}, a {kind}, as one file. {features} Add a navigation link for each section.',
]


# A feature as the request names it ("reading list", "kilometers to miles converter"), for the plan.
PHRASE = re.compile(r"(?:Add|Include|It needs|I need|I want|Show) (?:an? |the )?(.+?)(?: where| that| which| with| so | on the page|:| \(| from| in tabs|\.)")
CLEARER = {'button': 'random tip button', '"Book now" button': '"Book now" popup', 'menu, prices and location': 'menu, prices and location tabs',
           'small menu': 'small menu with a cart', 'message box': 'message box with a character counter'}


def plan_name(ask):
    name = PHRASE.match(ask).group(1)
    return CLEARER.get(name, name)


def section_id(html):
    return html.split('"')[1] if html.startswith('<section') else None


def label(identifier):
    return identifier.removesuffix('-section').replace('-', ' ').title()


def big_page(theme, features, rng, plan=False):
    name, kind, tagline, perks, (bg, surface, text, accent), icon = theme
    sections = [feature(theme, rng) for feature in features]
    if plan:  # "2/5 reading list" before each feature's HTML and script (CSS stays unmarked: the window is full)
        marks = [f'{n}/{len(sections)} {plan_name(s[0])}' for n, s in enumerate(sections, 1)]
        sections = [(ask, f'<!-- {mark} -->\n{html}', css, f'// {mark}\n{js}') for mark, (ask, html, css, js) in zip(marks, sections)]
    ids = [section_id(s[1].split('\n', 1)[1] if plan else s[1]) for s in sections]
    nav = ''.join(f'<a href="#{i}">{label(i)}</a>' for i in ids if i)
    perk_cards = '\n'.join(f'    <div class="card"><h3>{p}</h3></div>' for p in perks)
    css = BASE_CSS.format(bg=bg, surface=surface, text=text, accent=accent, font=rng.choice(FONTS)) + '\n' + '\n'.join(s[2] for s in sections)
    css += '\nnav { display: flex; flex-wrap: wrap; gap: 0.25rem; }'
    script = '\n\n'.join(s[3] for s in sections) + "\n\ndocument.getElementById('year').textContent = new Date().getFullYear();"
    first = next((i for i in ids if i), 'features')
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{name}</title>
<style>
{css}
</style>
</head>
<body>
<header>
  <span class="logo">{icon} {name}</span>
  <nav>{nav}</nav>
</header>
<section class="hero">
  <h1>{name}</h1>
  <p>{tagline}</p>
  <a href="#{first}"><button type="button">{rng.choice(['Get started', 'Take a look', 'Try it now', 'See more'])}</button></a>
</section>
<main>
  <section id="features" class="features">
{perk_cards}
  </section>
{chr(10).join(s[1] for s in sections)}
</main>
<footer>&copy; <span id="year"></span> {name}</footer>
<script>
{script}
</script>
</body>
</html>'''
    return [s[0] for s in sections], html, script


def parses(script, cache={}):
    if script not in cache:
        with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False) as handle:
            handle.write(script)
        try:
            cache[script] = subprocess.run(['node', '--check', handle.name], capture_output=True, timeout=30).returncode == 0
        finally:
            Path(handle.name).unlink(missing_ok=True)
    return cache[script]


def example(theme, rng, split, tokenizer, plan=False):
    features = rng.sample(FEATURES, rng.randint(4, 8))
    asks, html, script = big_page(theme, features, rng, plan)
    name, kind = theme[0], theme[1]
    prompt = rng.choice(ASKS).format(name=name, kind=kind, features=' '.join(asks))
    if rng.random() < 0.15:
        prompt = prompt.lower()
    reply = rng.choice(INTROS).format(name=name)
    if plan:
        reply += f'\n\nIt has {len(asks)} parts:\n' + '\n'.join(f'{n}. {plan_name(a)[0].upper()}{plan_name(a)[1:]}' for n, a in enumerate(asks, 1))
    reply += '\n\n```html\n' + html + '\n```'
    outro = rng.choice(OUTROS)
    if outro:
        reply += '\n\n' + outro
    if not check(html) or len(tokenizer.encode(prompt)) + len(tokenizer.encode(reply)) + 8 > (PLAN_MAX_TOKENS if plan else MAX_TOKENS) or not parses(script):
        return None
    return {'split': split, 'family': 'website_big', 'source': 'make_big_websites', 'language': 'html',
            'features': [f.__name__ for f in features], 'messages': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': reply}]}


def build(seed=2420, plan=False):
    rng = random.Random(seed)
    tokenizer = Model.load(TOKENIZER)[0].tokenizer
    train_themes = [t for t in THEMES if t[0] not in HELD_OUT_THEMES]
    held_out = [t for t in THEMES if t[0] in HELD_OUT_THEMES]
    rows = []
    for split, themes, wanted in (('train', train_themes, TRAIN_EXAMPLES), ('validation', held_out, VALIDATION_EXAMPLES)):
        made, seen, tries = 0, set(), 0
        while made < wanted:
            tries += 1
            row = example(rng.choice(themes), rng, split, tokenizer, plan)
            if row and row['messages'][1]['content'] not in seen:
                seen.add(row['messages'][1]['content'])
                rows.append(row)
                made += 1
        print(f'{split}: {made} pages from {tries} tries', file=sys.stderr)
    return rows, tokenizer


if __name__ == '__main__':
    plan = '--plan' in sys.argv
    rows, tokenizer = build(plan=plan)
    destination = ROOT / ('data/websites-big-v2.jsonl' if plan else 'data/websites-big-v1.jsonl')
    destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    sizes = sorted(len(tokenizer.encode(r['messages'][1]['content'])) for r in rows[:400])
    features = sorted(len(r['features']) for r in rows)
    print(f'{destination}: {len(rows):,} sites; reply tokens median {sizes[len(sizes) // 2]:,}, longest {sizes[-1]:,} (first 400); '
          f'features median {features[len(features) // 2]}')
