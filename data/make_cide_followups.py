"""cide-followups-v1: coding chats with more than one request, for Cide 4.3 (2026-11-03).

Cide 4.2 trained on 13,992 single requests and 8 two-turn chats, so an earlier page in the chat derailed the next
request: on the site, "Code me a Homepage website with html" after a login page gave one garbled sentence and no code,
and the site had to stop sending Cide the history. Every chat here starts with a page from data/make_web_basics.py and
goes on with one of:

- new: an unrelated request (another page, or code in another language from cide-v7) answered as if it came alone;
- edit: a change to the page ("make the buttons red", "change the title to ...", "add a contact form") answered with
  the whole updated page, everything else unchanged. Edits are made on the page text, the way a person would: one CSS
  property, the title, a footer, or a feature's HTML, CSS and script added at the end of <main>, <style> and <script>.

Chats are short enough that the site and the CLI keep the whole history (the prompt plus a third of the 4,096-token
window fits). Validation rows use held-out themes and wordings and list what the last page must contain (`expect`),
so kelus.website_eval can tell an applied edit from a copied page; data/check_websites.mjs clicks through a sample of
the last pages in a browser.

    .venv/bin/python data/make_cide_followups.py   # writes data/cide-followups-v1.jsonl
"""
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'data'))
from kelus.model import Model  # noqa: E402
from kelus.tokenizer import conversation  # noqa: E402
import make_web_basics as basics  # noqa: E402
from make_websites import FEATURES, HELD_OUT_THEMES, THEMES  # noqa: E402

TOKENIZER = ROOT / 'checkpoints/cide-4-2-release.npz'
CONTEXT = 4_096
PROMPT_MAX = CONTEXT - CONTEXT // 3  # site/cide.js codingContext keeps earlier turns while prompt + a third of the window fits
TRAIN_NEW, TRAIN_EDIT, VALIDATION_NEW, VALIDATION_EDIT = 3_000, 4_000, 12, 24

COLORS = {'red': '#dc2626', 'blue': '#2563eb', 'green': '#16a34a', 'purple': '#7c3aed', 'orange': '#ea580c',
          'pink': '#db2777', 'teal': '#0d9488', 'navy': '#1e3a8a', 'black': '#111111', 'gold': '#ca8a04'}
DARK, LIGHT = ('#111827', '#1f2937', '#f9fafb'), ('#ffffff', '#f3f4f6', '#111827')
FONTS = [('Georgia', 'Georgia, serif'), ('Arial', 'Arial, sans-serif'), ('Courier New', "'Courier New', monospace"),
         ('Verdana', 'Verdana, sans-serif'), ('Times New Roman', "'Times New Roman', serif"),
         ('Trebuchet MS', "'Trebuchet MS', sans-serif"), ('Tahoma', 'Tahoma, sans-serif')]
NAMES = ['Sunny Side Bakery', 'My Portfolio', 'Study Planner', 'Pixel Arcade', 'Green Leaf Cafe', 'Team Dashboard',
         'Daily Tasks', 'Moon Coffee', 'Code Club', 'Blue Wave Surf', 'Maple Library', 'Quick Notes', 'Rocket Store',
         'Fox Fitness', 'Pine Hill Camp', 'Night Owl Books', 'City Bikes', 'Bright Smile Dental', 'Paper Plane Travel']
HELD_OUT_NAMES = ['Harbor Lights Diner', 'Starfield Books', 'Cedar Ridge Camp', 'Neon Noodle Bar']
PEOPLE = ['Sam Lee', 'Maya Cruz', 'Jordan Park', 'Priya Shah', 'Leo Novak', 'Ava Brown']
FOOTERS = ['© 2026 {person}', 'Made by {person}', 'Contact: hello@example.com', 'Thanks for visiting!',
           '© 2026 All rights reserved', 'Built by {person} with HTML, CSS and JavaScript']
NEXT = ['now ', 'ok now ', 'Next, ', 'Thanks! Now ', 'cool. ', 'Great. Next: ', 'new one: ', 'Another one: ']

# edit -> (training wordings, held-out wordings)
EDIT_ASKS = {
    'accent': (['make the buttons {color}', 'Change the main color to {color}.', 'can you make it {color}', 'use {color} instead',
                '{color} buttons please', 'Change the accent color to {color}.', 'make the color {color}',
                'I want the buttons to be {color}.'], ['switch the color to {color}', 'could the buttons be {color}?']),
    'heading': (['make the heading {color}', 'Change the title color to {color}.', 'make the h1 {color}', 'color the heading {color}'],
                ['i want the big heading in {color}']),
    'dark': (['make it dark mode', 'Make the background dark.', 'dark theme please', 'can you make it darker',
              'change the background to black', 'Give it a dark theme.'], ['i want a dark background']),
    'light': (['make the background white', 'make it light mode', 'Light theme please.', 'Make it brighter with a white background.'],
              ['can the background be light?']),
    'title': (['change the title to {name}', 'Rename it to "{name}".', 'make the heading say {name}', 'call it {name}',
               'Change the name to {name}.', 'the title should be {name}'], ['put "{name}" as the title']),
    'font': (['change the font to {font}', 'Use {font} for the font.', 'use {font} font', 'can you switch the font to {font}'],
             ['I prefer {font}, change the font']),
    'bigger': (['make the text bigger', 'Increase the font size.', 'bigger font please', 'the text is too small'],
               ['can you make the writing larger?']),
    'rounded': (['make the buttons rounded', 'Round the buttons.', 'make the buttons pill shaped', 'rounder buttons please'],
                ['give the buttons round corners']),
    'footer': (['add a footer that says "{text}"', 'Add a footer with {text}.', 'put "{text}" at the bottom',
                'can you add a footer: {text}'], ['I need a footer saying {text}']),
    'feature': (['add {a}', 'Can you add {a} too?', 'also add {a}', 'put {a} on the page', 'Add {a} below it.', 'now add {a}'],
                ['could you also put {a} on it?']),
}
WEIGHTS = {'accent': 1.0, 'heading': 0.5, 'dark': 0.8, 'light': 0.5, 'title': 1.0, 'font': 0.7, 'bigger': 0.6, 'rounded': 0.5,
           'footer': 0.7, 'feature': 1.6}
SAME = ['', ' Everything else is the same.', ' The rest of the page is unchanged.']


def style_bounds(html):
    start = html.find('<style>')
    return (start + len('<style>'), html.index('</style>', start)) if start >= 0 else None


def set_property(html, selector, prop, value):
    """`prop: value` in the one-line CSS rule for exactly `selector`; the property, rule or <style> block is added when
    missing. None when the rule spans lines (left alone rather than reformatted)."""
    bounds = style_bounds(html)
    if not bounds:
        return html.replace('</head>', f'<style>\n{selector} {{ {prop}: {value}; }}\n</style>\n</head>', 1)
    start, end = bounds
    css = html[start:end]
    rule = re.search(rf'(?m)^([ \t]*){re.escape(selector)} \{{([^}}]*)\}}', css)
    if not rule:
        lines = [line for line in css.splitlines() if line.strip()]
        indent = re.match(r'[ \t]*', lines[-1]).group(0) if lines else ''
        trailing = css[len(css.rstrip()):]
        css = css.rstrip() + f'\n{indent}{selector} {{ {prop}: {value}; }}' + trailing
        return html[:start] + css + html[end:]
    body = rule.group(2)
    if '\n' in body:
        return None
    if re.search(rf'(?:^|[;\s]){re.escape(prop)}\s*:', body):
        body = re.sub(rf'((?:^|[;\s]){re.escape(prop)}\s*:\s*)[^;]+', lambda m: m.group(1) + value, body, count=1)
    else:
        body = body.rstrip()
        body = (body + ('' if not body.strip() or body.endswith(';') else ';') + f' {prop}: {value}; ')
    css = css[:rule.start(2)] + body + css[rule.end(2):]
    return html[:start] + css + html[end:]


def root_colors(html):
    match = re.search(r':root \{[^}]*--bg: (#[0-9a-fA-F]{6})', html)
    return match.group(1) if match else None


def body_background(html):
    match = re.search(r'(?m)^[ \t]*body \{[^}]*background: (#[0-9a-fA-F]{6})', html)
    return match.group(1) if match else None


def dark(color):
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return 0.299 * r + 0.587 * g + 0.114 * b < 110


def set_root(html, **values):
    def replace(match):
        rule = match.group(0)
        for name, value in values.items():
            rule = re.sub(rf'--{name}: [^;]+', f'--{name}: {value}', rule)
        return rule
    return re.sub(r':root \{[^}]*\}', replace, html, count=1)


def insert_before(html, marker, text, last=False):
    at = html.rfind(marker) if last else html.find(marker)
    return html[:at] + text + html[at:] if at >= 0 else None


def edit(kind, html, theme, features, rng, held_out):
    """(ask, intro, new page, expected strings, features after) for one edit, or None when it does not fit this page."""
    asks = EDIT_ASKS[kind][1 if held_out else 0]
    root = ':root {' in html
    if kind in ('accent', 'rounded') and '<button' not in html:
        return None
    if kind == 'accent':
        color = rng.choice(list(COLORS))
        hex_ = COLORS[color]
        if root:
            page = set_root(html, accent=hex_)
            intro = rng.choice([f'Here is the page with {color} buttons. I changed --accent in the :root block, so every button uses it.',
                                f'Done: --accent in :root is now {hex_} ({color}), and the buttons take their color from it.'])
        else:
            page = set_property(html, 'button', 'background', hex_)
            intro = f'Here it is with {color} buttons: the button rule now sets the background to {hex_}.'
        return rng.choice(asks).format(color=color), intro, page, [hex_], features
    if kind == 'heading':
        if '<h1' not in html:
            return None
        color = rng.choice(list(COLORS))
        page = set_property(html, 'h1', 'color', COLORS[color])
        return (rng.choice(asks).format(color=color), f'Here is the page with a {color} heading: the h1 rule now sets color: {COLORS[color]}.',
                page, [COLORS[color]], features)
    if kind in ('dark', 'light'):
        current = root_colors(html) if root else body_background(html)
        if bool(current and dark(current)) != (kind == 'light'):
            return None
        bg, surface, text = DARK if kind == 'dark' else LIGHT
        if root:
            page = set_root(html, bg=bg, surface=surface, text=text)
            intro = (f'Here is the {kind} version. The colors in :root now use {bg} for the background and {text} for the text, '
                     'so every part of the page follows.')
        else:
            page = set_property(html, 'body', 'background', bg)
            page = page and set_property(page, 'body', 'color', text)
            intro = f'Here is the {kind} version: body now has background {bg} and text color {text}.'
        return rng.choice(asks), intro, page, [bg], features
    if kind == 'title':
        name = rng.choice(HELD_OUT_NAMES if held_out else NAMES)
        if '<title>' not in html or features == ['hello']:  # a renamed hello page no longer says hello
            return None
        page = re.sub(r'<title>[^<]*</title>', f'<title>{name}</title>', html, count=1)
        expect = [f'<title>{name}</title>']
        if re.search(r'<h1>[^<]*</h1>', page):
            page = re.sub(r'<h1>[^<]*</h1>', f'<h1>{name}</h1>', page, count=1)
            expect.append(f'<h1>{name}</h1>')
        intro = (f'Here is the page renamed to {name}: the <title> in <head> and the main heading both changed.' if len(expect) == 2
                 else f'Here it is with the title {name} in the <title> tag.')
        return rng.choice(asks).format(name=name), intro, page, expect, features
    if kind == 'font':
        name, value = rng.choice([f for f in FONTS if f[1] not in html])
        page = set_property(html, 'body', 'font-family', value)
        return (rng.choice(asks).format(font=name), f'Here it is in {name}. The font-family on body sets the font for the whole page.',
                page, [value], features)
    if kind == 'bigger':
        if 'font-size: 1.25rem' in html:
            return None
        page = set_property(html, 'body', 'font-size', '1.25rem')
        return (rng.choice(asks), 'Here is the page with bigger text: body now has font-size: 1.25rem, and the rest of the text scales with it.',
                page, ['font-size: 1.25rem'], features)
    if kind == 'rounded':
        page = set_property(html, 'button', 'border-radius', '999px')
        return (rng.choice(asks), 'Here are rounded buttons: border-radius: 999px on the button rule makes them pill shaped.',
                page, ['999px'], features)
    if kind == 'footer':
        if '<footer' in html:
            return None
        text = rng.choice(FOOTERS).format(person=rng.choice(PEOPLE))
        line = re.search(r'(?m)^([ \t]*)</main>', html)
        if line:
            page = html[:line.end()] + f'\n{line.group(1)}<footer>{text}</footer>' + html[line.end():]
        else:  # a page without <main>: the footer goes last in <body>, before the script
            line = re.search(r'(?m)^([ \t]*)<script>', html) or re.search(r'(?m)^([ \t]*)</body>', html)
            indent = line.group(1) + ('  ' if line.group(0).strip() == '</body>' else '')
            page = html[:line.start()] + f'{indent}<footer>{text}</footer>\n' + html[line.start():]
        return rng.choice(asks).format(text=text), 'Here is the page with the footer added at the bottom.', page, [text], features
    if kind == 'feature':
        if not root or not all(tag in html for tag in ('</main>', '<style>', '<script>')):  # features use the :root colors
            return None
        options = [f for f in FEATURES if f.__name__ not in features]
        feature = rng.choice(options)
        _, section, css, js = feature(theme, rng)
        name = rng.choice(basics.STANDALONE[feature.__name__])
        page = insert_before(html, '</main>', section + '\n', last=True)
        page = insert_before(page, '</style>', css + '\n')
        page = insert_before(page, '</script>', '\n' + js + '\n', last=True)
        intro = rng.choice([f'Here is the page with {basics.article(name)} added. Its HTML goes at the end of <main>, its CSS at the end '
                            'of <style> and its script at the end of <script>.',
                            f'Done: {basics.article(name)} is now on the page, below what was there. I added its HTML, CSS and '
                            'JavaScript to the same file.'])
        return rng.choice(asks).format(a=basics.article(name)), intro, page, [], features + [feature.__name__]
    raise ValueError(kind)


def reply_for(intro, page, rng):
    return intro + rng.choice(SAME) + '\n\n```html\n' + page + '\n```'


def page_of(message):
    return message['content'].split('```html\n', 1)[1].rsplit('\n```', 1)[0]


def fits(messages, tokenizer):
    prompt, _ = conversation(messages[:-1], generation=True, tokenizer=tokenizer)
    whole, _ = conversation(messages, tokenizer=tokenizer)
    return len(prompt) <= PROMPT_MAX and len(whole) <= CONTEXT - 8


def first_page(themes, rng, split):
    """A page from make_web_basics plus the theme it was built with (example() draws the theme first)."""
    state = rng.getstate()
    theme = rng.choice(themes)
    rng.setstate(state)
    row = basics.example(themes, rng, split)
    return (row, theme) if row else (None, None)


def follow_up(prompt, rng):
    if rng.random() < 0.35:
        lead = rng.choice(NEXT)
        prompt = lead + (prompt[0].lower() + prompt[1:] if lead[-2:] != ': ' or rng.random() < 0.5 else prompt)
    return prompt


def new_chat(themes, rng, split, code_rows, tokenizer):
    first, _ = first_page(themes, rng, split)
    if not first:
        return None
    messages, features = list(first['messages']), first['features']
    turns = 3 if split == 'train' and rng.random() < 0.1 else 2
    language = 'html'
    for _ in range(turns - 1):
        if split == 'train' and rng.random() < 0.25:
            other = rng.choice(code_rows)
            if rng.random() < 0.4 and len(messages) == 2:  # code first, then the page
                messages = other['messages'] + messages
                continue
            messages = messages + [{'role': 'user', 'content': follow_up(other['messages'][0]['content'], rng)}, other['messages'][1]]
            features, language = [], other.get('language')
            continue
        second = basics.example(themes, rng, split)
        if not second or second['features'][0] in features:
            return None
        messages = messages + [{'role': 'user', 'content': follow_up(second['messages'][0]['content'], rng)}, second['messages'][1]]
        features, language = second['features'], 'html'
    if not fits(messages, tokenizer):
        return None
    return {'split': split, 'family': 'followup_new', 'source': 'make_cide_followups', 'language': language, 'features': features,
            'messages': messages}


def edit_chat(themes, rng, split, tokenizer, kind=None):
    first, theme = first_page(themes, rng, split)
    if not first:
        return None
    messages, features, html = list(first['messages']), first['features'], page_of(first['messages'][1])
    kinds = [kind] if kind else rng.choices(list(WEIGHTS), weights=list(WEIGHTS.values()), k=2 if split == 'train' and rng.random() < 0.12 else 1)
    expect, done = [], []
    if {'dark', 'light'} <= set(kinds):
        return None
    for k in kinds:
        made = edit(k, html, theme, features, rng, split != 'train')
        if not made or not made[2] or not basics.valid(made[2]) or made[2] == html or k in done:
            return None
        ask, intro, html, wanted, features = made
        messages += [{'role': 'user', 'content': ask}, {'role': 'assistant', 'content': reply_for(intro, html, rng)}]
        expect, done = wanted, done + [k]
    if not fits(messages, tokenizer):
        return None
    return {'split': split, 'family': 'followup_edit', 'source': 'make_cide_followups', 'language': 'html', 'features': features,
            'edits': done, 'expect': expect, 'messages': messages}


def build(seed=2433):
    rng = random.Random(seed)
    tokenizer = Model.load(TOKENIZER)[0].tokenizer
    code = [json.loads(line) for line in (ROOT / 'data/cide-v7.jsonl').read_text(encoding='utf-8').splitlines()]
    code_rows = [r for r in code if r['split'] == 'train' and len(r['messages']) == 2 and r.get('language') not in (None, 'html')
                 and len(r['messages'][1]['content']) < 2_500]
    train_themes = [t for t in THEMES if t[0] not in HELD_OUT_THEMES]
    held_out = [t for t in THEMES if t[0] in HELD_OUT_THEMES]
    seen, rows = set(), {'new': [], 'edit': []}

    def add(row, bucket):
        if row and row['messages'][-1]['content'] not in seen:
            seen.add(row['messages'][-1]['content'])
            rows[bucket].append(row)
            return True
        return False

    while len(rows['new']) < TRAIN_NEW:
        add(new_chat(train_themes, rng, 'train', code_rows, tokenizer), 'new')
    while len(rows['edit']) < TRAIN_EDIT:
        add(edit_chat(train_themes, rng, 'train', tokenizer), 'edit')
    train = rows['new'] + rows['edit']
    rng.shuffle(train)
    # kelus.website_eval reads the first validation rows: a new page, then two edits of different kinds, round robin.
    new_rows = []
    while len(new_rows) < VALIDATION_NEW:
        row = new_chat(held_out, rng, 'validation', code_rows, tokenizer)
        if row and row['messages'][-1]['content'] not in seen:
            seen.add(row['messages'][-1]['content'])
            new_rows.append(row)
    edit_rows, kinds, misses = [], list(WEIGHTS), 0
    while len(edit_rows) < VALIDATION_EDIT:
        kind = kinds[len(edit_rows) % len(kinds)]
        row = edit_chat(held_out, rng, 'validation', tokenizer, kind)
        if row and row['messages'][-1]['content'] not in seen:
            seen.add(row['messages'][-1]['content'])
            edit_rows.append(row)
            misses = 0
        elif (misses := misses + 1) > 300:  # no held-out page takes this edit (say, no dark theme to lighten)
            kinds.remove(kind)
            misses = 0
    validation = []
    for i in range(max(len(new_rows), (len(edit_rows) + 1) // 2)):
        validation += new_rows[i:i + 1] + edit_rows[2 * i:2 * i + 2]
    return train + validation, tokenizer


if __name__ == '__main__':
    rows, tokenizer = build()
    destination = ROOT / 'data/cide-followups-v1.jsonl'
    destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    train = [r for r in rows if r['split'] == 'train']
    sizes = sorted(len(conversation(r['messages'], tokenizer=tokenizer)[0]) for r in train[:600])
    edits = {}
    for r in train:
        for k in r.get('edits', []):
            edits[k] = edits.get(k, 0) + 1
    print(f'{destination}: {len(train):,} training chats ({sum(r["family"] == "followup_new" for r in train):,} new requests, '
          f'{sum(r["family"] == "followup_edit" for r in train):,} edits {edits}), {len(rows) - len(train)} held out; '
          f'tokens median {sizes[len(sizes) // 2]:,}, longest {sizes[-1]:,}')
