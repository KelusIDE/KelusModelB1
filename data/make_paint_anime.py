"""Kani Painter: anime-style character portraits (flat vector), for paint-v4.

A head-and-shoulders portrait composed from Kani Paint's character parts (hair_back, outfit, face,
anime_eye, bangs, cat_ears, hair_bow in kelus/paint.py) plus a drawn mouth: four hair styles (long,
short bob, twin tails, ponytail), big eyes, outfits (school uniform with a bow, hoodie, dress) and
optional cat ears or a hair bow, over a pastel, sakura or starry background. Parts keep a portrait
near 400 tokens, inside the model's 1,024-token context. Requests name any mix of those, e.g. "draw an
anime girl with long pink hair and blue eyes in a school uniform".
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'data'))
import make_paint as mp  # noqa: E402

HAIR = {'pink': 'pink', 'blue': 'blue', 'black': 'charcoal', 'brown': 'brown', 'blonde': 'gold', 'silver': 'silver', 'purple': 'purple',
        'red': 'crimson', 'green': 'teal', 'white': 'snow'}
EYES = {'blue': 'blue', 'green': 'green', 'red': 'red', 'purple': 'violet', 'brown': 'brown', 'gold': 'gold', 'pink': 'magenta'}
STYLES = {'long': ['long'], 'short': ['short'], 'twin tails': ['twintails', 'pigtails'], 'ponytail': ['ponytail']}
STYLE_ARG = {'long': 'long', 'short': 'short', 'twin tails': 'twintails', 'ponytail': 'ponytail'}
OUTFITS = {'school uniform': ('navy', True), 'hoodie': (None, False), 'dress': (None, False)}
BACKGROUNDS = {'pastel': ['', ''], 'sakura': ['under cherry blossoms', 'with cherry blossoms'], 'night': ['at night', 'under the stars'],
               'sparkles': ['with sparkles', 'with a sparkly background']}


def background(rng, kind):
    if kind == 'night':
        lines = ["gradient(0, 0, 640, 400, '#0b1033', '#3d2f7a')", 'for i in range(140):', "    circle(rand(0, 640), rand(0, 400), rand(0.4, 1.6), 'white/80')"]
    else:
        top, bottom = rng.choice([('#ffd6e8', '#fff1f7'), ('#d6ecff', '#f2f8ff'), ('#e8ddff', '#fbf7ff'), ('#fff1c9', '#ffe3ef')])
        lines = [f"gradient(0, 0, 640, 400, '{top}', '{bottom}')"]
    if kind == 'sakura':
        lines += ['for i in range(40):', "    ellipse(rand(0, 640), rand(0, 400), rand(4, 8), rand(2, 4), mix('pink', 'white', rand(0, 0.6)))"]
    if kind == 'sparkles':
        lines += ['for i in range(30):', "    glow(rand(0, 640), rand(0, 400), rand(6, 16), 'white/70')"]
    return lines


def portrait(rng, hair, eyes, style, outfit, extra):
    """Code lines for the character, from Kani Paint's character parts (kelus/paint.py)."""
    h, e = HAIR[hair], EYES[eyes]
    body = OUTFITS[outfit][0] or rng.choice(['red', 'blue', 'purple', 'teal', 'pink', 'gray', 'white'])
    kind = {'school uniform': 'uniform', 'hoodie': 'hoodie', 'dress': 'dress'}[outfit]
    lines = ['x = 320', 'y = 190', f"hair_back(x, y, '{h}', '{STYLE_ARG[style]}')", f"outfit(x, y, '{kind}', '{body}')", 'face(x, y)',
             f"anime_eye(x - 30, y + 6, '{e}', -1)", f"anime_eye(x + 30, y + 6, '{e}', 1)",
             rng.choice(["polygon([(x - 10, y + 48), (x + 10, y + 48), (x, y + 56)], 'crimson')", "line(x - 8, y + 50, x + 8, y + 50, 'crimson', 2)"]),
             f"bangs(x, y, '{h}')"]
    if extra == 'cat ears':
        lines.append(f"cat_ears(x, y, '{h}')")
    elif extra == 'bow':
        lines.append(f"hair_bow(x, y, '{rng.choice(['red', 'pink', 'white', 'blue'])}')")
    return lines


def record(rng, split='train'):
    """Validation holds out ponytails at night, so those portraits are never trained."""
    while True:
        gender = rng.choice(['girl', 'girl', 'girl', 'boy'])
        hair = rng.choice(list(HAIR))
        eyes = rng.choice(list(EYES))
        style = 'short' if gender == 'boy' else rng.choice(list(STYLES))
        outfit = rng.choice(list(OUTFITS)) if gender == 'girl' else rng.choice(['school uniform', 'hoodie'])
        extra = rng.choice([None, None, 'cat ears', 'bow']) if gender == 'girl' else rng.choice([None, None, 'cat ears'])
        kind = rng.choice(list(BACKGROUNDS))
        if (style == 'ponytail' and kind == 'night') == (split == 'validation'):
            break
    program = background(rng, kind) + portrait(rng, hair, eyes, style, outfit, extra)

    parts = []
    named = {k for k in ('hair', 'eyes', 'style', 'outfit', 'extra', 'background') if rng.random() < 0.5}
    hair_words = ' '.join(w for w in (style if 'style' in named and style != 'twin tails' else '', hair if 'hair' in named else '') if w)
    if 'style' in named and style == 'twin tails':
        parts.append(f"{hair + ' ' if 'hair' in named else ''}{rng.choice(STYLES[style])}")
    elif hair_words:
        parts.append(f'{hair_words} hair')
    if 'eyes' in named:
        parts.append(f'{eyes} eyes')
    if 'extra' in named and extra:
        parts.append(extra if extra == 'cat ears' else 'a hair bow')
    text = f"{rng.choice(mp.VERBS)} {rng.choice(['an anime', 'a cute anime', 'an anime-style'])} {gender}"
    if parts:
        text += ' with ' + (parts[0] if len(parts) == 1 else ', '.join(parts[:-1]) + ' and ' + parts[-1])
    if 'outfit' in named:
        text += f" in a{'n' if outfit[0] in 'aeiou' else ''} {outfit}"
    if 'background' in named and BACKGROUNDS[kind][0]:
        text += ' ' + rng.choice(BACKGROUNDS[kind])
    scene = {'anime': True, 'hair': HAIR[hair] if 'hair' in named else None, 'eyes': EYES[eyes] if 'eyes' in named else None,
             'extra': extra if 'extra' in named else None}
    return {'split': split, 'family': 'paint', 'scene': scene,
            'messages': [{'role': 'user', 'content': text}, {'role': 'assistant', 'content': '```python\ncanvas(640, 400)\n' + '\n'.join(program) + '\n```'}]}
