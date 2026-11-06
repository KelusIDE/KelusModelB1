"""Kani Painter v4 data: fruit on a table and anime-style portraits (data/make_paint_anime.py), added to paint-v3.

Nine fruits (apple, banana, orange, lemon, pear, strawberry, cherries, grapes, watermelon slice)
drawn alone ("draw a banana") or as a still life ("draw three red apples and a banana on a
table"). Each fruit has a highlight with its own opacity ('white/30' for apples, ...), which
kelus/paint_eval.py uses to tell them apart. Validation holds out still lifes that pair a banana
with strawberries, so those combinations are never trained.

    python3 data/make_paint_fruit.py   # writes data/paint-v4.jsonl and data/paint-v4-subjects.json
"""
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'data'))
import make_paint as mp  # noqa: E402
from kelus.paint import render  # noqa: E402

n, at = mp.n, mp.at


def shadow(s, width):
    return f"ellipse(x, y + {n(2*s)}, {n(width*s)}, {n(6*s)}, 'black/20')"


def apple(s, skin):
    return [shadow(s, 26), f"circle(x, y - {n(26*s)}, {n(26*s)}, '{skin}')", f"circle(x - {n(9*s)}, y - {n(34*s)}, {n(7*s)}, 'white/30')",
            f"line(x, y - {n(50*s)}, x + {n(3*s)}, y - {n(62*s)}, 'brown', {n(3*s)})", f"ellipse(x + {n(10*s)}, y - {n(58*s)}, {n(8*s)}, {n(4*s)}, 'green')"]


def pear(s, skin):
    return [shadow(s, 22), f"circle(x, y - {n(22*s)}, {n(22*s)}, '{skin}')", f"circle(x, y - {n(48*s)}, {n(14*s)}, '{skin}')",
            f"circle(x - {n(7*s)}, y - {n(28*s)}, {n(6*s)}, 'white/25')", f"line(x, y - {n(60*s)}, x - {n(3*s)}, y - {n(72*s)}, 'brown', {n(3*s)})"]


def banana(s):
    outer = [(-50, 42), (-32, 16), (0, 6), (32, 16), (50, 42)]
    inner = [(42, 44), (22, 30), (0, 24), (-22, 30), (-42, 44)]
    points = ', '.join(f'(x {at(dx*s)}, y - {n(dy*s)})' for dx, dy in outer + inner)
    return [shadow(s, 44), f"polygon([{points}], 'yellow')", f"ellipse(x, y - {n(14*s)}, {n(22*s)}, {n(3*s)}, 'cream/60')",
            f"circle(x - {n(50*s)}, y - {n(43*s)}, {n(3*s)}, 'dark_brown')", f"circle(x + {n(50*s)}, y - {n(43*s)}, {n(3*s)}, 'dark_brown')"]


def orange(s):
    return [shadow(s, 26), f"circle(x, y - {n(26*s)}, {n(26*s)}, 'orange')", f"circle(x - {n(9*s)}, y - {n(34*s)}, {n(7*s)}, 'white/22')",
            f"circle(x, y - {n(51*s)}, {n(3*s)}, 'forest')", f"ellipse(x + {n(9*s)}, y - {n(54*s)}, {n(8*s)}, {n(3*s)}, 'green')"]


def lemon(s):
    return [shadow(s, 30), f"ellipse(x, y - {n(20*s)}, {n(30*s)}, {n(20*s)}, 'yellow')", f"circle(x - {n(30*s)}, y - {n(20*s)}, {n(5*s)}, 'yellow')",
            f"circle(x + {n(30*s)}, y - {n(20*s)}, {n(5*s)}, 'yellow')", f"ellipse(x - {n(8*s)}, y - {n(28*s)}, {n(10*s)}, {n(4*s)}, 'cream/50')"]


def strawberry(s):
    return [shadow(s, 16), f"polygon([(x - {n(18*s)}, y - {n(34*s)}), (x + {n(18*s)}, y - {n(34*s)}), (x + {n(12*s)}, y - {n(12*s)}), (x, y), "
            f"(x - {n(12*s)}, y - {n(12*s)})], 'red')", f"circle(x - {n(5*s)}, y - {n(26*s)}, {n(5*s)}, 'white/15')",
            f"for k in range(7):", f"    circle(x + rand(-{n(11*s)}, {n(11*s)}), y - rand({n(10*s)}, {n(30*s)}), {n(max(1, 1.4*s))}, 'cream')",
            f"polygon([(x - {n(16*s)}, y - {n(34*s)}), (x, y - {n(42*s)}), (x + {n(16*s)}, y - {n(34*s)}), (x, y - {n(30*s)})], 'green')"]


def cherries(s):
    return [shadow(s, 22), f"line(x - {n(10*s)}, y - {n(12*s)}, x, y - {n(48*s)}, 'green', {n(2*s)})",
            f"line(x + {n(10*s)}, y - {n(12*s)}, x, y - {n(48*s)}, 'green', {n(2*s)})", f"circle(x - {n(10*s)}, y - {n(10*s)}, {n(10*s)}, 'crimson')",
            f"circle(x + {n(10*s)}, y - {n(10*s)}, {n(10*s)}, 'crimson')", f"circle(x - {n(13*s)}, y - {n(13*s)}, {n(3*s)}, 'white/40')"]


def grapes(s, skin):
    berries = [(0, 14), (-10, 24), (10, 24), (-18, 36), (0, 36), (18, 36), (-10, 48), (10, 48), (0, 58)]
    return [shadow(s, 24), *(f"circle(x {at(dx*s)}, y - {n(dy*s)}, {n(9*s)}, '{skin}')" for dx, dy in berries),
            f"circle(x - {n(4*s)}, y - {n(40*s)}, {n(3*s)}, 'white/45')", f"line(x, y - {n(66*s)}, x + {n(4*s)}, y - {n(76*s)}, 'brown', {n(3*s)})"]


def watermelon(s):
    rind = ', '.join(f'(x {at(-46*s*c)}, y - {n(46*s*w)})' for c, w in [(1, 0.06), (0.87, 0.5), (0.5, 0.87), (0, 1), (-0.5, 0.87), (-0.87, 0.5), (-1, 0.06)])
    flesh = ', '.join(f'(x {at(-40*s*c)}, y - {n(40*s*w + 6*s)})' for c, w in [(1, 0.06), (0.87, 0.5), (0.5, 0.87), (0, 1), (-0.5, 0.87), (-0.87, 0.5), (-1, 0.06)])
    return [shadow(s, 46), f"polygon([{rind}], 'green')", f"polygon([{flesh}], 'coral')",
            f"for k in range(6):", f"    circle(x + rand(-{n(24*s)}, {n(24*s)}), y - rand({n(16*s)}, {n(34*s)}), {n(max(1, 2*s))}, 'charcoal')",
            f"ellipse(x - {n(14*s)}, y - {n(34*s)}, {n(10*s)}, {n(3*s)}, 'white/10')"]


FRUIT = {
    'apple': ('apple', 'apples', ['red', 'green', 'yellow']), 'banana': ('banana', 'bananas', None), 'orange': ('orange', 'oranges', None),
    'lemon': ('lemon', 'lemons', None), 'pear': ('pear', 'pears', ['green', 'yellow']), 'strawberry': ('strawberry', 'strawberries', None),
    'cherries': ('pair of cherries', 'pairs of cherries', None), 'grapes': ('bunch of grapes', 'bunches of grapes', ['purple', 'green']),
    'watermelon': ('slice of watermelon', 'slices of watermelon', None)}
WIDTH = {'apple': 60, 'banana': 110, 'orange': 60, 'lemon': 76, 'pear': 52, 'strawberry': 40, 'cherries': 50, 'grapes': 56, 'watermelon': 100}
WALLS = [('#fff1c9', '#ffd6b0'), ('#e8f1ff', '#c9dcf2'), ('#f4f7ff', '#dfe4ec'), ('#e9f7df', '#cfe8c0'), ('#fde2e4', '#f7c6cc')]
TABLES = ['tan', 'brown', 'white', 'sand']
WHERE = ['on a table', 'on a wooden table', 'on the kitchen table', '']
VERBS = mp.VERBS


def body(name, s, color):
    return {'apple': lambda: apple(s, color or 'red'), 'pear': lambda: pear(s, color or 'green'), 'banana': lambda: banana(s), 'orange': lambda: orange(s),
            'lemon': lambda: lemon(s), 'strawberry': lambda: strawberry(s), 'cherries': lambda: cherries(s),
            'grapes': lambda: grapes(s, color or 'purple'), 'watermelon': lambda: watermelon(s)}[name]()


def phrase(name, count, color):
    singular, plural, _ = FRUIT[name]
    word = singular if count == 1 else plural
    adjective = f'{color} ' if color else ''
    quantity = ('an' if (adjective or word)[0] in 'aeiou' else 'a') if count == 1 else mp.NUMBERS[count]
    return f'{quantity} {adjective}{word}'


def table(rng, scene):
    top, bottom = rng.choice(WALLS)
    surface = rng.choice(TABLES)
    scene.add(f"gradient(0, 0, 640, 290, '{top}', '{bottom}')", f"rect(0, 290, 640, 110, '{surface}')", "rect(0, 290, 640, 6, 'dark_brown/40')")


def still_life(rng, split):
    """One to three fruits, each one to five times, on a table."""
    while True:
        names = rng.sample(list(FRUIT), rng.choice([1, 1, 2, 2, 3]))
        held = 'banana' in names and 'strawberry' in names
        if held == (split == 'validation'):
            break
    scene = mp.Scene(rng)
    table(rng, scene)
    lane, picks = 500 / len(names), []
    for k, name in enumerate(names):
        s = rng.choice([1.3, 1.5])
        count = max(1, min(rng.choice([1, 1, 2, 3, 4, 5]), int(lane // (WIDTH[name] * s))))
        colors = FRUIT[name][2]
        color = rng.choice(colors) if colors and rng.random() < 0.6 else None
        margin = min(lane / 2 - 1, WIDTH[name] * s / 2)
        scene.place(count, (70 + k * lane + margin, 70 + (k + 1) * lane - margin), rng.choice([330, 345]), body(name, s, color), jitter=6)
        picks.append((name, count, color))
    parts = [phrase(*p) for p in picks]
    listed = parts[0] if len(parts) == 1 else ', '.join(parts[:-1]) + ' and ' + parts[-1]
    where = rng.choice(WHERE)
    style = rng.random()
    text = (f"{rng.choice(VERBS)} a still life with {listed}" if style < 0.2 else
            f"{rng.choice(VERBS)} {listed}{' ' + where if where else ''}")
    text = text[0].upper() + text[1:] if rng.random() < 0.3 else text
    return {'split': split, 'family': 'paint', 'scene': {'still_life': True, 'objects': [list(p) for p in picks]},
            'messages': [{'role': 'user', 'content': text}, {'role': 'assistant', 'content': '```python\n' + '\n'.join(scene.lines) + '\n```'}]}


def single(rng):
    """One fruit drawn large: 'draw a banana', 'paint a green apple on a table'."""
    name = rng.choice(list(FRUIT))
    colors = FRUIT[name][2]
    color = rng.choice(colors) if colors and rng.random() < 0.6 else None
    scene = mp.Scene(rng)
    table(rng, scene)
    scene.place(1, (320, 320), 345, body(name, rng.choice([2.2, 2.6, 3.0]) * (0.8 if name in ('banana', 'watermelon') else 1), color))
    where = rng.choice(WHERE + ['', ''])
    text = f"{rng.choice(VERBS)} {phrase(name, 1, color)}{' ' + where if where else ''}"
    return {'split': 'train', 'family': 'paint', 'scene': {'object': name, 'color': color, 'still_life': True},
            'messages': [{'role': 'user', 'content': text}, {'role': 'assistant', 'content': '```python\n' + '\n'.join(scene.lines) + '\n```'}]}


def subjects():
    """kelus/paint_subjects.json once v4 is the served painter: v3's subjects plus fruit."""
    known = mp.subjects()
    words = set(known['words'])
    for singular, plural, _ in FRUIT.values():
        words.update(w for w in f'{singular} {plural}'.split() if w not in mp.STOP and w not in ('pair', 'pairs', 'bunch', 'bunches', 'slice', 'slices'))
    words.update(['fruit', 'fruits', 'table', 'cherry', 'grape', 'kitchen', 'anime', 'girl', 'boy', 'character', 'portrait', 'manga', 'chibi'])
    return {**known, 'version': 4, 'objects': known['objects'] + ['apples', 'bananas', 'oranges', 'lemons', 'pears', 'strawberries', 'cherries', 'grapes', 'watermelon', 'anime girls and boys'],
            'settings': known['settings'] + ['a table'], 'words': sorted(words)}


def build(seed=2035, still=6000, validation=150, singles=3000, portraits=6000, max_tokens=900):
    from kelus.tokenizer import from_dict
    tokenizer = from_dict(json.loads((ROOT / 'data/foundation/tokenizer.json').read_text()))
    rng = random.Random(seed)
    import make_paint_anime as anime
    rows = [still_life(rng, 'train') for _ in range(still)] + [still_life(rng, 'validation') for _ in range(validation)] + [single(rng) for _ in range(singles)]
    rows += [anime.record(rng, 'train') for _ in range(portraits)] + [anime.record(rng, 'validation') for _ in range(validation)]
    unique = {}
    for row in rows:
        unique.setdefault(row['messages'][0]['content'], row)
    rows = [r for r in unique.values() if sum(len(tokenizer.encode(m['content'])) for m in r['messages']) <= max_tokens]
    for row in rows[:600]:
        render(row['messages'][1]['content'][10:-3])
    base = [json.loads(line) for line in open(ROOT / 'data/paint-v3.jsonl', encoding='utf-8')]
    rows = base + rows
    rng.shuffle(rows)
    return rows


if __name__ == '__main__':
    rows = build()
    (ROOT / 'data/paint-v4.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    (ROOT / 'data/paint-v4-subjects.json').write_text(json.dumps(subjects(), indent=1) + '\n', encoding='utf-8')
    fruit = [r for r in rows if r['scene'].get('still_life')]
    portraits = [r for r in rows if r['scene'].get('anime')]
    print(json.dumps({'records': len(rows), 'fruit': len(fruit), 'anime': len(portraits), 'anime_validation': sum(r['split'] == 'validation' for r in portraits)}))
