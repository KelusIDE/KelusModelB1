"""Kani Paint: draw images from a small, safe subset of Python.

The model writes ordinary-looking Python, for example

    canvas(640, 400)
    gradient(0, 0, 640, 260, '#0b1d3a', '#f28c38')
    for i in range(80):
        circle(rand(0, 640), rand(0, 150), rand(0.6, 1.6), '#ffffffaa')
    mountains(300, 90, '#2a2440', seed=3)

and this module renders it. Generated code is never passed to exec or eval: the
source is parsed with `ast` and only whitelisted statements, expressions and
drawing calls are interpreted, with limits on loop iterations, shapes and canvas
size. No imports, attributes, files, network or arbitrary functions exist here.

    python3 -m kelus.paint scene.py --out scene.png
"""
import argparse
import ast
import functools
import json
import math
from pathlib import Path
import random
import re

import numpy as np

MAX_SIDE = 1024
MAX_SHAPES = 6000
MAX_STEPS = 200000
HEX = re.compile(r'#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$')


class PaintError(ValueError):
    """A program that is not valid Kani Paint, with the offending line when known."""



NAMED = {
    'white': '#ffffff', 'snow': '#f4f7ff', 'silver': '#c0c6d0', 'gray': '#808891', 'charcoal': '#2f3338', 'black': '#000000',
    'red': '#e5484d', 'crimson': '#b4232c', 'orange': '#ff8c3a', 'gold': '#f5b83d', 'yellow': '#ffd84d', 'cream': '#fff1c9',
    'tan': '#d2a679', 'sand': '#e8c98f', 'brown': '#7a4a2a', 'dark_brown': '#4a2c1a', 'lime': '#9be15d', 'green': '#3fa34d',
    'forest': '#1f5a3a', 'dark_green': '#123826', 'teal': '#2aa198', 'cyan': '#5fd4e8', 'sky': '#7cc4f2', 'blue': '#3a7bd5',
    'navy': '#1b2a5a', 'indigo': '#3d2f7a', 'purple': '#7a4fc9', 'violet': '#a78bfa', 'pink': '#ff7eb6', 'magenta': '#d63aa8',
    'peach': '#ffb38a', 'coral': '#ff7a59',
}
NAMED_ALPHA = re.compile(r'([a-z_]+)(?:/(\d{1,3}))?$')


def color(value):
    """'#rrggbb', '#rrggbbaa', a NAMED color, or 'name/NN' for NN% opacity."""
    named = NAMED_ALPHA.match(value) if isinstance(value, str) else None
    if named and named[1] in NAMED and int(named[2] or 100) <= 100:
        return color(NAMED[named[1]])[:3] + (round(int(named[2] or 100) * 2.55),)
    if not isinstance(value, str) or not HEX.match(value):
        raise PaintError(f'Colors are names like "gold", "navy/40", or "#rrggbb" strings, not {value!r}')
    channels = [int(value[i:i + 2], 16) for i in range(1, len(value), 2)]
    return tuple(channels + [255] * (4 - len(channels)))


def mix(a, b, t):
    """Blend two hex colors; t=0 gives a, t=1 gives b."""
    a, b, t = color(a), color(b), min(1.0, max(0.0, float(t)))
    return '#' + ''.join(f'{round(x + (y - x) * t):02x}' for x, y in zip(a, b))


class Canvas:
    def __init__(self):
        self.image = self.draw = None
        self.cost = 0

    def require(self, cost=1):
        if self.image is None:
            raise PaintError('Call canvas(width, height) first')
        self.cost += cost
        if self.cost > MAX_SHAPES:
            raise PaintError(f'Drawing budget of {MAX_SHAPES} shapes exceeded')

    def paste(self, pixels, x, y):
        from PIL import Image
        patch = Image.fromarray(np.ascontiguousarray(pixels), 'RGBA')
        self.image.paste(patch, (int(x), int(y)), patch)


    def canvas(self, width, height, background='#000000'):
        from PIL import Image, ImageDraw
        width, height = int(width), int(height)
        if not (16 <= width <= MAX_SIDE and 16 <= height <= MAX_SIDE):
            raise PaintError(f'Canvas sides must be 16..{MAX_SIDE}')
        self.image = Image.new('RGB', (width, height), color(background)[:3])
        self.draw = ImageDraw.Draw(self.image, 'RGBA')

    def gradient(self, x, y, width, height, top, bottom):
        """A rectangle filled with a vertical gradient."""
        self.require(20)
        width, height = min(MAX_SIDE, max(1, int(width))), min(MAX_SIDE, max(1, int(height)))
        t = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
        a, b = np.array(color(top), np.float32), np.array(color(bottom), np.float32)
        self.paste(np.broadcast_to(a + (b - a) * t, (height, width, 4)).round().astype(np.uint8), x, y)

    def glow(self, cx, cy, radius, fill):
        """A soft radial glow that fades to transparent at `radius`."""
        self.require(20)
        radius = min(MAX_SIDE, max(1.0, float(radius)))
        size = int(2 * radius) + 1
        ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
        distance = np.sqrt((xs - radius) ** 2 + (ys - radius) ** 2) / radius
        r, g, b, a = color(fill)
        alpha = (np.clip(1 - distance, 0, 1) ** 2 * a).astype(np.uint8)
        full = lambda v: np.full_like(alpha, v)
        self.paste(np.dstack([full(r), full(g), full(b), alpha]), cx - radius, cy - radius)

    def rect(self, x, y, width, height, fill):
        self.require()
        self.draw.rectangle([x, y, x + width, y + height], fill=color(fill))

    def ellipse(self, cx, cy, rx, ry, fill):
        self.require()
        self.draw.ellipse([cx - abs(rx), cy - abs(ry), cx + abs(rx), cy + abs(ry)], fill=color(fill))

    def circle(self, cx, cy, r, fill):
        self.ellipse(cx, cy, r, r, fill)

    def line(self, x0, y0, x1, y1, fill, width=1):
        self.require()
        self.draw.line([x0, y0, x1, y1], fill=color(fill), width=max(1, min(64, int(width))))

    def polygon(self, points, fill):
        self.require(2)
        if not isinstance(points, list) or not 3 <= len(points) <= 2000 or \
                not all(isinstance(p, tuple) and len(p) == 2 and all(isinstance(v, (int, float)) for v in p) for p in points):
            raise PaintError('polygon takes a list of 3..2000 (x, y) number pairs')
        self.draw.polygon([(float(x), float(y)) for x, y in points], fill=color(fill))

    def mountains(self, base, height, fill, seed=0, roughness=0.55):
        """A jagged ridge across the full width, filled down to the bottom edge."""
        width, bottom = self.image.size if self.image else (0, 0)
        rng = random.Random(int(seed))
        count = 129
        ys = [0.0] * count
        ys[0], ys[-1] = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)
        step, scale = count - 1, 1.0
        while step > 1:
            half = step // 2
            for left in range(0, count - 1, step):
                ys[left + half] = (ys[left] + ys[left + step]) / 2 + rng.uniform(-scale, scale) * 0.5
            step, scale = half, scale * min(0.9, max(0.1, float(roughness)))
        low, high = min(ys), max(ys)
        points = [(i * width / (count - 1), base - height * (y - low) / max(1e-6, high - low)) for i, y in enumerate(ys)]
        self.polygon(points + [(width, bottom), (0, bottom)], fill)



    def hair_back(self, x, y, fill, style='long'):
        """Hair behind the head: 'long', 'short', 'twintails' or 'ponytail'."""
        if style == 'long':
            self.polygon([(x - 92, y - 40), (x + 92, y - 40), (x + 108, y + 210), (x - 108, y + 210)], fill)
        elif style == 'twintails':
            for side in (-1, 1):
                self.ellipse(x + side * 120, y + 60, 38, 120, fill)
                self.circle(x + side * 92, y - 36, 12, 'red')
        elif style == 'ponytail':
            self.ellipse(x + 96, y + 30, 34, 110, fill)
        elif style != 'short':
            raise PaintError("hair_back style must be 'long', 'short', 'twintails' or 'ponytail'")
        self.ellipse(x, y - 30, 96, 104 if style == 'short' else 98, fill)
        if style == 'short':
            self.polygon([(x - 96, y - 20), (x + 96, y - 20), (x + 92, y + 66), (x - 92, y + 66)], fill)

    def outfit(self, x, y, kind, fill, skin='#ffe3d3'):
        """Neck and shoulders down to the bottom edge: 'uniform' (sailor collar and bow), 'hoodie' or 'dress'."""
        bottom = self.image.size[1] if self.image else 400
        self.rect(x - 22, y + 50, 44, 50, skin)
        self.polygon([(x - 140, bottom), (x - 110, y + 110), (x - 28, y + 92), (x + 28, y + 92), (x + 110, y + 110), (x + 140, bottom)], fill)
        if kind == 'uniform':
            self.polygon([(x - 70, y + 100), (x, y + 170), (x + 70, y + 100), (x + 30, y + 92), (x, y + 130), (x - 30, y + 92)], 'white')
            for side in (-1, 1):
                self.polygon([(x, y + 150), (x + side * 34, y + 132), (x + side * 34, y + 172)], 'red')
            self.circle(x, y + 151, 9, 'crimson')
        elif kind == 'hoodie':
            self.polygon([(x - 60, y + 96), (x, y + 140), (x + 60, y + 96)], mix(fill, 'black', 0.2))
            for side in (-1, 1):
                self.line(x + side * 16, y + 120, x + side * 18, y + 170, 'white', 4)
        elif kind == 'dress':
            self.polygon([(x - 40, y + 94), (x, y + 116), (x + 40, y + 94)], 'white')
        else:
            raise PaintError("outfit kind must be 'uniform', 'hoodie' or 'dress'")

    def face(self, x, y, skin='#ffe3d3'):
        """Face with a narrower chin, blush and a small nose."""
        self.ellipse(x, y - 6, 74, 78, skin)
        self.polygon([(x - 66, y + 10), (x + 66, y + 10), (x + 30, y + 64), (x, y + 74), (x - 30, y + 64)], skin)
        for side in (-1, 1):
            self.ellipse(x + side * 44, y + 34, 14, 6, 'pink/45')
        self.line(x - 2, y + 30, x + 2, y + 34, '#e0a898', 2)

    def anime_eye(self, x, y, iris, side=1, brow='charcoal'):
        """A large eye centered at (x, y) with iris, pupil, highlights, lashes and brow; side -1 is the left eye."""
        side = 1 if side >= 0 else -1
        out = lambda dx: x + side * dx
        self.ellipse(x, y, 20, 26, 'white')
        self.ellipse(x, y + 2, 16, 22, iris)
        self.ellipse(x, y + 10, 14, 12, mix(iris, 'black', 0.35))
        self.ellipse(x, y + 4, 7, 10, 'charcoal')
        self.circle(x - 6, y - 8, 6, 'white')
        self.circle(x + 6, y + 12, 3, 'white/80')
        self.polygon([(out(-20), y - 18), (x, y - 28), (out(22), y - 26), (out(29), y - 17), (out(20), y - 21), (x, y - 23), (out(-20), y - 14)], 'charcoal')
        self.line(out(-16), y - 40, out(14), y - 45, brow, 3)

    def bangs(self, x, y, fill):
        """Top of the head and a zigzag fringe that stops above the eyes."""
        tips = [(80, -30), (62, -46), (46, -24), (28, -48), (12, -26), (-6, -50), (-22, -26), (-40, -48), (-56, -24), (-74, -44), (-84, -20)]
        self.polygon([(x - 92, y - 40), (x - 74, y - 96), (x, y - 120), (x + 74, y - 96), (x + 92, y - 40)] + [(x + dx, y + dy) for dx, dy in tips], fill)
        self.ellipse(x - 30, y - 92, 26, 8, mix(fill, 'white', 0.35))

    def cat_ears(self, x, y, fill):
        for side in (-1, 1):
            self.polygon([(x + side * 84, y - 70), (x + side * 70, y - 150), (x + side * 30, y - 104)], fill)
            self.polygon([(x + side * 74, y - 86), (x + side * 68, y - 132), (x + side * 44, y - 104)], 'pink')

    def hair_bow(self, x, y, fill='red'):
        """A bow on the right side of the head."""
        for side in (-1, 1):
            self.polygon([(x + 60, y - 96), (x + 60 + side * 30, y - 120), (x + 60 + side * 30, y - 72)], fill)
        self.circle(x + 60, y - 96, 8, mix(fill, 'black', 0.25))


SAFE_MATH = {'sin': math.sin, 'cos': math.cos, 'pi': math.pi, 'abs': abs, 'min': min, 'max': max, 'round': round, 'int': int}


class Interpreter:
    def __init__(self, seed=0):
        self.canvas = Canvas()
        self.rng = random.Random(seed)
        self.names = {}
        self.steps = 0
        calls = ('canvas', 'gradient', 'glow', 'rect', 'ellipse', 'circle', 'line', 'polygon', 'mountains',
                 'hair_back', 'outfit', 'face', 'anime_eye', 'bangs', 'cat_ears', 'hair_bow')
        self.functions = {name: getattr(self.canvas, name) for name in calls}
        self.functions.update(SAFE_MATH, mix=mix, rand=lambda a, b: self.rng.uniform(a, b), randint=lambda a, b: self.rng.randint(int(a), int(b)))

    def tick(self):
        self.steps += 1
        if self.steps > MAX_STEPS:
            raise PaintError(f'Program ran more than {MAX_STEPS} steps')

    def run(self, source):
        try:
            tree = ast.parse(source)
        except SyntaxError as error:
            raise PaintError(f'Syntax error on line {error.lineno}: {error.msg}') from None
        except ValueError as error:
            raise PaintError(f'Not a program: {error}') from None
        self.block(tree.body)
        if self.canvas.image is None:
            raise PaintError('Nothing drawn: call canvas(width, height)')
        return self.canvas.image

    def block(self, statements):
        for statement in statements:
            self.tick()
            try:
                self.statement(statement)
            except PaintError as error:
                if str(error).startswith('Line '):
                    raise
                raise PaintError(f'Line {statement.lineno}: {error}') from None
            except (TypeError, ValueError, ZeroDivisionError, OverflowError) as error:
                raise PaintError(f'Line {statement.lineno}: {error}') from None

    def statement(self, node):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            self.expression(node.value)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            self.names[node.targets[0].id] = self.expression(node.value)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
            current = self.names.get(node.target.id)
            if current is None:
                raise PaintError(f'{node.target.id} is not defined')
            self.names[node.target.id] = self.binary(node.op, current, self.expression(node.value))
        elif isinstance(node, ast.For) and isinstance(node.target, ast.Name) and not node.orelse:
            for value in self.expression(node.iter):
                self.names[node.target.id] = value
                self.block(node.body)
        else:
            raise PaintError(f'Unsupported statement: {type(node).__name__}')

    def binary(self, op, a, b):
        if isinstance(a, list) and isinstance(op, ast.Add) and isinstance(b, list):
            if len(a) + len(b) > 5000:
                raise PaintError('Lists are limited to 5000 items')
            return a + b
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (a, b)):
            raise PaintError('Arithmetic works on numbers only')
        operations = {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b, ast.Div: lambda: a / b,
                      ast.FloorDiv: lambda: a // b, ast.Mod: lambda: a % b, ast.Pow: lambda: float(a) ** min(b, 8)}
        if type(op) not in operations:
            raise PaintError(f'Unsupported operator: {type(op).__name__}')
        value = operations[type(op)]()
        if not math.isfinite(value) or abs(value) > 1e9:
            raise PaintError('Number out of range')
        return value

    def expression(self, node):
        self.tick()
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in self.names:
                return self.names[node.id]
            if node.id in SAFE_MATH and not callable(SAFE_MATH[node.id]):
                return SAFE_MATH[node.id]
            raise PaintError(f'{node.id} is not defined')
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = self.expression(node.operand)
            if not isinstance(value, (int, float)):
                raise PaintError('Signs apply to numbers only')
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            return self.binary(node.op, self.expression(node.left), self.expression(node.right))
        if isinstance(node, ast.Subscript):
            items, index = self.expression(node.value), self.expression(node.slice)
            if not isinstance(items, (list, tuple)) or not isinstance(index, int) or isinstance(index, bool) or not -len(items) <= index < len(items):
                raise PaintError('Indexing takes a list and an integer position inside it')
            return items[index]
        if isinstance(node, (ast.Tuple, ast.List)):
            values = [self.expression(item) for item in node.elts]
            return tuple(values) if isinstance(node, ast.Tuple) else values
        if isinstance(node, ast.ListComp) and len(node.generators) == 1:
            generator = node.generators[0]
            if not isinstance(generator.target, ast.Name) or generator.ifs or generator.is_async:
                raise PaintError('List comprehensions take one "for name in range(...)"')
            out = []
            for value in self.expression(generator.iter):
                self.names[generator.target.id] = value
                out.append(self.expression(node.elt))
            return out
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name == 'range':
                bounds = [self.expression(a) for a in node.args]
                if node.keywords or not 1 <= len(bounds) <= 3 or not all(isinstance(b, int) for b in bounds):
                    raise PaintError('range takes 1-3 integers')
                values = range(*bounds)
                if len(values) > MAX_STEPS:
                    raise PaintError('range is too long')
                return values
            if name not in self.functions:
                raise PaintError(f'Unknown function {name}()')
            args = [self.expression(a) for a in node.args]
            kwargs = {k.arg: self.expression(k.value) for k in node.keywords if k.arg}
            if len(kwargs) != len(node.keywords):
                raise PaintError('**kwargs are not supported')
            return self.functions[name](*args, **kwargs)
        raise PaintError(f'Unsupported expression: {type(node).__name__}')


def render(source, seed=0):
    """Run a Kani Paint program and return a Pillow RGB image. Raises PaintError."""
    return Interpreter(seed).run(source)


def code_block(text):
    """The first ```python block of a reply, or the whole reply."""
    match = re.search(r'```(?:python|py)?\n(.*?)```', text, re.S)
    return match.group(1) if match else text



DRAW = re.compile(r'^(?:(?:please|can you|could you|would you)\s+)?(?:draw|paint|sketch)\b'
                  r'|^(?:(?:please|can you|could you)\s+)?(?:make|generate|create|show)\s+(?:me\s+)?(?:a|an)\s+'
                  r'(?:picture|image|drawing|painting|scene)\b', re.I)


SUBJECTS = Path(__file__).with_name('paint_subjects.json')


def unknown_subject(description, subjects=None):
    """A reply for a request that names nothing Kani Painter learned to draw, or None.

    The painter would otherwise draw its nearest practiced scene (a house for "a person"
    before v2), which looks like an answer but is not one.
    """
    if subjects is None:
        if not SUBJECTS.exists():
            return None
        subjects = json.loads(SUBJECTS.read_text(encoding='utf-8'))
    if set(re.findall(r'[a-z]+', description.lower())) & set(subjects['words']):
        return None
    things = subjects['objects']
    return ("Kani Painter hasn't learned to draw that yet, so it would only guess. It can draw "
            f"{', '.join(things[:-1])} and {things[-1]}, in scenes like {', '.join(subjects['settings'][:-1])} "
            f"or {subjects['settings'][-1]}. Try: draw a person in a meadow at sunset.")


def _near(a, b):
    """True when a and b differ by one inserted, deleted or changed letter, or one swap of neighbours."""
    if a == b or abs(len(a) - len(b)) > 1:
        return a == b
    if len(a) == len(b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        return len(diff) == 1 or (len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]])
    short, long = sorted((a, b), key=len)
    i = 0
    while i < len(short) and short[i] == long[i]:
        i += 1
    return short[i:] == long[i + 1:]


def drawing_words():
    """Words of every subject Kani Painter (kelus/paint_subjects.json) or Kani Sketch (Quick, Draw!) can draw."""
    words = set(json.loads(SUBJECTS.read_text(encoding='utf-8'))['words']) if SUBJECTS.exists() else set()
    categories = Path(__file__).resolve().parent.parent / 'data/quickdraw/all-categories.txt'
    if categories.exists():
        words.update(re.findall(r'[a-z]+', categories.read_text().lower()))
    return words


WORDS = Path(__file__).with_name('english_words.txt')


@functools.cache
def english_words():
    """Common words that fix_spelling leaves alone, so "pink hair" never becomes "pink chair"."""
    return frozenset(WORDS.read_text(encoding='utf-8').split()) if WORDS.exists() else frozenset()


def fix_spelling(description, known=None, common=None):
    """'draw me a bnana' -> 'draw me a banana': a word of 4+ letters, not a common word, one typo away from exactly one subject word."""
    known = drawing_words() if known is None else known
    common = english_words() if common is None else common

    def fix(match):
        word = match.group(0)
        lower = word.lower()
        if len(lower) < 4 or lower in known or lower in common:
            return word
        close = [w for w in known if len(w) >= 4 and _near(lower, w)]
        return close[0] if len(close) == 1 else word
    return re.sub(r'[A-Za-z]+', fix, description)


def draw_request(text):
    """The description for Kani Painter, or None: "/draw a red balloon" or "draw a red balloon"."""
    text = text.strip()
    if re.match(r'/draw\b', text):
        return text[5:].strip() or None
    return text if DRAW.search(text) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('program', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    render(code_block(args.program.read_text(encoding='utf-8')), args.seed).save(args.out)
    print(args.out)


if __name__ == '__main__':
    main()
