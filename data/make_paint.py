"""Kani Painter curriculum: drawing requests paired with Kani Paint programs.

Every target is a short, deterministic program in the safe Python subset that
kelus/paint.py renders. Scenes are composed from parts: a setting (ocean, lake,
mountains, desert, forest, city, meadow, snow, space), a time of day, weather and
objects with counts and colors taken from the request. Single-object requests
("draw a red hot air balloon") are included too.

Validation measures composition, not memory: some setting/time and setting/weather
pairs never appear in training (HELD_OUT), so passing requires combining parts that
were only seen separately. Every program is rendered once while building to prove
it runs.

    python3 data/make_paint.py   # writes data/paint-v1.jsonl
"""
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kelus.paint import PaintError, render  # noqa: E402

W, H = 640, 400
NUMBERS = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five'}
TIMES = {
    'day': (['on a sunny day', 'at noon', 'during the day', 'in the daytime'], [('#3b82c4', '#bfe3ff'), ('#4a90d9', '#d6eeff')]),
    'sunset': (['at sunset', 'at golden hour', 'as the sun sets'], [('#1b2a5a', '#ff9a4d'), ('#2b1b4a', '#ff7a59')]),
    'dawn': (['at dawn', 'at sunrise', 'in the early morning'], [('#3a4a7a', '#ffc3a0'), ('#4b5a8f', '#ffd6b0')]),
    'night': (['at night', 'under the stars', 'on a starry night', 'at midnight'], [('#04071a', '#1c2550'), ('#02010a', '#2a1b4a')]),
}
SETTINGS = {
    'ocean': ['over the ocean', 'by the sea', 'on the ocean', 'at the beach'],
    'lake': ['on a mountain lake', 'by a lake in the mountains', 'on a calm lake'],
    'mountains': ['in the mountains', 'of a mountain range', 'of snowy peaks'],
    'desert': ['in the desert', 'of sand dunes', 'in a sandy desert'],
    'forest': ['in a pine forest', 'in the woods', 'of a forest'],
    'city': ['of a city skyline', 'of a big city', 'of downtown skyscrapers'],
    'meadow': ['of a green meadow', 'of rolling hills', 'of a flower field'],
    'snow': ['of a snowy village', 'of a winter landscape', 'of a snowy field'],
    'space': ['in outer space', 'of deep space', 'of a galaxy far away'],
}
WEATHER = {
    'clear': [''],
    'cloudy': ['with clouds', 'on a cloudy day', 'under fluffy clouds'],
    'rainy': ['in the rain', 'while it rains', 'during a rainstorm'],
    'snowing': ['while it snows', 'with falling snow', 'in a snowfall'],
    'foggy': ['in the fog', 'on a foggy day', 'in thick mist'],
}

HELD_OUT = {('desert', 'night'), ('city', 'dawn'), ('meadow', 'sunset'), ('ocean', 'day'), ('snow', 'dawn'),
            ('city', 'snowing'), ('forest', 'rainy'), ('lake', 'foggy')}
COLORS = ['red', 'blue', 'yellow', 'green', 'orange', 'purple', 'pink', 'white', 'teal', 'gold']
VERBS = ['draw', 'paint', 'make a picture of', 'generate an image of', 'can you draw', 'create an image of', 'show me', 'draw me',
         'please draw', 'sketch', 'i want a picture of']


def n(v):
    return str(int(round(v)))



def sailboat(s, sail):
    return [f"polygon([(x - {n(36*s)}, y), (x + {n(36*s)}, y), (x + {n(28*s)}, y + {n(14*s)}), (x - {n(28*s)}, y + {n(14*s)})], 'charcoal')",
            f"line(x, y, x, y - {n(72*s)}, 'charcoal', 3)",
            f"polygon([(x + 3, y - {n(68*s)}), (x + 3, y - 4), (x + {n(38*s)}, y - 4)], '{sail}')",
            f"polygon([(x - 3, y - {n(58*s)}), (x - 3, y - 4), (x - {n(26*s)}, y - 4)], '{sail}/85')"]


def pine(s, snowy=False):
    lines = [f"rect(x - 3, y - 6, 6, 10, 'dark_brown')"]
    for k, (top, half) in enumerate([(100, 26), (78, 32), (54, 38)]):
        lines.append(f"polygon([(x, y - {n(top*s)}), (x - {n(half*s)}, y - {n((top-42)*s)}), (x + {n(half*s)}, y - {n((top-42)*s)})], 'forest')")
        if snowy:
            lines.append(f"polygon([(x, y - {n(top*s)}), (x - {n(half*s/2.2)}, y - {n((top-18)*s)}), (x + {n(half*s/2.2)}, y - {n((top-18)*s)})], 'snow/90')")
    return lines


def tree(s):
    return [f"rect(x - {n(5*s)}, y - {n(40*s)}, {n(10*s)}, {n(40*s)}, 'brown')",
            f"circle(x, y - {n(58*s)}, {n(26*s)}, 'green')", f"circle(x - {n(18*s)}, y - {n(44*s)}, {n(18*s)}, 'forest')",
            f"circle(x + {n(18*s)}, y - {n(46*s)}, {n(19*s)}, 'green')"]


def house(s, wall, dark):
    window = 'yellow' if dark else 'sky/80'
    lines = [f"rect(x - {n(40*s)}, y - {n(50*s)}, {n(80*s)}, {n(50*s)}, '{wall}')",
             f"polygon([(x - {n(48*s)}, y - {n(50*s)}), (x, y - {n(86*s)}), (x + {n(48*s)}, y - {n(50*s)})], 'crimson')",
             f"rect(x - {n(8*s)}, y - {n(26*s)}, {n(16*s)}, {n(26*s)}, 'dark_brown')",
             f"rect(x - {n(32*s)}, y - {n(40*s)}, {n(16*s)}, {n(14*s)}, '{window}')",
             f"rect(x + {n(16*s)}, y - {n(40*s)}, {n(16*s)}, {n(14*s)}, '{window}')"]
    if dark:
        lines.append(f"glow(x, y - {n(30*s)}, {n(60*s)}, 'yellow/25')")
    return lines


def cactus(s):
    return [f"rect(x - {n(7*s)}, y - {n(70*s)}, {n(14*s)}, {n(70*s)}, 'green')", f"circle(x, y - {n(70*s)}, {n(7*s)}, 'green')",
            f"rect(x - {n(24*s)}, y - {n(46*s)}, {n(10*s)}, {n(22*s)}, 'green')", f"rect(x - {n(24*s)}, y - {n(28*s)}, {n(18*s)}, {n(8*s)}, 'green')",
            f"rect(x + {n(14*s)}, y - {n(56*s)}, {n(10*s)}, {n(24*s)}, 'forest')", f"rect(x + {n(6*s)}, y - {n(36*s)}, {n(18*s)}, {n(8*s)}, 'forest')"]


def pyramid(s):
    return [f"polygon([(x - {n(70*s)}, y), (x, y - {n(70*s)}), (x + {n(70*s)}, y)], 'sand')",
            f"polygon([(x, y - {n(70*s)}), (x + {n(70*s)}, y), (x + {n(18*s)}, y)], 'tan')"]


def snowman(s):
    return [f"circle(x, y - {n(22*s)}, {n(24*s)}, 'snow')", f"circle(x, y - {n(60*s)}, {n(17*s)}, 'snow')",
            f"circle(x, y - {n(88*s)}, {n(12*s)}, 'snow')", f"rect(x - {n(11*s)}, y - {n(112*s)}, {n(22*s)}, {n(14*s)}, 'charcoal')",
            f"rect(x - {n(16*s)}, y - {n(100*s)}, {n(32*s)}, 3, 'charcoal')", f"circle(x - 4, y - {n(91*s)}, 2, 'black')",
            f"circle(x + 4, y - {n(91*s)}, 2, 'black')", f"polygon([(x, y - {n(87*s)}), (x + {n(14*s)}, y - {n(85*s)}), (x, y - {n(83*s)})], 'orange')"]


def balloon(s, fabric):
    return [f"ellipse(x, y, {n(30*s)}, {n(36*s)}, '{fabric}')", f"ellipse(x, y, {n(12*s)}, {n(36*s)}, 'white/35')",
            f"polygon([(x - {n(24*s)}, y + {n(20*s)}), (x + {n(24*s)}, y + {n(20*s)}), (x + {n(8*s)}, y + {n(48*s)}), (x - {n(8*s)}, y + {n(48*s)})], '{fabric}')",
            f"line(x - {n(8*s)}, y + {n(48*s)}, x - {n(6*s)}, y + {n(58*s)}, 'dark_brown')", f"line(x + {n(8*s)}, y + {n(48*s)}, x + {n(6*s)}, y + {n(58*s)}, 'dark_brown')",
            f"rect(x - {n(8*s)}, y + {n(58*s)}, {n(16*s)}, {n(11*s)}, 'brown')"]


def lighthouse(s, dark):
    lines = [f"polygon([(x - {n(18*s)}, y), (x - {n(11*s)}, y - {n(120*s)}), (x + {n(11*s)}, y - {n(120*s)}), (x + {n(18*s)}, y)], 'white')",
             f"rect(x - {n(16*s)}, y - {n(40*s)}, {n(32*s)}, {n(14*s)}, 'red')", f"rect(x - {n(13*s)}, y - {n(90*s)}, {n(26*s)}, {n(14*s)}, 'red')",
             f"rect(x - {n(12*s)}, y - {n(136*s)}, {n(24*s)}, {n(16*s)}, 'gold')", f"polygon([(x - {n(15*s)}, y - {n(136*s)}), (x, y - {n(152*s)}), (x + {n(15*s)}, y - {n(136*s)})], 'red')"]
    if dark:
        lines += [f"glow(x, y - {n(128*s)}, {n(90*s)}, 'yellow/60')", f"polygon([(x, y - {n(128*s)}), (x - {n(260*s)}, y - {n(150*s)}), (x - {n(260*s)}, y - {n(110*s)})], 'yellow/20')"]
    return lines


def tent(s, dark):
    lines = [f"polygon([(x - {n(44*s)}, y), (x, y - {n(54*s)}), (x + {n(44*s)}, y)], 'orange')",
             f"polygon([(x - {n(10*s)}, y), (x, y - {n(36*s)}), (x + {n(10*s)}, y)], 'dark_brown')",
             f"rect(x + {n(62*s)}, y - 4, {n(26*s)}, 5, 'brown')",
             f"polygon([(x + {n(64*s)}, y - 4), (x + {n(75*s)}, y - {n(30*s)}), (x + {n(86*s)}, y - 4)], 'orange')",
             f"polygon([(x + {n(69*s)}, y - 4), (x + {n(75*s)}, y - {n(18*s)}), (x + {n(81*s)}, y - 4)], 'yellow')"]
    if dark:
        lines.append(f"glow(x + {n(75*s)}, y - {n(14*s)}, {n(80*s)}, 'orange/45')")
    return lines


def rocket(s):
    return [f"glow(x, y + {n(40*s)}, {n(36*s)}, 'orange/70')",
            f"polygon([(x - {n(9*s)}, y + {n(24*s)}), (x, y + {n(52*s)}), (x + {n(9*s)}, y + {n(24*s)})], 'yellow')",
            f"ellipse(x, y, {n(16*s)}, {n(32*s)}, 'silver')", f"polygon([(x - {n(13*s)}, y - {n(18*s)}), (x, y - {n(46*s)}), (x + {n(13*s)}, y - {n(18*s)})], 'red')",
            f"polygon([(x - {n(15*s)}, y + {n(10*s)}), (x - {n(26*s)}, y + {n(30*s)}), (x - {n(10*s)}, y + {n(24*s)})], 'red')",
            f"polygon([(x + {n(15*s)}, y + {n(10*s)}), (x + {n(26*s)}, y + {n(30*s)}), (x + {n(10*s)}, y + {n(24*s)})], 'red')",
            f"circle(x, y - {n(4*s)}, {n(7*s)}, 'sky')"]


def planet(s, body, ringed):
    lines = [f"glow(x, y, {n(80*s)}, '{body}/30')", f"circle(x, y, {n(46*s)}, '{body}')",
             f"ellipse(x, y - {n(14*s)}, {n(42*s)}, {n(6*s)}, 'white/20')", f"ellipse(x, y + {n(12*s)}, {n(40*s)}, {n(5*s)}, 'black/20')"]
    if ringed:
        lines += [f"ellipse(x, y, {n(86*s)}, {n(14*s)}, 'cream/70')", f"ellipse(x, y, {n(70*s)}, {n(8*s)}, 'black/60')"]
    return lines


def at(v):
    """'+ 5' or '- 5' for an offset from x or y."""
    return f'+ {n(v)}' if v >= 0 else f'- {n(-v)}'


SKIN = ['#ffdbac', '#f1c27d', '#e0ac69', '#c68642', '#8d5524']
HAIR = ['dark_brown', 'black', 'brown', 'gold', 'charcoal']


def person(s, shirt, skin, hair, pants, long_hair):
    lines = [f"ellipse(x, y - {n(62*s)}, {n(12*s)}, {n(16*s)}, '{hair}')"] if long_hair else []
    return lines + [
        f"line(x - {n(5*s)}, y - {n(30*s)}, x - {n(7*s)}, y, '{pants}', {n(6*s)})",
        f"line(x + {n(5*s)}, y - {n(30*s)}, x + {n(7*s)}, y, '{pants}', {n(6*s)})",
        f"polygon([(x - {n(12*s)}, y - {n(58*s)}), (x + {n(12*s)}, y - {n(58*s)}), (x + {n(10*s)}, y - {n(28*s)}), (x - {n(10*s)}, y - {n(28*s)})], '{shirt}')",
        f"line(x - {n(11*s)}, y - {n(55*s)}, x - {n(20*s)}, y - {n(34*s)}, '{shirt}', {n(5*s)})",
        f"line(x + {n(11*s)}, y - {n(55*s)}, x + {n(22*s)}, y - {n(70*s)}, '{shirt}', {n(5*s)})",
        f"circle(x - {n(20*s)}, y - {n(33*s)}, {n(3*s)}, '{skin}')", f"circle(x + {n(22*s)}, y - {n(71*s)}, {n(3*s)}, '{skin}')",
        f"circle(x, y - {n(68*s)}, {n(10*s)}, '{skin}')", f"ellipse(x, y - {n(75*s)}, {n(10*s)}, {n(5*s)}, '{hair}')",
        f"circle(x - {n(3.5*s)}, y - {n(68*s)}, {n(max(1, 1.3*s))}, 'charcoal')", f"circle(x + {n(3.5*s)}, y - {n(68*s)}, {n(max(1, 1.3*s))}, 'charcoal')"]


def cat(s, fur):
    return [f"ellipse(x, y - {n(16*s)}, {n(16*s)}, {n(16*s)}, '{fur}')",
            f"line(x + {n(14*s)}, y - {n(6*s)}, x + {n(30*s)}, y - {n(26*s)}, '{fur}', {n(5*s)})",
            f"circle(x, y - {n(38*s)}, {n(12*s)}, '{fur}')",
            f"polygon([(x - {n(11*s)}, y - {n(42*s)}), (x - {n(9*s)}, y - {n(56*s)}), (x - {n(2*s)}, y - {n(48*s)})], '{fur}')",
            f"polygon([(x + {n(11*s)}, y - {n(42*s)}), (x + {n(9*s)}, y - {n(56*s)}), (x + {n(2*s)}, y - {n(48*s)})], '{fur}')",
            f"circle(x - {n(4*s)}, y - {n(40*s)}, {n(2*s)}, 'lime')", f"circle(x + {n(4*s)}, y - {n(40*s)}, {n(2*s)}, 'lime')",
            f"polygon([(x - {n(2*s)}, y - {n(35*s)}), (x + {n(2*s)}, y - {n(35*s)}), (x, y - {n(33*s)})], 'pink')"]


def dog(s, fur):
    legs = [f"line(x {at(dx*s)}, y - {n(16*s)}, x {at(dx*s)}, y, '{fur}', {n(5*s)})" for dx in (-16, -8, 10, 18)]
    return legs + [f"ellipse(x, y - {n(22*s)}, {n(24*s)}, {n(11*s)}, '{fur}')",
                   f"line(x - {n(22*s)}, y - {n(26*s)}, x - {n(34*s)}, y - {n(40*s)}, '{fur}', {n(4*s)})",
                   f"circle(x + {n(24*s)}, y - {n(36*s)}, {n(11*s)}, '{fur}')",
                   f"ellipse(x + {n(34*s)}, y - {n(32*s)}, {n(7*s)}, {n(5*s)}, '{fur}')",
                   f"circle(x + {n(40*s)}, y - {n(33*s)}, {n(2.5*s)}, 'black')",
                   f"ellipse(x + {n(19*s)}, y - {n(36*s)}, {n(4*s)}, {n(9*s)}, 'dark_brown')",
                   f"circle(x + {n(27*s)}, y - {n(39*s)}, {n(2*s)}, 'black')"]


def car(s, paint):
    lines = [f"polygon([(x - {n(24*s)}, y - {n(22*s)}), (x - {n(14*s)}, y - {n(38*s)}), (x + {n(16*s)}, y - {n(38*s)}), (x + {n(28*s)}, y - {n(22*s)})], '{paint}')",
             f"rect(x - {n(44*s)}, y - {n(24*s)}, {n(88*s)}, {n(16*s)}, '{paint}')",
             f"polygon([(x - {n(18*s)}, y - {n(24*s)}), (x - {n(11*s)}, y - {n(35*s)}), (x - {n(2*s)}, y - {n(35*s)}), (x - {n(2*s)}, y - {n(24*s)})], 'sky/80')",
             f"polygon([(x + {n(3*s)}, y - {n(24*s)}), (x + {n(3*s)}, y - {n(35*s)}), (x + {n(14*s)}, y - {n(35*s)}), (x + {n(22*s)}, y - {n(24*s)})], 'sky/80')"]
    for dx in (-26, 26):
        lines += [f"circle(x {at(dx*s)}, y - {n(8*s)}, {n(9*s)}, 'charcoal')", f"circle(x {at(dx*s)}, y - {n(8*s)}, {n(4*s)}, 'silver')"]
    return lines + [f"rect(x + {n(38*s)}, y - {n(20*s)}, {n(6*s)}, {n(4*s)}, 'yellow')"]


def flower(s, petals):
    lines = [f"line(x, y, x, y - {n(40*s)}, 'green', {n(3*s)})", f"ellipse(x + {n(7*s)}, y - {n(16*s)}, {n(7*s)}, {n(3*s)}, 'green')"]
    for dx, dy in [(0, 9), (8.5, 2.8), (5.3, -7.3), (-5.3, -7.3), (-8.5, 2.8)]:
        lines.append(f"circle(x {at(dx*s)}, y - {n((40 + dy)*s)}, {n(6*s)}, '{petals}')")
    return lines + [f"circle(x, y - {n(40*s)}, {n(4*s)}, 'gold')"]


def bird(s):
    return [f"line(x - {n(12*s)}, y - {n(6*s)}, x, y, 'charcoal', {n(2*s)})", f"line(x, y, x + {n(12*s)}, y - {n(6*s)}, 'charcoal', {n(2*s)})"]


def castle(s, dark):
    window = 'yellow' if dark else 'charcoal'
    return [f"rect(x - {n(50*s)}, y - {n(60*s)}, {n(100*s)}, {n(60*s)}, 'silver')",
            f"rect(x - {n(62*s)}, y - {n(86*s)}, {n(26*s)}, {n(86*s)}, 'gray')", f"rect(x + {n(36*s)}, y - {n(86*s)}, {n(26*s)}, {n(86*s)}, 'gray')",
            f"for k in range(5):", f"    rect(x - {n(46*s)} + k * {n(20*s)}, y - {n(68*s)}, {n(10*s)}, {n(8*s)}, 'silver')",
            f"polygon([(x - {n(66*s)}, y - {n(86*s)}), (x - {n(49*s)}, y - {n(112*s)}), (x - {n(32*s)}, y - {n(86*s)})], 'navy')",
            f"polygon([(x + {n(32*s)}, y - {n(86*s)}), (x + {n(49*s)}, y - {n(112*s)}), (x + {n(66*s)}, y - {n(86*s)})], 'navy')",
            f"rect(x - {n(53*s)}, y - {n(66*s)}, {n(8*s)}, {n(12*s)}, '{window}')", f"rect(x + {n(45*s)}, y - {n(66*s)}, {n(8*s)}, {n(12*s)}, '{window}')",
            f"polygon([(x - {n(14*s)}, y), (x - {n(14*s)}, y - {n(28*s)}), (x, y - {n(38*s)}), (x + {n(14*s)}, y - {n(28*s)}), (x + {n(14*s)}, y)], 'dark_brown')",
            f"line(x - {n(49*s)}, y - {n(112*s)}, x - {n(49*s)}, y - {n(130*s)}, 'charcoal', 2)",
            f"polygon([(x - {n(49*s)}, y - {n(130*s)}), (x - {n(33*s)}, y - {n(125*s)}), (x - {n(49*s)}, y - {n(120*s)})], 'red')"]


def cloud(s, color):
    return [f"ellipse(x, y, {n(46*s)}, {n(16*s)}, '{color}')", f"circle(x - {n(16*s)}, y - {n(10*s)}, {n(18*s)}, '{color}')",
            f"circle(x + {n(12*s)}, y - {n(16*s)}, {n(22*s)}, '{color}')"]


OBJECTS = {

    'sailboat': ('sailboat', 'sailboats', {'ocean', 'lake'}),
    'lighthouse': ('lighthouse', None, {'ocean'}),
    'pine': ('pine tree', 'pine trees', {'forest', 'mountains', 'snow', 'lake'}),
    'tree': ('tree', 'trees', {'meadow'}),
    'house': ('house', 'houses', {'meadow', 'snow', 'mountains', 'forest'}),
    'cactus': ('cactus', 'cacti', {'desert'}),
    'pyramid': ('pyramid', 'pyramids', {'desert'}),
    'snowman': ('snowman', 'snowmen', {'snow'}),
    'balloon': ('hot air balloon', 'hot air balloons', {'meadow', 'desert', 'mountains', 'city', 'lake'}),
    'tent': ('tent with a campfire', None, {'forest', 'mountains', 'desert'}),
    'rocket': ('rocket', 'rockets', {'space'}),
    'planet': ('planet', 'planets', {'space'}),
    'person': ('person', 'people', {'meadow', 'snow', 'forest', 'city', 'desert', 'mountains'}),
    'cat': ('cat', 'cats', {'meadow', 'snow', 'city', 'forest'}),
    'dog': ('dog', 'dogs', {'meadow', 'snow', 'city', 'forest', 'mountains'}),
    'car': ('car', 'cars', {'city', 'desert', 'meadow', 'snow'}),
    'flower': ('flower', 'flowers', {'meadow'}),
    'bird': ('bird', 'birds', {'ocean', 'lake', 'meadow', 'mountains', 'desert', 'city', 'forest', 'snow'}),
    'castle': ('castle', None, {'meadow', 'mountains', 'snow', 'forest'}),
}

ALIASES = {'person': [('person', 'people'), ('human', 'humans'), ('man', 'men'), ('woman', 'women'), ('kid', 'kids')],
           'cat': [('cat', 'cats'), ('kitten', 'kittens')], 'dog': [('dog', 'dogs'), ('puppy', 'puppies')]}
COLORED = {'house', 'balloon', 'sailboat', 'planet', 'person', 'cat', 'dog', 'car', 'flower'}
PALETTE = {'cat': ['orange', 'gray', 'black', 'white'], 'dog': ['brown', 'tan', 'gold', 'white', 'black'],
           'flower': ['red', 'yellow', 'pink', 'purple', 'white', 'orange', 'blue']}
SINGLE = {'lighthouse', 'tent', 'castle'}
VERSION = 3


def noun(rng, name, count):
    singular, plural = rng.choice(ALIASES.get(name, [OBJECTS[name][:2]]))
    return singular if count == 1 or plural is None else plural


def phrase(rng, name, count, color):
    """'two red cars', 'a person in a blue shirt'."""
    word = noun(rng, name, count)
    adjective = f'{color} ' if color and name != 'person' else ''
    quantity = ('an' if (adjective or word)[0] in 'aeiou' else 'a') if count == 1 else NUMBERS[count]
    shirt = (f' in a {color} shirt' if count == 1 else f' in {color} shirts') if color and name == 'person' else ''
    return f'{quantity} {adjective}{word}{shirt}'


def subject(rng, name, s, color, dark, setting=None):
    """Code lines for one object at x, y."""
    return {'sailboat': lambda: sailboat(s, color or 'white'), 'lighthouse': lambda: lighthouse(s, dark),
            'pine': lambda: pine(s, setting == 'snow'), 'tree': lambda: tree(s), 'house': lambda: house(s, color or 'tan', dark),
            'cactus': lambda: cactus(s), 'pyramid': lambda: pyramid(s), 'snowman': lambda: snowman(s), 'tent': lambda: tent(s, dark),
            'balloon': lambda: balloon(s, color or 'red'), 'rocket': lambda: rocket(s), 'planet': lambda: planet(s, color or 'orange', True),
            'person': lambda: person(s, color or rng.choice(COLORS), rng.choice(SKIN), rng.choice(HAIR),
                                     rng.choice(['navy', 'charcoal', 'blue', 'dark_brown']), rng.random() < 0.4),
            'cat': lambda: cat(s, color or rng.choice(PALETTE['cat'])), 'dog': lambda: dog(s, color or rng.choice(PALETTE['dog'])),
            'car': lambda: car(s, color or rng.choice(COLORS)), 'flower': lambda: flower(s, color or rng.choice(PALETTE['flower'])),
            'bird': lambda: bird(s), 'castle': lambda: castle(s, dark)}[name]()


class Scene:
    def __init__(self, rng):
        self.rng, self.lines = rng, ['canvas(640, 400)']

    def add(self, *lines):
        self.lines += lines

    def place(self, count, xs, y, body, jitter=20):
        """Draw `body` (lines using x, y) count times, spread across xs = (left, right)."""
        left, right = xs
        if count == 1:
            self.add(f'x = {n(self.rng.uniform(left, right))}', f'y = {n(y)}', *body)
            return
        step = (right - left) / max(1, count - 1)
        self.add(f'for i in range({count}):', f'    x = {n(left)} + i * {n(step)} + rand(-{jitter}, {jitter})', f'    y = {n(y)}',
                 *['    ' + line for line in body])


def sky(scene, time, weather):
    rng = scene.rng
    top, bottom = rng.choice(TIMES[time][1])
    if weather in ('rainy', 'foggy'):
        top, bottom = {'night': ('#0b0f1a', '#2a2f3d')}.get(time, ('#4a5260', '#a7b0bd'))
    scene.add(f"gradient(0, 0, 640, 400, '{top}', '{bottom}')")
    if time == 'night' and weather not in ('rainy', 'foggy'):
        scene.add(f'for i in range({rng.choice([90, 120, 160])}):', "    circle(rand(0, 640), rand(0, 220), rand(0.4, 1.5), 'white/80')")
    return top, bottom


def light(scene, time, weather, horizon):
    rng = scene.rng
    if weather in ('rainy', 'foggy'):
        return None
    x = rng.choice([110, 160, 470, 520])
    if time == 'night':
        y = rng.choice([70, 90])
        scene.add(f"glow({x}, {y}, 110, 'snow/30')", f"circle({x}, {y}, 28, 'snow')")
    elif time == 'day':
        y = rng.choice([70, 90])
        scene.add(f"glow({x}, {y}, 140, 'yellow/40')", f"circle({x}, {y}, 34, 'yellow')")
    else:
        y = horizon - 16
        scene.add(f"glow({x}, {y}, 220, 'peach/60')", f"circle({x}, {y}, 40, 'cream')")
    return x


def ground(scene, setting, time, top, bottom, sun_x):
    rng, dark = scene.rng, time == 'night'
    horizon = 260
    if setting in ('ocean', 'lake'):
        if setting == 'lake':
            scene.add(f"mountains(262, 120, '{'indigo' if dark else 'purple'}', seed={rng.randint(1, 99)})",
                      f"mountains(262, 70, '{'navy' if dark else 'indigo'}', seed={rng.randint(1, 99)})")
        scene.add(f"gradient(0, 260, 640, 140, '{bottom}', '{top}')")
        if sun_x is not None:
            scene.add('for i in range(24):', f"    rect({sun_x} - rand(10, 60), 266 + i * 5, rand(20, 120), 2, 'cream/40')")
    elif setting == 'mountains':
        scene.add(f"mountains(240, 150, '{'indigo' if dark else 'silver'}', seed={rng.randint(1, 99)})",
                  f"mountains(270, 100, '{'navy' if dark else 'purple'}', seed={rng.randint(1, 99)})",
                  f"gradient(0, 300, 640, 100, '{'dark_green' if dark else 'green'}', 'forest')")
        horizon = 320
    elif setting == 'desert':
        scene.add(f"mountains(280, 50, 'tan', seed={rng.randint(1, 99)}, roughness=0.3)",
                  "gradient(0, 300, 640, 100, 'sand', 'tan')")
        horizon = 330
    elif setting == 'forest':
        scene.add(f"mountains(250, 110, '{'navy' if dark else 'forest'}', seed={rng.randint(1, 99)})",
                  f"gradient(0, 300, 640, 100, '{'dark_green' if dark else 'green'}', 'dark_green')")
        horizon = 340
    elif setting == 'city':
        scene.add(f'for i in range({rng.choice([18, 22, 26])}):', '    w = rand(22, 46)', '    h = rand(80, 230)',
                  f"    rect(i * 30 - 10, 400 - h, w, h, mix('navy', 'charcoal', rand(0, 1)))")
        if dark:
            scene.add('    for r in range(int(h // 16)):', "        rect(i * 30 - 4, 406 - h + r * 16, 6, 7, mix('navy', 'yellow', rand(0, 1)))",
                      "        rect(i * 30 + 6, 406 - h + r * 16, 6, 7, mix('navy', 'yellow', rand(0, 1)))")
        horizon = 400
    elif setting == 'meadow':
        scene.add(f"mountains(290, 60, '{'forest' if dark else 'lime'}', seed={rng.randint(1, 99)}, roughness=0.25)",
                  f"gradient(0, 320, 640, 80, '{'forest' if dark else 'green'}', 'forest')")
        horizon = 350
    elif setting == 'snow':
        scene.add(f"mountains(250, 90, '{'navy' if dark else 'silver'}', seed={rng.randint(1, 99)})",
                  f"gradient(0, 290, 640, 110, 'snow', '{'silver' if dark else 'sky'}')")
        horizon = 340
    return horizon


def weather_layer(scene, weather):
    rng = scene.rng
    if weather == 'cloudy':
        scene.place(rng.choice([3, 4, 5]), (60, 580), rng.choice([60, 80]), cloud(rng.uniform(0.9, 1.4), 'white/90'), jitter=40)
    elif weather == 'rainy':
        scene.place(4, (40, 600), 60, cloud(1.6, 'gray/90'), jitter=30)
        scene.add('for i in range(160):', '    x = rand(0, 640)', '    y = rand(60, 400)', "    line(x, y, x - 4, y + 14, 'sky/50')")
    elif weather == 'snowing':
        scene.add('for i in range(220):', "    circle(rand(0, 640), rand(0, 400), rand(0.8, 2.2), 'snow/90')")
    elif weather == 'foggy':
        scene.add("gradient(0, 120, 640, 280, 'silver/0', 'silver/80')")


GROUND = ('pine', 'tree', 'house', 'cactus', 'pyramid', 'snowman', 'tent', 'person', 'cat', 'dog', 'car', 'flower', 'castle')
SKY = ('balloon', 'rocket', 'planet', 'bird')
WIDTH = {'house': 100, 'snowman': 60, 'pyramid': 150, 'cactus': 60, 'sailboat': 80, 'tent': 180, 'tree': 60, 'pine': 30,
         'balloon': 70, 'rocket': 60, 'planet': 120, 'person': 50, 'cat': 50, 'dog': 80, 'car': 100, 'flower': 30, 'bird': 40, 'castle': 140}


def layout(picks):
    """Give each group of objects its own horizontal lane so groups never overlap."""
    lanes = {}
    for layer in (GROUND + ('sailboat',), SKY):
        names = [name for name, _, _ in picks if name in layer]
        right = 470 if any(name == 'lighthouse' for name, _, _ in picks) else 570
        width = (right - 70) / max(1, len(names))
        for k, name in enumerate(names):
            margin = min(width / 2 - 1, WIDTH[name] / 2)
            lanes[name] = (70 + k * width + margin, 70 + (k + 1) * width - margin)
    lanes['lighthouse'] = (550, 580)
    return lanes


def fits(name, count, groups):
    """Largest count of `name` that fits side by side in its lane."""
    lane = 500 / max(1, groups)
    return max(1, min(count, int(lane // WIDTH.get(name, 60))))


def build_scene(rng, setting, time, weather, picks):
    """picks: list of (object, count, color)."""
    scene = Scene(rng)
    dark = time == 'night'
    if setting == 'space':
        scene.add("gradient(0, 0, 640, 400, '#02010a', '#120a2e')", f"glow({rng.randint(100, 300)}, {rng.randint(200, 320)}, 280, 'purple/25')",
                  f"glow({rng.randint(380, 560)}, {rng.randint(80, 200)}, 240, 'teal/25')",
                  f'for i in range({rng.choice([300, 400])}):', "    circle(rand(0, 640), rand(0, 400), rand(0.3, 1.4), mix('white', 'sky', rand(0, 1)))")
        horizon = 400
    else:
        top, bottom = sky(scene, time, weather)
        sun_x = light(scene, time, weather, 260)
        horizon = ground(scene, setting, time, top, bottom, sun_x)
    lanes = layout(picks)
    for name, count, color in picks:
        s = rng.choice([0.9, 1.0, 1.2])
        xs = lanes[name]
        if name == 'sailboat':
            scene.place(count, xs, rng.choice([300, 320]), sailboat(s, color or 'white'))
        elif name == 'lighthouse':
            scene.place(1, xs, 290, lighthouse(1.0, dark))
        elif name in GROUND:
            body = subject(rng, name, s, color, dark, setting)
            base = horizon + (40 if setting in ('meadow', 'snow', 'forest') else 10) if setting != 'lake' else 262
            scene.place(count, xs, base, body)
        elif name == 'balloon':
            scene.place(count, xs, rng.choice([110, 140]), balloon(s, color or 'red'), jitter=20)
        elif name == 'rocket':
            scene.place(count, xs, rng.choice([150, 220]), rocket(s), jitter=20)
        elif name == 'planet':
            scene.place(count, xs, rng.choice([170, 220]), planet(s, color or 'orange', ringed=count == 1), jitter=20)
        elif name == 'bird':
            scene.place(count, xs, rng.choice([80, 110]), bird(s), jitter=20)
    if setting != 'space':
        weather_layer(scene, weather)
        if dark and setting not in ('city',) and weather == 'clear' and rng.random() < 0.3:
            scene.add("gradient(0, 0, 640, 400, 'navy/0', 'navy/30')")
    return '\n'.join(scene.lines) + '\n'



WHERE = {'ocean': ['over the ocean', 'by the sea', 'on the ocean', 'at the beach'], 'lake': ['on a mountain lake', 'by a lake', 'on a calm lake'],
         'mountains': ['in the mountains', 'on a mountain'], 'desert': ['in the desert', 'on the sand dunes'],
         'forest': ['in a forest', 'in the woods', 'in a pine forest'], 'city': ['in the city', 'in a big city', 'downtown'],
         'meadow': ['in a meadow', 'on a green hill', 'in a flower field'], 'snow': ['in the snow', 'in a snowy village', 'in winter'],
         'space': ['in space', 'in outer space']}

MOMENT = {'sunset': ['a sunset', 'a beautiful sunset'], 'dawn': ['a sunrise'], 'night': ['a starry night', 'a night sky']}


def describe(rng, setting, time, weather, picks):
    """One of three phrasings: scene first, objects first, or the time of day first."""
    parts = [phrase(rng, name, count, color) for name, count, color in picks]
    listed = parts[0] if len(parts) == 1 else ', '.join(parts[:-1]) + ' and ' + parts[-1] if parts else ''
    when = '' if setting == 'space' else ' ' + rng.choice(TIMES[time][0])
    sky_words = rng.choice(WEATHER[weather]) if setting != 'space' else ''
    weather_words = (' ' + sky_words) if sky_words else ''
    verb = rng.choice(VERBS)
    style = rng.random()
    if parts and style < 0.4:
        text = f"{verb} {listed} {rng.choice(WHERE[setting])}{when}{weather_words}"
    elif time in MOMENT and setting != 'space' and style < 0.6:
        objects = f' with {listed}' if parts else ''
        text = f"{verb} {rng.choice(MOMENT[time])} {rng.choice(WHERE[setting])}{objects}{weather_words}"
    else:
        article = 'a picture' if verb in ('draw', 'paint', 'can you draw', 'draw me', 'show me', 'please draw', 'sketch') else 'a scene'
        objects = f' with {listed}' if parts else ''
        text = f"{verb} {article} {rng.choice(SETTINGS[setting])}{when}{objects}{weather_words}"
    return text[0].upper() + text[1:] if rng.random() < 0.3 else text


def sample(rng, split):
    while True:
        setting = rng.choice(list(SETTINGS))
        time = 'night' if setting == 'space' else rng.choice(list(TIMES))
        weather = 'clear' if setting == 'space' else rng.choice(['clear', 'clear', 'cloudy', 'rainy', 'snowing', 'foggy'])
        if setting == 'desert' and weather == 'snowing':
            continue
        held = (setting, time) in HELD_OUT or (setting, weather) in HELD_OUT
        if held != (split == 'validation'):
            continue
        options = [o for o, (_, _, where) in OBJECTS.items() if setting in where]
        chosen = rng.sample(options, min(len(options), rng.choice([1, 1, 2, 2, 3])))
        picks = []
        for name in chosen:
            layer = SKY if name in SKY else GROUND + ('sailboat',)
            count = 1 if name in SINGLE else rng.choice([1, 2, 3, 4, 5])
            if name != 'pine':
                count = fits(name, count, sum(other in layer for other in chosen))
            color = rng.choice(PALETTE.get(name, COLORS)) if name in COLORED and rng.random() < 0.6 else None
            picks.append((name, count, color))
        if rng.random() < 0.1:
            picks = []
        return setting, time, weather, picks


def single(rng, split):
    """One object, drawn large: 'draw a red hot air balloon at night'."""
    name = rng.choice(list(OBJECTS))
    time = rng.choice(['day', 'day', 'sunset', 'night'])
    color = rng.choice(PALETTE.get(name, COLORS)) if name in COLORED and rng.random() < 0.7 else None
    scene, dark, s = Scene(rng), time == 'night', rng.choice([1.6, 1.8, 2.0])
    top, bottom = rng.choice(TIMES[time][1])
    scene.add(f"gradient(0, 0, 640, 400, '{top}', '{bottom}')")
    space = name in ('rocket', 'planet')
    if name in SKY:
        if space:
            scene.lines[-1] = "gradient(0, 0, 640, 400, '#02010a', '#120a2e')"
            scene.add('for i in range(200):', "    circle(rand(0, 640), rand(0, 400), rand(0.3, 1.4), 'white/80')")
        scene.place(1, (320, 320), 160, subject(rng, name, s * (2.5 if name == 'bird' else 1), color, dark))
    else:
        ground_color = {'cactus': 'sand', 'pyramid': 'sand', 'snowman': 'snow', 'car': 'gray', 'sailboat': None, 'lighthouse': None}.get(name, 'green')
        if ground_color:
            scene.add(f"gradient(0, 330, 640, 70, '{ground_color}', '{'charcoal' if dark else ground_color}')")
        else:
            scene.add(f"gradient(0, 300, 640, 100, '{bottom}', '{top}')")
        scale = {'tent': 0.8, 'lighthouse': 0.75, 'castle': 0.9}.get(name, 1)
        scene.place(1, (300, 300) if name == 'tent' else (320, 320), 340 if ground_color else 310, subject(rng, name, s * scale, color, dark))
    when = '' if space else rng.choice(['', '', ' ' + rng.choice(TIMES[time][0])])
    if not when and time != 'day' and not space:
        when = ' ' + rng.choice(TIMES[time][0])
    prompt = f"{rng.choice(VERBS)} {phrase(rng, name, 1, color)}{when}"
    return {'split': split, 'family': 'paint', 'scene': {'object': name, 'time': time, 'color': color},
            'messages': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': '```python\n' + '\n'.join(scene.lines) + '\n```'}]}


def record(rng, split):
    setting, time, weather, picks = sample(rng, split)
    program = build_scene(rng, setting, time, weather, picks)
    return {'split': split, 'family': 'paint', 'scene': {'setting': setting, 'time': time, 'weather': weather,
            'objects': [list(p) for p in picks]},
            'messages': [{'role': 'user', 'content': describe(rng, setting, time, weather, picks)},
                         {'role': 'assistant', 'content': '```python\n' + program + '```'}]}


STOP = {'a', 'an', 'the', 'of', 'in', 'on', 'at', 'by', 'with', 'under', 'during', 'while', 'it', 'as', 'far', 'away', 'big', 'calm',
        'over', 'day', 'green', 'golden', 'hot', 'air', 'early', 'deep', 'outer', 'rolling', 'sets', 'thick', 'fluffy', 'falling', 'hour', 'range', 'beautiful'}


def subjects():
    """What Kani Painter knows, for telling people when a request is outside it (kelus/paint_subjects.json)."""
    words = set()
    for name, (singular, plural, _) in OBJECTS.items():
        for pair in ALIASES.get(name, []) + [(singular, plural)]:
            words.update(w for w in ' '.join(w for w in pair if w).split() if w not in STOP)
    for phrases in list(SETTINGS.values()) + [t[0] for t in TIMES.values()] + list(WEATHER.values()) + list(WHERE.values()) + list(MOMENT.values()):
        for text in phrases:
            words.update(w for w in text.split() if w not in STOP)
    words.update(['sun', 'moon', 'sky', 'sea', 'sunrise', 'stars', 'snow', 'rain', 'fog', 'clouds', 'mountain', 'hill', 'field'])
    names = [plural or f'a {singular}' for singular, plural, _ in OBJECTS.values()]
    return {'version': VERSION, 'objects': names, 'settings': list(SETTINGS), 'words': sorted(words)}


def build(seed=2032, train=30000, validation=600, objects=9000, check=1500, max_tokens=900):
    from kelus.tokenizer import from_dict
    tokenizer = from_dict(json.loads((ROOT / 'data/foundation/tokenizer.json').read_text()))
    rng = random.Random(seed)
    rows = [record(rng, 'train') for _ in range(train)] + [record(rng, 'validation') for _ in range(validation)]
    rows += [single(rng, 'train') for _ in range(objects)]
    unique = {}
    for row in rows:
        unique.setdefault(row['messages'][0]['content'], row)

    rows = [r for r in unique.values() if sum(len(tokenizer.encode(m['content'])) for m in r['messages']) <= max_tokens]
    for row in rows[:check] + [r for r in rows if r['split'] == 'validation']:
        render(row['messages'][1]['content'][10:-3])
    rng.shuffle(rows)
    return rows


if __name__ == '__main__':
    rows = build()
    destination = ROOT / f'data/paint-v{VERSION}.jsonl'
    destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    (ROOT / 'kelus/paint_subjects.json').write_text(json.dumps(subjects(), indent=1) + '\n', encoding='utf-8')
    counts = {}
    for row in rows:
        counts[row['split']] = counts.get(row['split'], 0) + 1
    print(json.dumps({'path': str(destination), 'records': len(rows), 'counts': counts}))
