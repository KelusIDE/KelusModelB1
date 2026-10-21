"""Rendered UI screenshots with exact region masks for kelus.vision.

Each screen is an original HTML layout (IDE, terminal app, minimal editor) filled
with real text from this repository: source code, the file tree, and training
logs. Headless Chrome renders it at desktop sizes. The generator places every
region itself, so each mask is exact rather than hand-drawn.

Validation uses a layout family that never appears in training (editor and
terminal side by side, sidebar on the right), so its score measures layouts the
model has not seen. These are still rendered pages, not real app screenshots.

    python3 data/make_ui_screens.py            # writes data/ui-render/ (gitignored)
    python3 -m kelus.vision train --data data/ui-render/manifest.jsonl --out checkpoints/vision-ui-render.npz
"""
import argparse
import html
import json
from pathlib import Path
import random
import subprocess
import tempfile

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
CLASSES = {'background': 0, 'toolbar': 1, 'sidebar': 2, 'editor': 3, 'terminal': 4, 'dialog': 5}
SIZES = [(1280, 800), (1440, 900), (1024, 768), (1366, 768), (1600, 1000)]
THEMES = [
    # desktop, title, activity, sidebar, editor, tabs, panel, status, fg, muted, accent, keyword, string, comment, dialog, border
    ('dark modern', '#3a3f4b', '#1f1f1f', '#181818', '#181818', '#1f1f1f', '#181818', '#181818', '#0078d4', '#cccccc', '#9d9d9d', '#0078d4', '#c586c0', '#ce9178', '#6a9955', '#252526', '#2b2b2b'),
    ('light modern', '#9bb0c7', '#f8f8f8', '#f8f8f8', '#f8f8f8', '#ffffff', '#f8f8f8', '#ffffff', '#f8f8f8', '#3b3b3b', '#767676', '#005fb8', '#af00db', '#a31515', '#008000', '#ffffff', '#e5e5e5'),
    ('monokai', '#2d2a2e', '#221f22', '#19181a', '#221f22', '#2d2a2e', '#221f22', '#19181a', '#ff6188', '#fcfcfa', '#939293', '#ffd866', '#ff6188', '#ffd866', '#727072', '#363337', '#403e41'),
    ('solarized dark', '#073642', '#002b36', '#00212b', '#00212b', '#002b36', '#00212b', '#002b36', '#073642', '#93a1a1', '#586e75', '#268bd2', '#859900', '#2aa198', '#586e75', '#073642', '#0d3a45'),
    ('solarized light', '#93a1a1', '#eee8d5', '#ddd6c1', '#eee8d5', '#fdf6e3', '#eee8d5', '#fdf6e3', '#eee8d5', '#657b83', '#93a1a1', '#268bd2', '#859900', '#2aa198', '#93a1a1', '#fdf6e3', '#d3cbb7'),
    ('high contrast', '#000000', '#000000', '#000000', '#000000', '#000000', '#000000', '#000000', '#000000', '#ffffff', '#ffffff', '#f38518', '#569cd6', '#ce9178', '#7ca668', '#0c141f', '#6fc3df'),
    ('nord', '#3b4252', '#2e3440', '#2e3440', '#3b4252', '#2e3440', '#3b4252', '#2e3440', '#3b4252', '#d8dee9', '#7b88a1', '#88c0d0', '#81a1c1', '#a3be8c', '#616e88', '#434c5e', '#4c566a'),
    ('github light', '#d0d7de', '#f6f8fa', '#eaeef2', '#f6f8fa', '#ffffff', '#f6f8fa', '#ffffff', '#f6f8fa', '#1f2328', '#59636e', '#0969da', '#cf222e', '#0a3069', '#59636e', '#ffffff', '#d0d7de'),
]
KEYS = ['name', 'desktop', 'title', 'activity', 'sidebar', 'editor', 'tabs', 'panel', 'status', 'fg', 'muted', 'accent',
        'keyword', 'string', 'comment', 'dialog', 'border']
UI_FONTS = ['-apple-system, sans-serif', 'Helvetica, sans-serif', 'Arial, sans-serif', 'Verdana, sans-serif']
CODE_FONTS = ['Menlo, monospace', 'Monaco, monospace', '"Courier New", monospace', '"SF Mono", Menlo, monospace']
KEYWORDS = {'def', 'class', 'return', 'import', 'from', 'if', 'else', 'elif', 'for', 'while', 'in', 'not', 'and', 'or',
            'function', 'const', 'let', 'export', 'async', 'await', 'try', 'except', 'with', 'as', 'yield', 'new'}


class Corpus:
    """Real text from this repository."""

    def __init__(self):
        listed = subprocess.run(['git', 'ls-files'], cwd=ROOT, capture_output=True, text=True).stdout.split()
        self.paths = [p for p in listed if not p.startswith(('site/vendor', 'assets/'))] or ['kelus/model.py']
        self.sources = [p for p in self.paths if p.endswith(('.py', '.mjs', '.js')) and (ROOT / p).stat().st_size < 200_000]
        self.logs = []
        for log in sorted((ROOT / 'training-runs').glob('*.log'))[:6]:
            self.logs += [line[:150] for line in log.read_text(errors='replace').splitlines()[:200] if line.strip()]
        self.logs = self.logs or ['{"step": 1, "train_loss": 3.58}']

    def code(self, rng, count):
        lines = (ROOT / rng.choice(self.sources)).read_text(errors='replace').splitlines() or ['pass']
        start = rng.randrange(max(1, len(lines) - count))
        return start + 1, lines[start:start + count]


def highlight(line, t):
    out = []
    for token in __import__('re').split(r'(\s+|[()\[\]{}:,.=])', line):
        text = html.escape(token)
        if token.lstrip().startswith(('#', '//')):
            out.append(f'<span style="color:{t["comment"]}">{html.escape("".join(token))}</span>')
        elif token in KEYWORDS:
            out.append(f'<span style="color:{t["keyword"]}">{text}</span>')
        elif token[:1] in '"\'`':
            out.append(f'<span style="color:{t["string"]}">{text}</span>')
        else:
            out.append(text)
    return ''.join(out)


def box(x, y, w, h, style='', inner=''):
    return f'<div style="position:absolute;left:{x}px;top:{y}px;width:{w}px;height:{h}px;overflow:hidden;box-sizing:border-box;{style}">{inner}</div>'


class Screen:
    def __init__(self, rng, corpus, size, family, theme):
        self.rng, self.corpus, (self.w, self.h), self.family = rng, corpus, size, family
        self.t = dict(zip(KEYS, theme))
        self.ui, self.mono = rng.choice(UI_FONTS), rng.choice(CODE_FONTS)
        self.regions, self.parts = [], []

    def add(self, label, x, y, w, h, style, inner=''):
        if w > 0 and h > 0:
            self.regions.append((CLASSES[label], x, y, w, h))
            self.parts.append(box(x, y, w, h, style, inner))

    # Region painters ------------------------------------------------------
    def titlebar(self, x, y, w, h):
        t = self.t
        lights = ''.join(f'<span style="display:inline-block;width:12px;height:12px;border-radius:6px;margin-right:8px;background:{c}"></span>'
                         for c in ('#ff5f57', '#febc2e', '#28c840'))
        name = Path(self.rng.choice(self.corpus.paths)).name
        self.add('toolbar', x, y, w, h, f'background:{t["title"]};border-bottom:1px solid {t["border"]};color:{t["muted"]};font:12px {self.ui};display:flex;align-items:center;padding-left:12px',
                 lights + f'<span style="flex:1;text-align:center;margin-right:80px">{html.escape(name)} — KelusModelB1</span>')

    def menubar(self, x, y, w, h):
        items = ''.join(f'<span style="margin-right:14px">{m}</span>' for m in 'File Edit Selection View Go Run Terminal Help'.split())
        self.add('toolbar', x, y, w, h, f'background:{self.t["title"]};color:{self.t["fg"]};font:12px {self.ui};padding:5px 10px', items)

    def tabs(self, x, y, w, h, names=None):
        t, rng = self.t, self.rng
        names = names or [Path(p).name for p in rng.sample(self.corpus.paths, min(5, len(self.corpus.paths)))][:rng.randint(1, 5)]
        active = rng.randrange(len(names))
        cells = ''.join(f'<span style="display:inline-block;height:{h}px;line-height:{h}px;padding:0 14px;border-right:1px solid {t["border"]};'
                        f'background:{t["editor"] if i == active else t["tabs"]};color:{t["fg"] if i == active else t["muted"]};'
                        f'{"border-top:2px solid " + t["accent"] if i == active else ""}">{html.escape(n)}</span>' for i, n in enumerate(names))
        self.add('toolbar', x, y, w, h, f'background:{t["tabs"]};border-bottom:1px solid {t["border"]};font:12px {self.ui};white-space:nowrap', cells)

    def status(self, x, y, w, h):
        items = ['⎇ main', f'Ln {self.rng.randint(1, 400)}, Col {self.rng.randint(1, 80)}', 'Spaces: 4', 'UTF-8', self.rng.choice(['Python 3.11', 'JavaScript', 'Markdown'])]
        inner = ''.join(f'<span style="margin-right:16px">{html.escape(i)}</span>' for i in items)
        color = '#ffffff' if self.t['status'] in ('#0078d4',) else self.t['fg']
        self.add('toolbar', x, y, w, h, f'background:{self.t["status"]};color:{color};font:11px {self.ui};padding:{max(0, h // 2 - 7)}px 10px;white-space:nowrap', inner)

    def activity(self, x, y, w, h):
        icons = ''.join(f'<div style="height:44px;line-height:44px;text-align:center;font-size:20px;color:{self.t["accent"] if i == 0 else self.t["muted"]}">{c}</div>'
                        for i, c in enumerate('⎘⌕⑂▷⊞⚙'))
        self.add('sidebar', x, y, w, h, f'background:{self.t["activity"]};border-right:1px solid {self.t["border"]}', icons)

    def explorer(self, x, y, w, h):
        rng, t = self.rng, self.t
        rows = [f'<div style="font-size:11px;letter-spacing:1px;padding:8px 12px;color:{t["muted"]}">{rng.choice(["EXPLORER", "FILES", "PROJECT", "OUTLINE"])}</div>']
        for path in sorted(rng.sample(self.corpus.paths, min(len(self.corpus.paths), h // 22))):
            depth = path.count('/')
            rows.append(f'<div style="padding:2px 0 2px {12 + depth * 12}px;white-space:nowrap">{"▸ " if rng.random() < .2 else ""}{html.escape(Path(path).name)}</div>')
        self.add('sidebar', x, y, w, h, f'background:{t["sidebar"]};color:{t["fg"]};font:13px {self.ui};border-right:1px solid {t["border"]}', ''.join(rows))

    def editor(self, x, y, w, h):
        size = self.rng.choice([12, 13, 14])
        first, lines = self.corpus.code(self.rng, h // (size + 6) + 2)
        body = ''.join(f'<div style="white-space:pre;height:{size + 6}px"><span style="color:{self.t["muted"]};display:inline-block;width:44px;text-align:right;margin-right:16px">{first + i}</span>{highlight(line, self.t)}</div>'
                       for i, line in enumerate(lines))
        self.add('editor', x, y, w, h, f'background:{self.t["editor"]};color:{self.t["fg"]};font:{size}px {self.mono};padding-top:6px', body)

    def terminal(self, x, y, w, h):
        rng, t = self.rng, self.t
        prompt = f'<span style="color:{t["accent"]}">hashtag@mac</span> <span style="color:{t["string"]}">KelusModelB1</span> % '
        commands = ['python3 -m unittest discover -s tests', 'git status', 'python3 -m kelus.progress', 'node --test', 'kani "/analyze"', 'ls -la']
        rows = []
        while len(rows) * 17 < h:
            rows.append(prompt + html.escape(rng.choice(commands)))
            rows += [html.escape(line) for line in rng.sample(self.corpus.logs, min(len(self.corpus.logs), rng.randint(1, 4)))]
        self.add('terminal', x, y, w, h, f'background:{t["panel"]};color:{t["fg"]};font:12px {self.mono};padding:6px 10px;line-height:17px',
                 ''.join(f'<div style="white-space:pre">{r}</div>' for r in rows))

    def dialog(self):
        rng, t = self.rng, self.t
        if rng.random() < 0.5:
            w, h = rng.randint(360, 520), rng.randint(150, 220)
            x, y = (self.w - w) // 2 + rng.randint(-40, 40), (self.h - h) // 2 + rng.randint(-60, 20)
            title, message = rng.choice([('Save changes?', 'Your changes will be lost if you do not save them.'),
                                         ('Delete file?', 'This action cannot be undone.'),
                                         ('Reload window?', 'An extension was updated.')])
            buttons = ''.join(f'<span style="display:inline-block;padding:6px 14px;margin-left:8px;border-radius:4px;background:{t["accent"] if i == 0 else t["border"]};color:#fff">{b}</span>'
                              for i, b in enumerate(rng.choice([['Save', 'Don\'t Save', 'Cancel'], ['Delete', 'Cancel'], ['Reload', 'Later']])))
            inner = f'<div style="font-weight:600;font-size:15px;margin-bottom:10px">{title}</div><div style="color:{t["muted"]}">{message}</div><div style="position:absolute;right:16px;bottom:16px">{buttons}</div>'
        else:  # a command palette
            w, h = rng.randint(420, 600), rng.randint(180, 300)
            x, y = (self.w - w) // 2, rng.randint(40, 90)
            items = ''.join(f'<div style="padding:5px 4px">{html.escape(Path(p).name)}</div>' for p in rng.sample(self.corpus.paths, min(len(self.corpus.paths), h // 28)))
            inner = f'<div style="border:1px solid {t["accent"]};padding:6px;margin-bottom:6px">&gt; </div>{items}'
        self.add('dialog', x, y, w, h, f'background:{t["dialog"]};color:{t["fg"]};font:13px {self.ui};padding:16px;border:1px solid {t["border"]};border-radius:8px;box-shadow:0 8px 30px rgba(0,0,0,.45)', inner)

    # Layout families --------------------------------------------------------
    def build(self):
        rng = self.rng
        margin = 0 if rng.random() < 0.6 else rng.randint(16, 90)
        x0, y0, w, h = margin, margin, self.w - 2 * margin, self.h - 2 * margin
        title = rng.randint(28, 36)
        self.titlebar(x0, y0, w, title)
        y, bottom = y0 + title, y0 + h
        if self.family != 'terminal_app' and rng.random() < 0.4:
            self.menubar(x0, y, w, 24)
            y += 24
        status = 0 if self.family == 'terminal_app' else rng.randint(20, 24)
        if status:
            self.status(x0, bottom - status, w, status)
        area_h = bottom - status - y
        getattr(self, self.family)(x0, y, w, area_h)
        if self.family != 'terminal_app' and rng.random() < 0.3:
            self.dialog()
        return self

    def ide(self, x, y, w, h):
        rng = self.rng
        right = rng.random() < 0.3
        bar, side = rng.randint(40, 52), rng.randint(180, 320)
        left_x = x + w - bar - side if right else x
        self.activity(x + w - bar if right else x, y, bar, h)
        self.explorer(x + w - bar - side if right else x + bar, y, side, h)
        mx = x if right else x + bar + side
        mw = w - bar - side
        tabs = rng.randint(30, 38)
        self.tabs(mx, y, mw, tabs)
        panel = int(h * rng.uniform(0.25, 0.45)) if rng.random() < 0.75 else 0
        self.editor(mx, y + tabs, mw, h - tabs - panel)
        if panel:
            self.tabs(mx, y + h - panel, mw, 28, names=['TERMINAL', 'PROBLEMS', 'OUTPUT'][:rng.randint(1, 3)])
            self.terminal(mx, y + h - panel + 28, mw, panel - 28)
        del left_x

    def editor_minimal(self, x, y, w, h):
        tabs = self.rng.randint(28, 36) if self.rng.random() < 0.7 else 0
        if tabs:
            self.tabs(x, y, w, tabs)
        self.editor(x, y + tabs, w, h - tabs)

    def terminal_app(self, x, y, w, h):
        rng = self.rng
        tabs = rng.randint(26, 32) if rng.random() < 0.6 else 0
        if tabs:
            self.tabs(x, y, w, tabs, names=[f'zsh ({i})' for i in range(1, rng.randint(2, 5))])
        side = rng.randint(160, 220) if rng.random() < 0.3 else 0
        if side:
            self.explorer(x, y + tabs, side, h - tabs)
        if rng.random() < 0.35:
            half = (w - side) // 2
            self.terminal(x + side, y + tabs, half - 1, h - tabs)
            self.terminal(x + side + half + 1, y + tabs, w - side - half - 1, h - tabs)
        else:
            self.terminal(x + side, y + tabs, w - side, h - tabs)

    def ide_split(self, x, y, w, h):
        """Validation only: sidebar on the right, editor and terminal side by side."""
        rng = self.rng
        side = rng.randint(180, 280)
        main = w - side
        editor_w = int(main * rng.uniform(0.5, 0.65))
        tabs = rng.randint(30, 36)
        self.tabs(x, y, editor_w, tabs)
        self.editor(x, y + tabs, editor_w, h - tabs)
        self.tabs(x + editor_w, y, main - editor_w, tabs, names=['TERMINAL'])
        self.terminal(x + editor_w, y + tabs, main - editor_w, h - tabs)
        self.explorer(x + main, y, side, h)

    def html(self):
        return (f'<div style="position:relative;width:{self.w}px;height:{self.h}px;overflow:hidden;'
                f'background:linear-gradient(135deg,{self.t["desktop"]},#1d2733)">' + ''.join(self.parts) + '</div>')

    def mask(self):
        image = Image.new('L', (self.w, self.h), CLASSES['background'])
        draw = ImageDraw.Draw(image)
        for label, x, y, w, h in self.regions:
            draw.rectangle([x, y, x + w - 1, y + h - 1], fill=label)
        return image


def render(screens, size, directory):
    """Stack screens vertically, take one Chrome screenshot, and crop them apart."""
    w, h = size
    page = '<!doctype html><meta charset="utf-8"><style>html,body{margin:0;padding:0}</style>' + ''.join(s.html() for s in screens)
    with tempfile.TemporaryDirectory() as temp:
        (Path(temp) / 'page.html').write_text(page, encoding='utf-8')
        shot = Path(temp) / 'shot.png'
        subprocess.run([CHROME, '--headless=new', '--disable-gpu', '--hide-scrollbars', '--force-device-scale-factor=1',
                        f'--window-size={w},{h * len(screens)}', f'--screenshot={shot}', f'file://{temp}/page.html'],
                       check=True, capture_output=True, timeout=120)
        with Image.open(shot) as sheet:
            return [sheet.crop((0, i * h, w, (i + 1) * h)).convert('RGB') for i in range(len(screens))]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=ROOT / 'data' / 'ui-render')
    parser.add_argument('--train', type=int, default=480)
    parser.add_argument('--validation', type=int, default=120)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    rng, corpus = random.Random(args.seed), Corpus()
    (args.out / 'images').mkdir(parents=True, exist_ok=True)
    (args.out / 'masks').mkdir(parents=True, exist_ok=True)
    plan = [('train', rng.choice(['ide', 'ide', 'editor_minimal', 'terminal_app'])) for _ in range(args.train)]
    plan += [('validation', 'ide_split') for _ in range(args.validation)]
    rows, counts = [], {}
    for size in SIZES:
        jobs = [(i, split, family) for i, (split, family) in enumerate(plan) if i % len(SIZES) == SIZES.index(size)]
        per_sheet = max(1, 12000 // size[1])
        for start in range(0, len(jobs), per_sheet):
            batch = jobs[start:start + per_sheet]
            screens = [Screen(random.Random(args.seed * 100_003 + i), corpus, size, family, rng.choice(THEMES)).build() for i, _, family in batch]
            for (i, split, family), screen, image in zip(batch, screens, render(screens, size, args.out)):
                name = f'{split}-{i:04d}'
                image.save(args.out / 'images' / f'{name}.png')
                screen.mask().save(args.out / 'masks' / f'{name}.png')
                rows.append({'image': f'images/{name}.png', 'mask': f'masks/{name}.png', 'split': split, 'family': family,
                             'theme': screen.t['name'], 'size': list(size)})
                counts[f'{split}/{family}'] = counts.get(f'{split}/{family}', 0) + 1
    rows.sort(key=lambda r: r['image'])
    (args.out / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    print(json.dumps({'out': str(args.out), 'images': len(rows), 'counts': counts}, indent=1))


if __name__ == '__main__':
    main()
