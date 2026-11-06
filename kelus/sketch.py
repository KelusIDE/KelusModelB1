"""Kani Sketch: draw any of Quick, Draw!'s 345 everyday subjects as pen strokes, learned from scratch.

Data: Google Creative Lab's Quick, Draw! dataset (CC BY 4.0, https://quickdraw.withgoogle.com/data),
sketch-rnn files data/quickdraw/<subject>.npz. Each drawing becomes tokens: BOS, the subject's token,
one token per pen position on a 64x64 grid, LIFT where the pen leaves the paper, then EOS. Our own
transformer (kelus.model) learns that stream with kelus.pretrain; sampling allows only pen positions,
LIFT and EOS after the subject token.

The .npz files store each drawing as a pickled array. They are read with an unpickler that rebuilds
plain NumPy arrays and refuses anything else, so a file cannot run code.

    python3 -m kelus.sketch prepare                       # writes the data/sketch-v1 corpus
    python3 -m kelus.pretrain --data data/sketch-v1 --run training-runs/kani-sketch-v1 --context 256 --width 256 --layers 6 --heads 8
    python3 -m kelus.sketch draw --checkpoint training-runs/kani-sketch-v1/best.npz --prompt "draw a cat" --out cat.png
"""
import argparse
import json
from pathlib import Path
import pickle
import random
import re
import zipfile

import numpy as np

from .tokenizer import BOS, EOS, VOCAB_SIZE

ROOT = Path(__file__).resolve().parent.parent
QUICKDRAW = ROOT / 'data/quickdraw'
GRID = 64
MAX_TOKENS = 250


class SketchTokenizer:
    """Token IDs below VOCAB_SIZE keep the shared specials (BOS, EOS); then LIFT, subjects, pen positions."""

    def __init__(self, categories, grid=GRID):
        self.categories, self.grid = list(categories), int(grid)
        self.LIFT = VOCAB_SIZE
        self.CATEGORY = VOCAB_SIZE + 1
        self.POINT = self.CATEGORY + len(self.categories)
        self.vocab_size = self.POINT + self.grid * self.grid

    def to_dict(self):
        return {'kind': 'sketch', 'grid': self.grid, 'categories': self.categories}

    @classmethod
    def from_dict(cls, value):
        return cls(value['categories'], value.get('grid', GRID))

    def encode(self, category, strokes):
        """Token IDs for one drawing: strokes is [[(x, y), ...], ...] on the grid."""
        tokens = [BOS, self.CATEGORY + self.categories.index(category)]
        for stroke in strokes:
            last = None
            for x, y in stroke:
                token = self.POINT + int(y) * self.grid + int(x)
                if token != last:
                    tokens.append(token)
                    last = token
            tokens.append(self.LIFT)
        return tokens + [EOS]

    def decode(self, tokens):
        """Strokes from generated tokens (pen positions and LIFTs; anything else is ignored)."""
        strokes, current = [], []
        for token in tokens:
            if token == self.LIFT:
                if current:
                    strokes.append(current)
                current = []
            elif self.POINT <= token < self.vocab_size:
                current.append(divmod(token - self.POINT, self.grid)[::-1])
        return strokes + ([current] if current else [])


class _PlainArrays(pickle.Unpickler):
    """Rebuilds plain NumPy arrays only; any other class in the pickle is refused."""
    ALLOWED = {('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct'), ('numpy', 'ndarray'), ('numpy', 'dtype')}

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f'Refusing {module}.{name} in a drawing file')
        return super().find_class(module, name)


def load_npz(path):
    """{'train': [...], 'valid': [...], 'test': [...]} of int arrays shaped (points, 3): dx, dy, pen lifted after."""
    out = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, fortran, dtype = read(stream)
                if dtype.hasobject:
                    array = _PlainArrays(stream, encoding='latin1').load()
                else:
                    array = np.frombuffer(stream.read(), dtype=dtype).reshape(shape, order='F' if fortran else 'C')
            out[name.removesuffix('.npy')] = array
    return out


def grid_strokes(drawing, grid=GRID):
    """sketch-rnn stroke-3 offsets -> strokes of integer grid points, centered and scaled to fill the grid."""
    xy = np.cumsum(np.asarray(drawing[:, :2], dtype=np.float64), axis=0)
    low, extent = xy.min(0), xy.max(0) - xy.min(0)
    span = max(float(extent.max()), 1.0)
    scaled = (xy - low) / span * (grid - 1)
    scaled += ((grid - 1) - extent / span * (grid - 1)) / 2
    points = np.clip(np.round(scaled), 0, grid - 1).astype(int)
    strokes, current = [], []
    for (x, y), lifted in zip(points, drawing[:, 2]):
        current.append((int(x), int(y)))
        if lifted:
            strokes.append(current)
            current = []
    return strokes + ([current] if current else [])


def categories():
    names = [line.strip() for line in (QUICKDRAW / 'all-categories.txt').read_text().splitlines() if line.strip()]
    return [name for name in names if (QUICKDRAW / f'{name}.npz').exists()]


def prepare(out, train=1200, validation=20, seed=2036):
    """Write a kelus.pretrain corpus (kelus-raw-corpus-v1) of tokenized drawings."""
    from .corpus import digest
    names = categories()
    tokenizer = SketchTokenizer(names)
    rng = random.Random(seed)
    documents = {'train': [], 'validation': []}
    for name in names:
        data = load_npz(QUICKDRAW / f'{name}.npz')
        for split, source, count in (('train', 'train', train), ('validation', 'valid', validation)):
            kept = 0
            for drawing in data[source]:
                tokens = tokenizer.encode(name, grid_strokes(drawing))
                if len(tokens) <= MAX_TOKENS:
                    documents[split].append(tokens)
                    kept += 1
                    if kept == count:
                        break
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'tokenizer.json').write_text(json.dumps(tokenizer.to_dict()) + '\n')
    totals = {}
    for split, docs in documents.items():
        rng.shuffle(docs)
        stream = np.asarray([t for doc in docs for t in doc], dtype='<u2')
        stream.tofile(out / f'{split}.bin')
        totals[split] = int(stream.size)
    sources = {'dataset': 'Quick, Draw!', 'by': 'Google Creative Lab', 'license': 'CC BY 4.0', 'url': 'https://quickdraw.withgoogle.com/data',
               'files': 'sketch-rnn .npz per subject (storage.googleapis.com/quickdraw_dataset/sketchrnn)',
               'selection': f'first {train} train and {validation} validation drawings per subject of at most {MAX_TOKENS} tokens'}
    (out / 'sources.json').write_text(json.dumps(sources, indent=2) + '\n')
    manifest = {'format': 'kelus-raw-corpus-v1', 'dtype': '<u2', 'vocabulary': tokenizer.vocab_size, 'tokenizer_sha256': digest(out / 'tokenizer.json'),
                'tokens': totals, 'documents': {s: len(d) for s, d in documents.items()}, 'subjects': len(names),
                'sources_sha256': digest(out / 'sources.json'), 'files': {s: digest(out / f'{s}.bin') for s in documents}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def find_subject(prompt, names):
    """The subject named in a request ('draw two cats' -> 'cat'), longest name first, or None."""
    text = ' '.join(re.findall(r'[a-z0-9]+', prompt.lower()))
    for name in sorted(names, key=len, reverse=True):
        base = ' '.join(re.findall(r'[a-z0-9]+', name.lower()))
        forms = {base, base + 's', base + 'es'} | ({base[:-1] + 'ies'} if base.endswith('y') else set())
        if any(re.search(rf'\b{re.escape(form)}\b', text) for form in forms):
            return name
    return None


def sample(model, category, temperature=0.8, seed=0, top_k=4):
    """Generated tokens for one drawing of `category`, sampled from the `top_k` likeliest next tokens."""
    tokenizer = model.tokenizer
    rng = np.random.default_rng(seed)
    allowed = np.zeros(tokenizer.vocab_size, dtype=bool)
    allowed[tokenizer.POINT:] = True
    allowed[[tokenizer.LIFT, EOS]] = True
    tokens, cache = [BOS, tokenizer.CATEGORY + tokenizer.categories.index(category)], {}
    while len(tokens) < MAX_TOKENS:
        logits = np.asarray(model.next_logits(tokens, cache), dtype=np.float64)
        logits[~allowed] = -np.inf
        stroke = 0
        for token in reversed(tokens[2:]):
            if token == tokenizer.LIFT:
                break
            stroke += 1
        if stroke < 2:
            logits[[tokenizer.LIFT, EOS]] = -np.inf
        top = np.argpartition(-logits, top_k)[:top_k]
        weights = np.exp((logits[top] - logits[top].max()) / max(temperature, 1e-3))
        token = int(rng.choice(top, p=weights / weights.sum()))
        if token == EOS:
            break
        tokens.append(token)
    return tokens


def tidy(strokes):
    """Drop stray dots and specks (strokes under 3 points or 4 grid units across); people rarely draw them."""
    span = lambda s: max(x for x, _ in s) - min(x for x, _ in s) + max(y for _, y in s) - min(y for _, y in s)
    kept = [s for s in strokes if len(s) >= 3 and span(s) >= 4]
    return kept or strokes


def render(strokes, size=512, grid=GRID, width=7, ink='#1f2430', paper='#ffffff'):
    """A clean line drawing; drawn at twice the size and scaled down for smooth edges."""
    from PIL import Image, ImageDraw
    big, margin = size * 2, size * 2 * 0.08
    image = Image.new('RGB', (big, big), paper)
    draw = ImageDraw.Draw(image)
    place = lambda p: (margin + p[0] / (grid - 1) * (big - 2 * margin), margin + p[1] / (grid - 1) * (big - 2 * margin))
    for stroke in strokes:
        points = [place(p) for p in stroke]
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse([x - width, y - width, x + width, y + width], fill=ink)
        else:
            draw.line(points, fill=ink, width=width * 2, joint='curve')
            for x, y in (points[0], points[-1]):
                draw.ellipse([x - width, y - width, x + width, y + width], fill=ink)
    return image.resize((size, size), Image.LANCZOS)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    command = commands.add_parser('prepare', help='write the tokenized corpus')
    command.add_argument('--out', default='data/sketch-v1')
    command.add_argument('--train', type=int, default=1200)
    command = commands.add_parser('draw', help='draw the subject named in a prompt')
    command.add_argument('--checkpoint', required=True)
    command.add_argument('--prompt', required=True)
    command.add_argument('--out', required=True)
    command.add_argument('--seed', type=int, default=0)
    command.add_argument('--temperature', type=float, default=0.8)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps(prepare(args.out, args.train)))
        return
    from .model import Model
    model = Model.load(args.checkpoint)[0]
    subject = find_subject(args.prompt, model.tokenizer.categories)
    if subject is None:
        raise SystemExit(json.dumps({'error': 'unknown subject', 'subjects': len(model.tokenizer.categories)}))
    for attempt in range(5):
        strokes = tidy(model.tokenizer.decode(sample(model, subject, args.temperature, args.seed + attempt)))
        if sum(len(stroke) for stroke in strokes) >= 6:
            break
    render(strokes).save(args.out)
    print(json.dumps({'subject': subject, 'strokes': len(strokes), 'out': args.out}))


if __name__ == '__main__':
    main()
