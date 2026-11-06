"""Experimental text-conditioned RGB flow model. NumPy + local autograd only.

Random initialization; no pretrained encoder, latent codec, or image weights.
Pillow is used exclusively for decoding, resizing and writing images.
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np
from .tensor import Tensor, no_grad
from .optim import AdamW


@dataclass(frozen=True)
class Config:
    size: int = 32
    width: int = 256
    buckets: int = 1024
    seed: int = 42

    def validate(self):
        if not (8 <= self.size <= 64 and 16 <= self.width <= 1024 and 32 <= self.buckets <= 8192):
            raise ValueError('Expected size 8..64, width 16..1024, buckets 32..8192')


def captions(texts, buckets):
    """Stable bag of hashed words; no borrowed tokenizer or language weights."""
    result = np.zeros((len(texts), buckets), np.float32)
    for row, text in enumerate(texts):
        words = re.findall(r"\w+", text.lower())
        for word in words:
            index = int.from_bytes(hashlib.sha256(word.encode()).digest()[:8], 'little') % buckets
            result[row, index] += 1
        result[row] /= max(1, len(words))
    return result


class PixelModel:
    def __init__(self, config=Config()):
        config.validate()
        self.config = config
        rng = np.random.default_rng(config.seed)
        pixels = config.size * config.size * 3
        shapes = {'input': (pixels, config.width), 'text': (config.buckets, config.width),
                  'time': (3, config.width), 'hidden': (config.width, config.width),
                  'output': (config.width, pixels), 'bias': (1, config.width),
                  'hidden_bias': (1, config.width), 'output_bias': (1, pixels)}
        self.parameters = {name: Tensor(np.zeros(shape, np.float32) if 'bias' in name else
            rng.normal(0, 1 / np.sqrt(shape[0]), shape).astype(np.float32), requires_grad=True)
            for name, shape in shapes.items()}

    def forward(self, pixels, times, text):
        p = self.parameters
        t = np.asarray(times, np.float32).reshape(-1, 1)
        clock = Tensor(np.concatenate([t, np.sin(np.pi * t), np.cos(np.pi * t)], axis=1))
        h = (Tensor(pixels) @ p['input'] + Tensor(text) @ p['text'] + clock @ p['time'] + p['bias']).tanh()
        h = (h @ p['hidden'] + p['hidden_bias']).tanh()
        return h @ p['output'] + p['output_bias']

    def loss(self, images, texts, rng):
        target = np.asarray(images, np.float32).reshape(len(images), -1)
        noise = rng.normal(size=target.shape).astype(np.float32)
        t = rng.uniform(0, 1, (len(images), 1)).astype(np.float32)

        estimate = self.forward((1 - t) * noise + t * target, t, captions(texts, self.config.buckets))
        error = estimate - Tensor(target - noise)
        return (error * error).mean()

    def sample(self, prompt, steps=50, seed=42):
        if not 1 <= steps <= 1000:
            raise ValueError('Sampling steps must be 1..1000')
        c = self.config
        x = np.random.default_rng(seed).normal(size=(1, c.size * c.size * 3)).astype(np.float32)
        text = captions([prompt], c.buckets)
        with no_grad():
            for step in range(steps):
                x += self.forward(x, [[step / steps]], text).data / steps
        return np.round((np.clip(x.reshape(c.size, c.size, 3), -1, 1) + 1) * 127.5).astype(np.uint8)

    def save(self, path, metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {'format': 'kani-pixels-v1', 'config': asdict(self.config), 'metadata': metadata}

        fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.npz')
        try:
            with os.fdopen(fd, 'wb') as out:
                np.savez(out, header=json.dumps(header), **{k: p.data for k, p in self.parameters.items()})
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            header = json.loads(str(archive['header']))
            if header['format'] != 'kani-pixels-v1':
                raise ValueError('Not a Kani pixel checkpoint')
            model = cls(Config(**header['config']))
            for name, parameter in model.parameters.items():
                value = archive[name]
                if value.shape != parameter.data.shape or not np.isfinite(value).all():
                    raise ValueError(f'Invalid parameter: {name}')
                parameter.data[:] = value
        return model, header['metadata']


def dataset(manifest, size):
    from PIL import Image, ImageOps
    path = Path(manifest)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    groups = {'train': [], 'validation': []}
    seen = set()
    for row in rows:
        if row.get('split') not in groups or not isinstance(row.get('caption'), str) or not row['caption'].strip():
            raise ValueError('Every row needs caption, image and split (train/validation)')
        image_path = path.parent / row['image']
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if digest in seen:
            raise ValueError('Duplicate image detected; remove duplicates before splitting')
        seen.add(digest)
        with Image.open(image_path) as image:
            rgb = ImageOps.fit(ImageOps.exif_transpose(image).convert('RGB'), (size, size), method=Image.Resampling.LANCZOS)
            pixels = np.asarray(rgb, np.float32) / 127.5 - 1
        groups[row['split']].append((pixels, row['caption']))
    if any(not group for group in groups.values()):
        raise ValueError('Provide nonempty train and validation splits')
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train')
    train.add_argument('--data', required=True)
    train.add_argument('--out', required=True)
    train.add_argument('--steps', type=int, default=1000)
    train.add_argument('--batch', type=int, default=8)
    train.add_argument('--size', type=int, default=32)
    train.add_argument('--width', type=int, default=256)
    train.add_argument('--lr', type=float, default=0.0003)
    sample = sub.add_parser('sample')
    sample.add_argument('--checkpoint', required=True)
    sample.add_argument('--prompt', required=True)
    sample.add_argument('--out', required=True)
    sample.add_argument('--steps', type=int, default=50)
    sample.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.command == 'sample':
        from PIL import Image
        model, _ = PixelModel.load(args.checkpoint)
        Image.fromarray(model.sample(args.prompt, args.steps, args.seed)).save(args.out)
        return
    if args.steps < 1 or not 1 <= args.batch <= 256 or not 0 < args.lr <= 0.1:
        parser.error('Invalid training steps, batch size or learning rate')
    if Path(args.out).exists():
        parser.error('Output exists; choose a new checkpoint path')
    model = PixelModel(Config(size=args.size, width=args.width))
    groups = dataset(args.data, args.size)
    optimizer = AdamW(model.parameters, learning_rate=args.lr)
    rng = np.random.default_rng(42)
    fingerprint = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    for step in range(1, args.steps + 1):
        chosen = [groups['train'][i] for i in rng.integers(len(groups['train']), size=args.batch)]
        optimizer.zero_grad()
        loss = model.loss(np.stack([r[0] for r in chosen]), [r[1] for r in chosen], rng)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            validation = groups['validation'][:32]
            with no_grad():
                val = model.loss(np.stack([r[0] for r in validation]), [r[1] for r in validation], np.random.default_rng(123))
            metadata = {'step': step, 'train_loss': float(loss.data), 'validation_loss': float(val.data),
                        'manifest_sha256': fingerprint, 'scope': 'Low-resolution pixel-flow research; no quality claim'}
            model.save(args.out, metadata)
            print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
