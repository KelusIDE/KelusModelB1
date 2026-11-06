"""Kani Pixels v2: a class-conditional diffusion transformer that makes 64x64 photos, trained from random weights.

The same family as current image generators, at small scale: the image is cut into 4x4-pixel patches
(256 tokens), a transformer with adaptive layer norm (adaLN-Zero, from DiT) reads them together with the
subject and the noise level, and is trained with rectified flow: for x_t = (1 - t) * photo + t * noise it
predicts the velocity noise - photo. Sampling starts from noise and takes Euler steps back to t = 0, with
classifier-free guidance (the subject is dropped from 10% of training examples). An exponential moving
average of the weights is kept for sampling. MLX runs it on the Mac's GPU.

Data: data/pixels-v1 (Open Images photos, CC BY 2.0, from data/make_pixels.py).

    .venv/bin/python -m kelus.pixel_dit train --data data/pixels-v1 --run training-runs/kani-pixels-v2 --steps 60000
    .venv/bin/python -m kelus.pixel_dit sample --run training-runs/kani-pixels-v2 --prompt "a cat" --out cat.png
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import time

import mlx.core as mx
import numpy as np

PATCH = 4
SIZE = 64


def init(classes, width=384, depth=12, heads=6, seed=0):
    """Weights as a flat dict; the conditioning output layers start at zero (adaLN-Zero)."""
    rng = np.random.default_rng(seed)
    dense = lambda fan_in, fan_out: mx.array(rng.normal(0, (1 / fan_in) ** 0.5, (fan_in, fan_out)).astype(np.float32))
    zeros = lambda *shape: mx.zeros(shape)
    p = {'patch.w': dense(PATCH * PATCH * 3, width), 'patch.b': zeros(width),
         'time.w1': dense(256, width), 'time.b1': zeros(width), 'time.w2': dense(width, width), 'time.b2': zeros(width),
         'label': mx.array(rng.normal(0, 0.02, (classes + 1, width)).astype(np.float32)),
         'final.ada.w': zeros(width, 2 * width), 'final.ada.b': zeros(2 * width),
         'final.w': zeros(width, PATCH * PATCH * 3), 'final.b': zeros(PATCH * PATCH * 3)}
    for i in range(depth):
        b = f'block{i}.'
        p.update({b + 'qkv': dense(width, 3 * width), b + 'qkv.b': zeros(3 * width), b + 'proj': dense(width, width), b + 'proj.b': zeros(width),
                  b + 'up': dense(width, 4 * width), b + 'up.b': zeros(4 * width), b + 'down': dense(4 * width, width), b + 'down.b': zeros(width),
                  b + 'ada.w': zeros(width, 6 * width), b + 'ada.b': zeros(6 * width)})
    return p


def positions(width, grid=SIZE // PATCH):
    """Fixed 2-D sine/cosine patch positions."""
    quarter = width // 4
    freq = 1 / 10000 ** (np.arange(quarter) / quarter)
    ys, xs = np.meshgrid(np.arange(grid), np.arange(grid), indexing='ij')
    parts = [np.sin(ys.reshape(-1, 1) * freq), np.cos(ys.reshape(-1, 1) * freq), np.sin(xs.reshape(-1, 1) * freq), np.cos(xs.reshape(-1, 1) * freq)]
    return mx.array(np.concatenate(parts, axis=1).astype(np.float32))


def timestep(t, dims=256):
    half = dims // 2
    freq = mx.exp(-math.log(10000) * mx.arange(half) / half)
    angles = (t * 1000)[:, None] * freq[None]
    return mx.concatenate([mx.cos(angles), mx.sin(angles)], axis=-1)


def patchify(images):
    b = images.shape[0]
    g = SIZE // PATCH
    return images.reshape(b, g, PATCH, g, PATCH, 3).transpose(0, 1, 3, 2, 4, 5).reshape(b, g * g, PATCH * PATCH * 3)


def unpatchify(tokens):
    b = tokens.shape[0]
    g = SIZE // PATCH
    return tokens.reshape(b, g, g, PATCH, PATCH, 3).transpose(0, 1, 3, 2, 4, 5).reshape(b, SIZE, SIZE, 3)


def _ln(x):
    return mx.fast.layer_norm(x, None, None, 1e-6)


def forward(p, x_t, t, labels, heads, pos):
    """Predicted velocity for noisy images x_t [B, 64, 64, 3] at times t [B] with subject indices [B]."""
    gelu = lambda h: 0.5 * h * (1 + mx.tanh(math.sqrt(2 / math.pi) * (h + 0.044715 * h ** 3)))
    x = patchify(x_t) @ p['patch.w'] + p['patch.b'] + pos
    c = gelu(timestep(t) @ p['time.w1'] + p['time.b1']) @ p['time.w2'] + p['time.b2'] + p['label'][labels]
    c = c * mx.sigmoid(c)
    b, n, d = x.shape
    depth = sum(1 for k in p if k.endswith('.qkv'))
    for i in range(depth):
        k = f'block{i}.'
        shift1, scale1, gate1, shift2, scale2, gate2 = mx.split((c @ p[k + 'ada.w'] + p[k + 'ada.b'])[:, None], 6, axis=-1)
        h = _ln(x) * (1 + scale1) + shift1
        q, kk, v = mx.split(h @ p[k + 'qkv'] + p[k + 'qkv.b'], 3, axis=-1)
        q, kk, v = (z.reshape(b, n, heads, d // heads).transpose(0, 2, 1, 3) for z in (q, kk, v))
        mixed = mx.fast.scaled_dot_product_attention(q, kk, v, scale=(d // heads) ** -0.5).transpose(0, 2, 1, 3).reshape(b, n, d)
        x = x + gate1 * (mixed @ p[k + 'proj'] + p[k + 'proj.b'])
        h = _ln(x) * (1 + scale2) + shift2
        x = x + gate2 * (gelu(h @ p[k + 'up'] + p[k + 'up.b']) @ p[k + 'down'] + p[k + 'down.b'])
    shift, scale = mx.split((c @ p['final.ada.w'] + p['final.ada.b'])[:, None], 2, axis=-1)
    return unpatchify((_ln(x) * (1 + scale) + shift) @ p['final.w'] + p['final.b'])


def flow_loss(p, images, labels, t, noise, heads, pos):
    x_t = (1 - t)[:, None, None, None] * images + t[:, None, None, None] * noise
    return mx.mean((forward(p, x_t, t, labels, heads, pos) - (noise - images)) ** 2)


def integrate(velocity, x, steps, solver='euler'):
    """Integrate the learned flow from t=1 to t=0; Heun uses two evaluations/step."""
    if not isinstance(steps, int) or steps <= 0:
        raise ValueError('Sampling steps must be a positive integer')
    if solver not in ('euler', 'heun'):
        raise ValueError('Unknown solver; choose euler or heun')
    dt = 1 / steps
    for i in range(steps, 0, -1):
        v = velocity(x, i / steps)
        proposal = x - dt * v
        if solver == 'heun':
            corrected = velocity(proposal, (i - 1) / steps)
            x = x - (dt / 2) * (v + corrected)
        else:
            x = proposal
        mx.eval(x)
    return x


def sample(p, labels, heads, pos, steps=50, guidance=4.0, seed=0, null=None, solver='euler'):
    """Images in [-1, 1] from noise; Euler preserves the original sampling behavior."""
    if not math.isfinite(guidance) or guidance < 0:
        raise ValueError('Guidance must be finite and nonnegative')
    count = len(labels)
    if not count:
        raise ValueError('At least one subject is required')
    classes = p['label'].shape[0] - 1
    if null is None:
        null = classes
    if null != classes or any(not isinstance(label, (int, np.integer)) or not 0 <= label < classes for label in labels):
        raise ValueError('Subject index or null class does not match the model')
    key = mx.random.key(seed)
    x = mx.random.normal((count, SIZE, SIZE, 3), key=key)
    cond = mx.array(labels)
    uncond = mx.full((count,), null)
    def velocity(x, time):
        t = mx.full((count,), time)
        both = forward(p, mx.concatenate([x, x]), mx.concatenate([t, t]), mx.concatenate([cond, uncond]), heads, pos)
        v_cond, v_uncond = both[:count], both[count:]
        return v_uncond + guidance * (v_cond - v_uncond)
    return mx.clip(integrate(velocity, x, steps, solver), -1, 1)


def find_subject(prompt, subjects):
    """The longest subject the prompt names as whole words ("two apples" names Apple, "a cartoon" names nothing), or None."""
    text = ' '.join(re.findall(r'[a-z]+', prompt.lower()))
    return next((s for s in sorted(subjects, key=len, reverse=True) if re.search(rf'\b{re.escape(s.lower())}(?:e?s)?\b', text)), None)


def to_png(images, path, columns=4, scale=2):
    from PIL import Image
    array = ((np.array(images) + 1) * 127.5).round().astype(np.uint8)
    rows = -(-len(array) // columns)
    sheet = Image.new('RGB', (columns * SIZE * scale, rows * SIZE * scale), 'white')
    for i, image in enumerate(array):
        sheet.paste(Image.fromarray(image).resize((SIZE * scale, SIZE * scale), Image.LANCZOS), ((i % columns) * SIZE * scale, (i // columns) * SIZE * scale))
    sheet.save(path)


def save(run, params, ema, opt, step, config, extra):
    weights = {f'p.{k}': np.array(v) for k, v in params.items()} | {f'ema.{k}': np.array(v) for k, v in ema.items()}
    weights |= {f'adam.m.{k}': np.array(v) for k, v in opt.m.items()} | {f'adam.v.{k}': np.array(v) for k, v in opt.v.items()}
    temp = run / 'latest.tmp.npz'
    np.savez(temp, **weights)
    temp.replace(run / 'latest.npz')
    (run / 'config.json').write_text(json.dumps({**config, 'step': step, 'adam_steps': opt.step_count, **extra}, indent=1) + '\n')


def load(run, ema=True):
    config = json.loads((run / 'config.json').read_text())
    with np.load(run / 'latest.npz') as archive:
        prefix = 'ema.' if ema else 'p.'
        params = {k[len(prefix):]: mx.array(archive[k]) for k in archive.files if k.startswith(prefix)}
    return params, config


def resume_config(run, config, resume, total_steps):
    """Fail before allocating GPU memory if resuming would change the model or labels."""
    checkpoint = run / 'latest.npz'
    metadata = run / 'config.json'
    if not resume:
        if checkpoint.exists() or metadata.exists():
            raise ValueError(f'{run} already contains a model; use --resume or a new run directory')
        return None
    if not checkpoint.is_file() or not metadata.is_file():
        raise ValueError('--resume requires both latest.npz and config.json')
    saved = json.loads(metadata.read_text())
    for key in ('width', 'depth', 'heads', 'classes', 'subjects', 'size', 'patch'):
        if saved.get(key) != config[key]:
            raise ValueError(f'Resume mismatch for {key}; use the original model configuration and subject order')
    if total_steps <= saved['step']:
        raise ValueError(f'--steps is the total target; it must exceed saved step {saved["step"]}')
    return saved


def train(args):
    from .gpu import AdamW
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    data = Path(args.data)
    images = np.load(data / 'images.npy', mmap_mode='r')
    labels = np.load(data / 'labels.npy').astype(np.int32)
    subjects = json.loads((data / 'manifest.json').read_text())['subjects']
    config = {'width': args.width, 'depth': args.depth, 'heads': args.heads, 'classes': len(subjects), 'subjects': subjects,
              'size': SIZE, 'patch': PATCH, 'data': str(data), 'objective': 'rectified flow, velocity = noise - image'}
    saved = resume_config(run, config, args.resume, args.steps)
    pos = positions(args.width)
    if saved is not None:
        with np.load(run / 'latest.npz') as archive:
            params = {k[2:]: mx.array(archive[k]) for k in archive.files if k.startswith('p.')}
            ema = {k[4:]: mx.array(archive[k]) for k in archive.files if k.startswith('ema.')}
            opt = AdamW(params, weight_decay=0.0)
            for name in params:
                opt.m[name], opt.v[name] = mx.array(archive[f'adam.m.{name}']), mx.array(archive[f'adam.v.{name}'])
        opt.step_count, step = saved['adam_steps'], saved['step']
    else:
        params = init(len(subjects), args.width, args.depth, args.heads, args.seed)
        ema = {k: mx.array(v) for k, v in params.items()}
        opt, step = AdamW(params, weight_decay=0.0), 0
    null = len(subjects)
    gradient = mx.value_and_grad(flow_loss)
    step_fn = mx.compile(lambda p, x, l, t, n: gradient(p, x, l, t, n, args.heads, pos))
    rng = np.random.default_rng(args.seed + step)
    if saved and saved.get('rng_state'):
        rng.bit_generator.state = saved['rng_state']
    started, first = time.monotonic(), step
    count = sum(int(np.prod(v.shape)) for v in params.values())
    def status(state, value=None):
        payload = {
        'stage': 'pretraining', 'state': state, 'pid': os.getpid(), 'step': step, 'total_steps': args.steps, 'parameters': count,
        'train_loss': value, 'validation_loss': None, 'steps_per_second': round((step - first) / max(time.monotonic() - started, 1e-9), 4),
        'updated_at': time.time(), 'device': 'gpu (mlx)', 'model': 'Kani Pixels v2 (diffusion transformer)'}
        temp = run / 'status.tmp.json'
        temp.write_text(json.dumps(payload, indent=1) + '\n')
        temp.replace(run / 'status.json')
    log = open(run.parent / f'{run.name}.log', 'a')
    status('running', saved.get('loss') if saved else None)
    while step < args.steps:
        index = rng.integers(0, len(images), args.batch)
        batch = images[np.sort(index)].astype(np.float32) / 127.5 - 1
        flip = rng.random(args.batch) < 0.5
        batch[flip] = batch[flip, :, ::-1]
        chosen = labels[np.sort(index)].copy()
        chosen[rng.random(args.batch) < 0.1] = null
        t = 1 / (1 + np.exp(-rng.normal(0, 1, args.batch)))
        noise = rng.normal(0, 1, batch.shape).astype(np.float32)
        value, grads = step_fn(params, mx.array(batch), mx.array(chosen), mx.array(t.astype(np.float32)), mx.array(noise))
        warm = min(1.0, (step + 1) / 1000)
        params, _ = opt.update(params, grads, args.lr * warm)
        ema = {k: args.ema * ema[k] + (1 - args.ema) * params[k] for k in params}
        mx.eval(params, ema, opt.m, opt.v, value)
        step += 1
        if step % 50 == 0:
            status('running', float(value))
        if step % args.checkpoint_every == 0 or step == args.steps:
            save(run, params, ema, opt, step, config, {'loss': float(value), 'rng_state': rng.bit_generator.state})
            report = {'step': step, 'train_loss': round(float(value), 4), 'elapsed_seconds': round(time.monotonic() - started, 1)}
            log.write(json.dumps(report) + '\n'); log.flush()
            print(json.dumps(report), flush=True)
        if step % args.preview_every == 0:
            picks = [subjects.index(s) for s in ('Cat', 'Dog', 'Car', 'Apple') if s in subjects][:4] or [0, 1, 2, 3]
            preview_seed = args.preview_seed if args.preview_seed is not None else step
            to_png(sample(ema, picks * 2, args.heads, pos, steps=30, seed=preview_seed, null=null), run / f'preview-{step:06d}.png')
    status('complete', float(value))
    log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    command = commands.add_parser('train')
    command.add_argument('--data', default='data/pixels-v1')
    command.add_argument('--run', default='training-runs/kani-pixels-v2')
    command.add_argument('--resume', action='store_true')
    command.add_argument('--preview-seed', type=int, help='Keep preview noise fixed for comparisons across checkpoints')
    for name, default in (('steps', 60000), ('batch', 64), ('width', 384), ('depth', 12), ('heads', 6), ('seed', 2039),
                          ('checkpoint_every', 1000), ('preview_every', 5000)):
        command.add_argument('--' + name.replace('_', '-'), type=int, default=default)
    command.add_argument('--lr', type=float, default=0.0002)
    command.add_argument('--ema', type=float, default=0.9995)
    command.add_argument('--memory-gb', type=float, default=12, help='GPU memory cap for this process')
    command = commands.add_parser('sample')
    command.add_argument('--run', default='training-runs/kani-pixels-v2')
    command.add_argument('--prompt', required=True)
    command.add_argument('--out', required=True)
    command.add_argument('--count', type=int, default=4)
    command.add_argument('--seed', type=int, default=0)
    command.add_argument('--guidance', type=float, default=4.0)
    command.add_argument('--steps', type=int, default=50, help='Integration steps; Heun evaluates the model twice per step')
    command.add_argument('--solver', choices=('euler', 'heun'), default='euler')
    command.add_argument('--device', choices=('cpu', 'gpu'), default='gpu')
    command.add_argument('--columns', type=int, default=4)
    command.add_argument('--scale', type=int, default=2, help='each 64×64 image is enlarged this many times')
    command = commands.add_parser('bench')
    for name, default in (('batch', 64), ('width', 384), ('depth', 12), ('heads', 6)):
        command.add_argument('--' + name, type=int, default=default)
    args = parser.parse_args()
    if args.command == 'train':
        from .gpu import limit_memory
        limit_memory(args.memory_gb)
        train(args)
    elif args.command == 'sample':
        mx.set_default_device(mx.cpu if args.device == 'cpu' else mx.gpu)
        params, config = load(Path(args.run))
        subject = find_subject(args.prompt, config['subjects'])
        if subject is None:

            print(json.dumps({'subject': None, 'out': None}))
            return
        images = sample(params, [config['subjects'].index(subject)] * args.count, config['heads'], positions(config['width']),
                        steps=args.steps, guidance=args.guidance, seed=args.seed, null=config['classes'], solver=args.solver)
        to_png(images, args.out, args.columns, args.scale)
        print(json.dumps({'subject': subject, 'out': args.out}))
    else:
        params = init(100, args.width, args.depth, args.heads)
        pos = positions(args.width)
        gradient = mx.value_and_grad(flow_loss)
        fn = mx.compile(lambda p, x, l, t, n: gradient(p, x, l, t, n, args.heads, pos))
        from .gpu import AdamW
        opt = AdamW(params, weight_decay=0.0)
        x = mx.random.normal((args.batch, SIZE, SIZE, 3))
        times = []
        for i in range(8):
            started = time.perf_counter()
            value, grads = fn(params, x, mx.zeros((args.batch,), dtype=mx.int32), mx.full((args.batch,), 0.5), x)
            params, _ = opt.update(params, grads, 1e-4)
            mx.eval(params, value)
            times.append(time.perf_counter() - started)
        count = sum(int(np.prod(v.shape)) for v in params.values())
        print(json.dumps({'parameters': count, 'batch': args.batch, 'seconds_per_step': round(float(np.median(times[3:])), 3)}))


if __name__ == '__main__':
    main()
