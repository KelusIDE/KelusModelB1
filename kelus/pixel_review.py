"""Compare Kani pixel sampling with identical noise and an equal model-evaluation budget.

    .venv/bin/python -m kelus.pixel_review --run training-runs/kani-pixels-v2 \
        --out training-runs/pixel-quality-review/sampling

Defaults to CPU so the local GPU trainer can continue. Saves every native 64x64
image, labeled contact sheets, and provenance. No external model is used to score
the images. The report is a visual review aid, not a held-out capability benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import mlx.core as mx
import numpy as np
from PIL import Image, ImageDraw

from .pixel_dit import SIZE, load, positions, sample


def review(run, out, subjects, seeds, evaluations=60, guidance=(1.0, 2.0, 4.0), device='cpu'):
    if evaluations < 2 or evaluations % 2:
        raise ValueError('Use a positive even model-evaluation budget of at least 2')
    run, out = Path(run), Path(out)
    if out.exists():
        raise ValueError('Choose a new output folder; previous reviews are preserved')
    if not subjects or not seeds or not guidance:
        raise ValueError('Provide subjects, seeds and guidance settings')
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError('Seeds must be distinct nonnegative integers')
    if device not in ('cpu', 'gpu'):
        raise ValueError('Device must be cpu or gpu')
    mx.set_default_device(mx.cpu if device == 'cpu' else mx.gpu)
    config_path = run / 'config.json'
    before = config_path.read_bytes()
    with (run / 'latest.npz').open('rb') as stream:
        checkpoint_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    params, config = load(run)
    if any(subject not in config['subjects'] for subject in subjects):
        raise ValueError('Review subject is not in the model vocabulary')
    labels = [config['subjects'].index(subject) for subject in subjects]
    pos = positions(config['width'])
    settings = [(solver, evaluations // (2 if solver == 'heun' else 1), float(cfg))
                for solver in ('euler', 'heun') for cfg in guidance]
    out.mkdir(parents=True)
    native = out / 'images'
    native.mkdir()
    report = {'checkpoint': str(run.resolve()), 'checkpoint_sha256': checkpoint_hash,
              'checkpoint_step': config['step'], 'weights': 'EMA', 'device': device,
              'native_size': SIZE, 'subjects': subjects, 'seeds': seeds,
              'evaluations_per_image': evaluations, 'results': [],
              'note': 'Visual comparison only. Seeds are fixed, all outputs retained. No quality ranking or held-out score.'}


    sheets = {scale: Image.new('RGB', (180 + len(labels) * len(seeds) * SIZE * scale,
                                     44 + len(settings) * (SIZE * scale + 24)), '#171b20') for scale in (1, 3)}
    for scale, sheet in sheets.items():
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 6), f'Kani Pixels | step {config["step"]} | {SIZE}x{SIZE} native | EMA', fill='white')
        for seed_index, seed in enumerate(seeds):
            for label_index, subject in enumerate(subjects):
                x = 180 + (seed_index * len(labels) + label_index) * SIZE * scale
                draw.text((x + 2, 26), f'{subject} s{seed}', fill='white')
    for row, (solver, steps, cfg) in enumerate(settings):
        started = time.monotonic()
        result = {'solver': solver, 'steps': steps, 'guidance': cfg, 'files': []}
        for scale, sheet in sheets.items():
            draw = ImageDraw.Draw(sheet)
            y = 44 + row * (SIZE * scale + 24)
            draw.text((8, y + 5), f'{solver} {steps} steps | cfg {cfg:g}', fill='white')
            draw.text((8, y + 21), f'{evaluations} model evaluations', fill='#b8c1cb')
        for seed_index, seed in enumerate(seeds):
            output = np.array(sample(params, labels, config['heads'], pos, steps=steps,
                                     guidance=cfg, seed=seed, null=config['classes'], solver=solver))
            if not np.isfinite(output).all():
                raise FloatingPointError('Sampler produced nonfinite pixels')
            pixels = ((output + 1) * 127.5).round().astype(np.uint8)
            for label_index, subject in enumerate(subjects):
                filename = f'{solver}-{steps}-cfg{cfg:g}-seed{seed}-class{labels[label_index]}.png'
                picture = Image.fromarray(pixels[label_index])
                picture.save(native / filename)
                result['files'].append({'subject': subject, 'seed': seed, 'image': f'images/{filename}'})
                for scale, sheet in sheets.items():
                    x = 180 + (seed_index * len(labels) + label_index) * SIZE * scale
                    y = 44 + row * (SIZE * scale + 24)
                    sheet.paste(picture.resize((SIZE * scale, SIZE * scale), Image.Resampling.NEAREST), (x, y))
        result['seconds'] = round(time.monotonic() - started, 3)
        report['results'].append(result)
        print(json.dumps({key: value for key, value in result.items() if key != 'files'}), flush=True)


    with (run / 'latest.npz').open('rb') as stream:
        after_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    if config_path.read_bytes() != before or after_hash != checkpoint_hash:
        raise RuntimeError('Checkpoint changed during review; review a stable snapshot')
    for scale, sheet in sheets.items():
        sheet.save(out / ('comparison-native.png' if scale == 1 else 'comparison-3x.png'))
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, default=Path('training-runs/kani-pixels-v2'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--subjects', nargs='+', default=['Cat', 'Dog', 'Car', 'Apple'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[2042, 2043])
    parser.add_argument('--evaluations', type=int, default=60)
    parser.add_argument('--guidance', nargs='+', type=float, default=[1, 2, 4])
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')
    args = parser.parse_args()
    review(args.run, args.out, args.subjects, args.seeds, args.evaluations, args.guidance, args.device)


if __name__ == '__main__':
    main()
