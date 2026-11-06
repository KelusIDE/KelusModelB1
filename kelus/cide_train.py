"""Train two separate descendants of our own Kani v7. Never replaces v7.

    python3 data/make_cide.py
    python3 -m kelus.cide_train

Uses the project's NumPy trainer, autograd and AdamW. Publishes local exports
only; production selection/deployment requires evaluating the new checkpoints.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', default='checkpoints/kani-b2-2f-v7-s10000.npz')
    parser.add_argument('--expert-steps', type=int, default=1500)
    parser.add_argument('--router-steps', type=int, default=1000)
    args = parser.parse_args()
    from .model import Model
    from .export_web import export
    for name, steps, role in [('cide-v1', args.expert_steps, 'coding'), ('kani-cide-v1', args.router_steps, 'router')]:
        output = ROOT / 'checkpoints' / (name + '.npz')
        subprocess.run([sys.executable, '-m', 'kelus.train', '--init-from', args.parent,
                        '--data', f'data/{name}.jsonl', '--out', str(output), '--steps', str(steps),
                        '--batch', '4', '--lr', '0.0001', '--patience', '0', '--last-reply-only',
                        '--log-every', '250', '--seed', '2041'], cwd=ROOT, check=True)
        selected = output.with_name(output.stem + '.best.npz')
        model, header, _ = Model.load(selected)

        release = ROOT / 'checkpoints' / (name + '-release.npz')
        if release.exists():
            raise FileExistsError(release)
        metadata = {**header['metadata'], 'selected_checkpoint': selected.name, 'specialist': role, 'display_name': 'Cide' if role == 'coding' else 'Kani + Cide',
                    'handoff_protocol': 'CIDE', 'training_backend': 'NumPy / Kelus autograd'}
        model.save(release, metadata)
        manifest = export(release, ROOT / 'training-runs/build-exports' / name)
        print(json.dumps({'release': str(release), 'role': role, 'sha256': manifest['weights_sha256']}), flush=True)


if __name__ == '__main__':
    main()
