"""Queue a pixel-model continuation after existing GPU trainers finish.

    python3 -m kelus.pixel_continue
    python3 -m kelus.progress training-runs/kani-pixels-v2-extended.log

Keeps the serving model untouched. This runner uses only the standard library;
MLX is loaded by the child after the GPU is free. Closing the monitor is safe.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
GPU_TRAINER = re.compile(r'\s-m\s+kelus\.(?:gpu_train|pixel_dit|sketch)\s+(?:train|pretrain|instruct|bench)\b')


def gpu_jobs(listing):
    jobs = []
    for line in listing.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2:
            continue
        pid, command = fields
        executable = command.split()[0]
        if 'python' in Path(executable).name.lower() and GPU_TRAINER.search(command):
            jobs.append({'pid': int(pid), 'command': command})
    return jobs


def write_json(path, payload):
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(payload, indent=2) + '\n')
    temp.replace(path)


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(source, run, data, steps, python):
    if source.resolve() == run.resolve():
        raise ValueError('Use a separate run directory to preserve the serving model')
    config = json.loads((source / 'config.json').read_text())
    manifest = json.loads((data / 'manifest.json').read_text())
    if manifest['subjects'] != config['subjects']:
        raise ValueError('Training subjects/order differ from the checkpoint')
    for path in (source / 'latest.npz', data / 'images.npy', data / 'labels.npy', python):
        if not path.is_file():
            raise ValueError(f'Missing required file: {path}')
    if steps <= config['step']:
        raise ValueError('Total target steps must exceed the source checkpoint step')
    if (source / 'status.json').exists():
        state = json.loads((source / 'status.json').read_text()).get('state')
        if state != 'complete':
            raise ValueError(f'Source run is {state}; only copy a completed checkpoint')
    command = [str(python), '-u', '-m', 'kelus.pixel_dit', 'train', '--resume',
               '--data', str(data), '--run', str(run), '--steps', str(steps),
               '--batch', '32', '--width', str(config['width']), '--depth', str(config['depth']),
               '--heads', str(config['heads']), '--lr', '0.0001', '--ema', '0.9995',
               '--checkpoint-every', '1000', '--preview-every', '2500',
               '--preview-seed', str(config['step']), '--memory-gb', '7']
    plan = {'source': str(source), 'source_step': config['step'], 'total_steps': steps,
            'additional_steps': steps - config['step'], 'command': command,
            'source_sha256': sha256(source / 'latest.npz')}
    plan_path = run / 'continuation.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError('Existing continuation plan differs; choose another run directory')
        current = json.loads((run / 'config.json').read_text())
        if not (run / 'latest.npz').is_file():
            raise ValueError('Continuation checkpoint is missing')
        if current['step'] >= steps:
            raise ValueError('Continuation already reached its target')
    else:
        if (run / 'latest.npz').exists() or (run / 'config.json').exists():
            raise ValueError('Destination contains an unrelated checkpoint')
        temp = run / 'seed.tmp.npz'
        shutil.copy2(source / 'latest.npz', temp)
        if sha256(temp) != plan['source_sha256']:
            temp.unlink()
            raise ValueError('Checkpoint copy verification failed')
        temp.replace(run / 'latest.npz')
        write_json(run / 'config.json', config)
        preview = source / f'preview-{config["step"]:06d}.png'
        if preview.exists():
            shutil.copy2(preview, run / f'baseline-{config["step"]:06d}.png')
        write_json(plan_path, plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'training-runs/kani-pixels-v2')
    parser.add_argument('--run', type=Path, default=ROOT / 'training-runs/kani-pixels-v2-extended')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/pixels-v1')
    parser.add_argument('--steps', type=int, default=80000, help='Total target, including the source checkpoint steps')
    args = parser.parse_args()
    run, source, data = args.run.resolve(), args.source.resolve(), args.data.resolve()
    run.mkdir(parents=True, exist_ok=True)
    with (run / 'runner.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit('A continuation runner already owns this run')

        with (ROOT / 'training-runs/pixel-gpu.lock').open('a') as gpu_lock:
            plan = prepare(source, run, data, args.steps, ROOT / '.venv/bin/python')
            log = run.with_suffix('.log')
            with log.open('a') as stream:
                stream.write(json.dumps({'event': 'queued', 'additional_steps': plan['additional_steps']}) + '\n')
            try:
                while True:
                    listing = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True, check=True).stdout
                    jobs = gpu_jobs(listing)
                    locked = False
                    try:
                        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        locked = True
                    except BlockingIOError:
                        pass
                    if not jobs and locked:
                        break
                    if locked:
                        fcntl.flock(gpu_lock, fcntl.LOCK_UN)
                    config = json.loads((run / 'config.json').read_text())
                    write_json(run / 'status.json', {
                        'stage': 'pretraining', 'state': 'queued (waiting for GPU)', 'pid': os.getpid(),
                        'step': config['step'], 'total_steps': args.steps, 'updated_at': time.time(),
                        'steps_per_second': 0, 'waiting_for': jobs,
                        'model': 'Kani Pixels v2 continuation', 'validation_loss': None})
                    print(f'Queued: waiting for {len(jobs)} GPU trainer(s) / continuation lock', flush=True)
                    time.sleep(30)
                print('GPU is free; resuming pixel training', flush=True)
                with run.with_suffix('.out').open('a') as output:
                    child = subprocess.Popen(plan['command'], cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
                    code = child.wait()
                status = json.loads((run / 'status.json').read_text())
                if code or status.get('state') != 'complete' or status.get('step', 0) < args.steps:
                    raise RuntimeError(f'Trainer exited {code}; inspect {run.with_suffix(".out")}')
                print(f'Completed {args.steps:,} total steps. Candidate awaits visual review: {run}', flush=True)
            except Exception as error:
                config = json.loads((run / 'config.json').read_text())
                write_json(run / 'status.json', {
                    'stage': 'pretraining', 'state': 'failed', 'pid': os.getpid(),
                    'step': config['step'], 'total_steps': args.steps, 'updated_at': time.time(), 'error': str(error)})
                raise


if __name__ == '__main__':
    main()
