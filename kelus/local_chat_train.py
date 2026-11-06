"""Local paragraph/conversation experiment: Rust preparation, Metal updates, direct-model probes.

    .venv/bin/python -m kelus.local_chat_train
    .venv/bin/python -m kelus.local_chat_train --resume

By default, uses our newer raw-text base and varied human-written instruction data,
preserving the existing site and checkpoints. Accepts a separate parent, run, data,
step budget and probe file for a new curriculum. See TOPIC_TRAINING.md.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

from .corpus import digest
from .pretrain import write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'training-runs/kani-paragraph-mac-v1'
DEFAULT_PARENT = ROOT / 'training-runs/runpod-big-v2/kani-advanced.npz'
DEFAULT_DATA = ROOT / 'training-runs/kani-advanced-instruction-pilot/data/instructions.jsonl'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--parent', type=Path, default=DEFAULT_PARENT)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--steps', type=int, default=8000)
    parser.add_argument('--cases', type=Path)
    parser.add_argument('--last-reply-only', action='store_true')
    parser.add_argument('--probe-max-tokens', type=int, default=128)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('Steps must be positive')
    if not 1 <= args.probe_max_tokens <= 512:
        parser.error('Probe output must be between 1 and 512 tokens')
    RUN, PARENT, DATA = args.run.resolve(), args.parent.resolve(), args.data.resolve()
    if PARENT in (RUN / 'instruction.npz', RUN / 'instruction.best.npz'):
        parser.error('Parent must be preserved outside the new run')
    RUN.mkdir(parents=True, exist_ok=True)
    lock = (RUN / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    recipe = dict(parent_sha256=digest(PARENT), data_sha256=digest(DATA), total_steps=args.steps,
                  batch=2, lr=0.00005, memory_gb=9, seed=4271, tokenizer='Rust BPE',
                  update_backend='MLX/Metal', auto_deploy=False)
    if args.last_reply_only:
        recipe['last_reply_only'] = True
    if args.cases:
        recipe['probes_sha256'] = digest(args.cases)
    if args.probe_max_tokens != 128:
        recipe['probe_max_tokens'] = args.probe_max_tokens
    recipe_file = RUN / 'recipe.json'
    if recipe_file.exists():
        if not args.resume or json.loads(recipe_file.read_text()) != recipe:
            parser.error('Existing run: use --resume with the same recipe')
    elif args.resume:
        parser.error('No run to resume')
    else:
        write_json(recipe_file, recipe)
    output = RUN / 'instruction.npz'
    probe_python = ROOT / 'training-runs/instruction-tools/bin/python'
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1'}
    child, stopped = None, False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        if child and child.poll() is None:
            child.send_signal(signal.SIGINT)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def state(value, **extra):
        write_json(RUN / 'status.json', dict(state=value, pid=os.getpid(), total_steps=args.steps,
                   updated_at=time.time(), recipe=recipe, **extra))

    def execute(command, filename):
        nonlocal child
        if stopped:
            raise InterruptedError('Stopped; periodic checkpoints retained')
        with (RUN / filename).open('a') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            code = child.wait()
        child = None
        if stopped:
            raise InterruptedError('Stopped; periodic checkpoints retained')
        if code:
            raise RuntimeError(f'Child exited {code}; inspect {RUN / filename}')


    cases = RUN / 'probes.json'
    probes = json.loads((ROOT / 'evaluation/kani_instruction_probes.json').read_text())
    selected = {'read-location', 'context-correction', 'writing-unfamiliar', 'rewrite'}
    probes = [p for p in probes if p['id'] in selected]
    probes += [dict(id='cats-conversation', kind='manual', diagnostic_only=True,
                   messages=[dict(role='user', content='I like cats'), dict(role='assistant', content='What do you like about them?'),
                             dict(role='user', content='Their curiosity. Write a short paragraph about that.')],
                   rubric='Responds to curiosity about cats with a coherent relevant paragraph; no generic programming reply.')]
    if args.cases:
        probes = json.loads(args.cases.read_text())
    if args.cases:

        cases.write_bytes(args.cases.read_bytes())
    else:
        write_json(cases, probes)

    def probe(source, label):
        execute([str(probe_python), '-u', '-m', 'kaggle.instruction_probe', '--checkpoint', str(source),
                 '--cases', str(cases), '--out', str(RUN / f'{label}.json'), '--device', 'mps',
                 '--max-tokens', str(args.probe_max_tokens)], f'{label}.log')

    try:
        baseline_path = RUN / 'baseline.json'
        baseline_complete = False
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text())
            baseline_complete = (len(baseline.get('results', [])) == len(probes)
                                 and baseline.get('cases_sha256') == digest(cases)
                                 and baseline.get('checkpoint_sha256') == recipe['parent_sha256'])
        if not baseline_complete:
            state('baseline')
            probe(PARENT, 'baseline')
        completed = 0
        if output.exists():
            import numpy as np
            with np.load(output, allow_pickle=False) as archive:
                completed = json.loads(str(archive['header']))['metadata']['steps']
        if completed < args.steps:
            free = shutil.disk_usage(RUN).free
            if free < 6.5 * 2**30:
                raise OSError(f'Checkpoint safety requires 6.5 GiB free; available {free / 2**30:.2f} GiB')
            state('training', completed_steps=completed)
            command = [sys.executable, '-u', '-m', 'kelus.gpu_train', 'instruct', '--data', str(DATA),
                       '--out', str(output), '--steps', str(args.steps-completed), '--batch', '2', '--lr', '0.00005',
                       '--log-every', '250', '--validation-examples', '64', '--report-every', '25',
                       '--group-lengths', '--memory-gb', '9', '--seed', '4271',
                       '--rust-tokenizer', '--weights-only-best', '--checkpoint-compression-level', '1']
            if args.last_reply_only:
                command.append('--last-reply-only')
            command += ['--resume', str(output)] if output.exists() else ['--init-from', str(PARENT)]
            execute(command, 'instruction.log')
        state('evaluating', completed_steps=args.steps)
        probe(RUN / 'instruction.best.npz', 'best')
        probe(output, 'final')
        state('complete', completed_steps=args.steps, note='Review actual responses before any release')
    except InterruptedError as error:
        state('stopped', reason=str(error))
    except BaseException as error:
        state('failed', reason=str(error))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
