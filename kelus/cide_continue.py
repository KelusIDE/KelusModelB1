"""Long local Cide continuation on CPU, leaving other GPU trainers alone.

    python3 -m kelus.cide_continue
    python3 -m kelus.progress training-runs/cide-continuation-v4/instruction.log --full

Uses the existing NumPy trainer; retains the parent and evaluates final/best
checkpoints without promoting them to an application.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .corpus import digest
from .model import Model
from .pretrain import write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--run', type=Path, default=ROOT / 'training-runs/cide-continuation-v4')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/cide-continuation-v4/instructions.jsonl')
    parser.add_argument('--init-from', type=Path, default=ROOT / 'training-runs/cide-crud-v3/instruction.npz')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('Steps must be positive')
    run, data, parent = args.run.resolve(), args.data.resolve(), args.init_from.resolve()
    if parent in (run / 'instruction.npz', run / 'instruction.best.npz'):
        parser.error('The parent checkpoint must be outside the new run')
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / 'run.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('This Cide continuation is already running')
    recipe = {'parent_sha256': digest(parent), 'data_sha256': digest(data),
              'steps': args.steps, 'batch': 2, 'lr': 0.00005, 'seed': 9244,
              'backend': 'CPU / NumPy and Kelus autograd', 'auto_deploy': False}
    if (run / 'recipe.json').exists():
        if not args.resume or json.loads((run / 'recipe.json').read_text()) != recipe:
            parser.error('Run exists; resume with its original recipe')
    else:
        if args.resume:
            parser.error('No continuation to resume')
        write_json(run / 'recipe.json', recipe)
    output = run / 'instruction.npz'
    child = None
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        if child and child.poll() is None:
            child.send_signal(signal.SIGINT)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    def state(value, **extra):
        write_json(run / 'status.json', {'model': run.name, 'state': value,
                   'pid': os.getpid(), 'total_steps': args.steps, 'device': 'cpu (numpy)',
                   'updated_at': time.time(), **extra})
    env = {**os.environ, 'VECLIB_MAXIMUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1'}
    def execute(command, log):
        nonlocal child
        if stopping:
            raise InterruptedError('Stopped; saved checkpoints retained')
        with log.open('a') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT)
            result = child.wait()
        child = None
        if stopping:
            raise InterruptedError('Stopped; saved checkpoints retained')
        if result:
            raise RuntimeError(f'Child exited {result}; inspect {log}')
    try:
        completed = 0
        if output.exists():
            model, header, states = Model.load(output)
            completed = header['metadata']['steps']
            del model, states
        state('training', completed_steps=completed)
        if completed < args.steps:
            command = [sys.executable, '-u', '-m', 'kelus.train', '--data', str(data),
                       '--out', str(output), '--steps', str(args.steps - completed),
                       '--batch', '2', '--lr', '0.00005', '--patience', '0',
                       '--last-reply-only', '--log-every', '250', '--seed', '9244']
            command += ['--resume', str(output)] if output.exists() else ['--init-from', str(parent)]
            execute(command, run / 'instruction.log')
        model, header, states = Model.load(output)
        completed = header['metadata']['steps']
        del model, states
        if completed < args.steps:
            raise InterruptedError('Training stopped before its budget')
        state('evaluating', completed_steps=completed)
        for label, checkpoint in [('final', output), ('best', run / 'instruction.best.npz')]:
            commands = [
                [sys.executable, '-m', 'kelus.cide_eval', '--checkpoint', str(checkpoint), '--role', 'coding', '--out', str(run / f'{label}-basic.json')],
                [sys.executable, '-m', 'kelus.cide_saas_eval', '--checkpoint', str(checkpoint), '--out', str(run / f'{label}-development.json')],
                [sys.executable, '-m', 'kelus.cide_crud_train', '--evaluate', str(checkpoint), '--out', str(run / f'{label}-reserved-crud.json')]]
            for command in commands:
                execute(command, run / 'evaluation.log')
        state('complete', completed_steps=completed, note='Capability reports require review; no automatic deployment')
    except InterruptedError as error:
        state('stopped', reason=str(error))
    except BaseException as error:
        state('failed', reason=str(error))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
