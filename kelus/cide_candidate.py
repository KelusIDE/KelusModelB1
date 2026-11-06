"""Queue a bounded Cide 2.1 candidate behind active GPU training; never auto-release.

Run with .venv/bin/python -m kelus.cide_candidate. Results and process state live
in training-runs/cide-2-1. Interruptions keep checkpoints; --resume uses only the
remaining step budget and verifies the immutable recipe.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
from .corpus import digest
from .pretrain import write_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'training-runs/cide-2-1'
DATA = ROOT / 'data/cide-v8'
PARENT = ROOT / 'checkpoints/cide-2-release.npz'
ENV = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}


def gpu_jobs():
    lines = subprocess.check_output(['ps', '-axo', 'pid=,command='], text=True).splitlines()
    return [int(line.split()[0]) for line in lines if 'kelus.gpu_train' in line.split()[1:]]


def completed_steps(path):
    if not path.exists():
        return 0
    with np.load(path, allow_pickle=False) as archive:
        return int(json.loads(str(archive['header']))['metadata']['steps'])


def gate(reports, baseline):
    counts = lambda name, key: int(reports[name][key].split('/')[0])
    checks = {
        'new_wording': reports['wording']['passed'] >= max(10, baseline['passed']),
        'natural_development': reports['development']['passed'] >= 3,
        'basic_no_regression': reports['basic']['passed'] >= 8,
        'reserved_crud_no_regression': reports['crud']['passed'] >= 23,
        'website_structure_no_regression': counts('websites', 'working_structure') >= 5,
        'language_no_regression': counts('languages', 'right_language') >= 113,
    }
    return {'decision': 'eligible_for_review' if all(checks.values()) else 'withhold',
            'checks': checks, 'automatically_deployed': False,
            'scope': 'Narrow component and structural checks; not proof of general code or website correctness.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--evaluate-wording', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.evaluate_wording:
        if not args.out:
            parser.error('--evaluate-wording needs --out')
        from .crud_tasks import evaluate
        write_json(args.out, evaluate(args.evaluate_wording, DATA / 'test-cases.json'))
        return

    RUN.mkdir(exist_ok=True)
    with (RUN / 'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Cide candidate is already queued or running')
        recipe = {'target_steps': 1500, 'batch': 2, 'lr': 0.00002, 'seed': 485,
                  'parent_sha256': digest(PARENT), 'data_sha256': digest(DATA / 'instructions.jsonl'),
                  'cases_sha256': digest(DATA / 'test-cases.json')}
        manifest = json.loads((DATA / 'manifest.json').read_text())
        if recipe['parent_sha256'] != manifest['parent_sha256'] or recipe['data_sha256'] != manifest['data_sha256']:
            parser.error('Parent or dataset changed after verification; prepare a new candidate')
        if (RUN / 'recipe.json').exists():
            if not args.resume:
                parser.error('Run exists; use --resume to continue without resetting its budget')
            if json.loads((RUN / 'recipe.json').read_text()) != recipe:
                parser.error('Resume recipe changed')
        else:
            write_json(RUN / 'recipe.json', recipe)
        child = None

        def state(value, **extra):
            write_json(RUN / 'status.json', {'state': value, 'pid': os.getpid(), 'updated_at': time.time(), **extra})

        def stop(signum, frame):
            if child and child.poll() is None:
                child.terminate()
            raise InterruptedError('Stopped; last periodic checkpoint retained')

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)

        def execute(command, log):
            nonlocal child
            with (RUN / log).open('a') as stream:
                child = subprocess.Popen(command, cwd=ROOT, env=ENV, stdin=subprocess.DEVNULL,
                                         stdout=stream, stderr=subprocess.STDOUT)
                code = child.wait()
            child = None
            if code:
                raise RuntimeError(f'Command exited {code}; see {log}')

        def module(name, *arguments, log='evaluation.log'):
            execute([sys.executable, '-m', name, *map(str, arguments)], log)

        try:


            while jobs := gpu_jobs():
                state('queued', waiting_for_pids=jobs, target_steps=1500)
                time.sleep(30)
            if shutil.disk_usage(ROOT).free < 6 * 2**30:
                raise RuntimeError('Need 6 GiB free for checkpoint saves; no files deleted')
            out = RUN / 'instruction.npz'
            completed = completed_steps(out)
            if completed < recipe['target_steps']:
                state('training', completed_steps=completed, target_steps=1500)
                source = ['--resume', out] if out.exists() else ['--init-from', PARENT]
                module('kelus.gpu_train', 'instruct', *source, '--data', DATA / 'instructions.jsonl', '--out', out,
                       '--steps', 1500 - completed, '--batch', 2, '--lr', recipe['lr'], '--seed', recipe['seed'],
                       '--log-every', 500, '--report-every', 50, '--validation-examples', 84,
                       '--group-lengths', '--rust-tokenizer', '--weights-only-best',
                       '--checkpoint-compression-level', 1, '--memory-gb', 8, log='instruction.log')
            if completed_steps(out) != 1500:
                raise RuntimeError('Candidate did not reach its expected step budget')
            state('evaluating', completed_steps=1500)
            if not (RUN / 'baseline-wording.json').exists():
                module('kelus.cide_candidate', '--evaluate-wording', PARENT, '--out', RUN / 'baseline-wording.json')
            baseline = json.loads((RUN / 'baseline-wording.json').read_text())
            if baseline['sha256'] != recipe['parent_sha256'] or baseline['cases_sha256'] != recipe['cases_sha256']:
                raise RuntimeError('Baseline does not match this candidate parent and test cases')
            decisions = {}
            for label, checkpoint in [('final', out), ('best', RUN / 'instruction.best.npz')]:
                reports = {}
                suites = {
                    'wording': ('kelus.cide_candidate', ['--evaluate-wording', checkpoint]),
                    'basic': ('kelus.cide_eval', ['--checkpoint', checkpoint, '--role', 'coding']),
                    'development': ('kelus.cide_saas_eval', ['--checkpoint', checkpoint]),
                    'crud': ('kelus.cide_crud_train', ['--evaluate', checkpoint]),
                    'websites': ('kelus.website_eval', ['--checkpoint', checkpoint]),
                    'languages': ('kelus.code_language_eval', ['--checkpoint', checkpoint]),
                }
                for suite, (name, arguments) in suites.items():
                    path = RUN / f'{label}-{suite}.json'
                    module(name, *arguments, '--out', path)
                    reports[suite] = json.loads(path.read_text())
                decisions[label] = gate(reports, baseline)
            write_json(RUN / 'review.json', decisions)
            state('complete', completed_steps=1500, review=str(RUN / 'review.json'), automatically_deployed=False)
        except InterruptedError as error:
            state('stopped', reason=str(error))
        except BaseException as error:
            state('failed', reason=str(error))
            raise
        finally:
            if child and child.poll() is None:
                child.terminate()
                child.wait(timeout=30)


if __name__ == '__main__':
    main()
