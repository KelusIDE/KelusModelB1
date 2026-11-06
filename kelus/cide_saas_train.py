"""Long, resumable Cide-only SaaS curriculum using the existing NumPy trainers.

30,000 raw web/code updates, then 20,000 assistant-target instruction updates.
Creates a candidate checkpoint; never changes Kani, the CLI build, or deployment.
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


def show_status(run, watch=False):
    from .progress import bar, timing
    def read(path):
        try: return json.loads(path.read_text())
        except (FileNotFoundError, ValueError): return {}
    try:
        while True:
            pipeline = read(run/'pipeline.json'); recipe = read(run/'recipe.json')
            code = read(run/'code/status.json'); instruction = {}
            log = run/'instruction.training.jsonl'
            if log.exists():
                for line in log.read_text().splitlines():
                    try: instruction = json.loads(line)
                    except ValueError: pass
            raw_step = int(code.get('step',0)); sft_step = int(instruction.get('step',0))
            total = recipe.get('code_steps',30000)+recipe.get('instruction_steps',20000)
            fraction = min(1, (raw_step+sft_step)/total)
            status = pipeline.get('state','not started'); phase = pipeline.get('phase','preparation')
            if status == 'running':
                try: os.kill(pipeline['pid'],0)
                except (OSError,KeyError): status = 'stopped (process absent)'
            text = f'Cide SaaS v2 [{bar(fraction)}] {fraction:.2%}  {raw_step+sft_step:,}/{total:,} steps · {phase} · {status}'
            if phase == 'code' and status == 'running' and code.get('eta_seconds'):
                text += '\nCode phase: '+timing(code['eta_seconds'])+'; instruction phase follows.'
            if watch and sys.stdout.isatty(): print('\033[2J\033[H',end='')
            print(text,flush=True)
            if not watch or status != 'running': return
            time.sleep(2)
    except KeyboardInterrupt: pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=ROOT/'training-runs/cide-saas-v2')
    p.add_argument('--data', type=Path, default=ROOT/'data/cide-saas-v2')
    p.add_argument('--parent', type=Path, default=ROOT/'checkpoints/cide-v1-release.npz')
    p.add_argument('--code-steps', type=int, default=30000)
    p.add_argument('--instruction-steps', type=int, default=20000)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--status', action='store_true', help='Print recorded progress without changing training')
    p.add_argument('--watch', action='store_true', help='Watch progress; Ctrl+C only closes the monitor')
    a = p.parse_args()
    if a.status or a.watch: return show_status(a.run,a.watch)
    if min(a.code_steps, a.instruction_steps) < 1: p.error('Training steps must be positive')
    a.run.mkdir(parents=True, exist_ok=True)
    lock = (a.run/'pipeline.lock').open('a')
    try: fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: p.error('This Cide pipeline is already running')
    recipe = dict(code_steps=a.code_steps, instruction_steps=a.instruction_steps,
                  parent=str(a.parent.resolve()), parent_sha256=digest(a.parent),
                  corpus_sha256=digest(a.data/'manifest.json'), instructions_sha256=digest(a.data/'instructions.jsonl'),
                  backend='NumPy / Kelus autograd', seed=2207)
    recipe_path = a.run/'recipe.json'
    if recipe_path.exists():
        if not a.resume: p.error('Run exists. Use --resume or a new --run.')
        if json.loads(recipe_path.read_text()) != recipe: p.error('Resume recipe/data/parent changed')
    else: write_json(recipe_path, recipe)
    state_path = a.run/'pipeline.json'; stopping = False; child = None
    def state(phase, status, **extra):
        value = dict(model='Cide SaaS v2', phase=phase, state=status, pid=os.getpid(),
                     total_steps=a.code_steps+a.instruction_steps, updated_at=time.time(), **extra)
        write_json(state_path, value); print(json.dumps(value), flush=True)
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        if child and child.poll() is None: child.send_signal(signal.SIGINT)
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    environment = {**os.environ, 'VECLIB_MAXIMUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1', 'OMP_NUM_THREADS':'1', 'PYTHONUNBUFFERED':'1'}
    def execute(command, log):
        nonlocal child
        if stopping: raise InterruptedError('Training stopped')
        with log.open('a') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            code = child.wait()
        child = None
        if stopping: raise InterruptedError('Training stopped; saved checkpoints retained')
        if code: raise RuntimeError(f'Trainer exited {code}; see {log}')
    raw = a.run/'code'; instruction = a.run/'instruction.npz'
    try:
        state('code', 'running')
        command = [sys.executable, '-m', 'kelus.pretrain', '--data', str(a.data), '--run', str(raw),
                   '--steps', str(a.code_steps), '--batch', '1', '--accumulate', '1', '--lr', '0.00015',
                   '--checkpoint-every', '100', '--validation-batches', '8', '--seed', '2207']
        command += ['--resume'] if (raw/'latest.npz').exists() else ['--init-from', str(a.parent)]
        execute(command, a.run/'code.log')
        raw_state = json.loads((raw/'status.json').read_text())
        if raw_state['step'] < a.code_steps: raise InterruptedError('Raw-code phase stopped before its budget')
        completed = 0
        if instruction.exists():
            _, header, _ = Model.load(instruction); completed = header['metadata']['steps']
        if completed < a.instruction_steps:
            state('instruction', 'running', completed_code_steps=raw_state['step'])
            command = [sys.executable, '-m', 'kelus.train', '--data', str(a.data/'instructions.jsonl'),
                       '--out', str(instruction), '--steps', str(a.instruction_steps-completed), '--batch', '2',
                       '--lr', '0.00008', '--patience', '0', '--last-reply-only', '--log-every', '250', '--seed', '2207']
            command += ['--resume', str(instruction)] if completed else ['--init-from', str(raw/'best.npz')]
            execute(command, a.run/'instruction.log')
        _, header, _ = Model.load(instruction)
        if header['metadata']['steps'] < a.instruction_steps: raise InterruptedError('Instruction phase stopped before its budget')
        model, header, _ = Model.load(a.run/'instruction.best.npz')
        candidate = a.run/'cide-saas-v2-candidate.npz'
        model.save(candidate, {**header['metadata'], 'specialist':'coding', 'display_name':'Cide SaaS v2 candidate',
                               'training_backend':recipe['backend'], 'release_status':'unevaluated candidate',
                               'pipeline_recipe':recipe, 'completed_code_steps':raw_state['step'],
                               'selection':'lowest instruction validation loss; not proof of app-building ability'})
        del model
        state('evaluation', 'running', candidate=str(candidate))
        execute([sys.executable, '-m', 'kelus.cide_eval', '--checkpoint', str(candidate), '--role', 'coding',
                 '--out', str(a.run/'code-eval.json')], a.run/'evaluation.log')
        execute([sys.executable, '-m', 'kelus.cide_saas_eval', '--checkpoint', str(candidate),
                 '--out', str(a.run/'saas-eval.json')], a.run/'evaluation.log')
        state('complete', 'complete', candidate=str(candidate), completed_steps=a.code_steps+a.instruction_steps,
              note='Training complete. Candidate requires review of capability results before release.')
    except InterruptedError as error:
        state('stopped', 'stopped', reason=str(error))
    except BaseException as error:
        state('failed', 'failed', reason=str(error)); raise
    finally: lock.close()


if __name__ == '__main__': main()
