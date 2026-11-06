"""Finish the active Big Kani run, then train/evaluate topic following sequentially.

One GPU job at a time, a 12-hour ceiling, one recovery attempt per stage,
atomic checkpoints, disk-space checks, and a saved report. Never deploys.
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
from .pretrain import write_json

ROOT = Path(__file__).resolve().parents[1]


def checkpoint_step(path):
    if not path.exists():
        return 0
    with np.load(path, allow_pickle=False) as archive:
        return int(json.loads(str(archive['header']))['metadata']['steps'])


def trainer_alive(pid):
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True)
    command = result.stdout.strip()
    return '-m kelus.gpu_train instruct' in command and 'kani-big-v2-chat.npz' in command


def outcome(folder):
    reports = {name: json.loads((folder / f'{name}.json').read_text()) for name in ['baseline','best','final']}
    base = reports['baseline']
    candidates = sorted(['best','final'], key=lambda name: reports[name]['automatic_passed'], reverse=True)
    selected = candidates[0]
    candidate = reports[selected]
    return dict(baseline=f"{base['automatic_passed']}/{base['automatic_total']}",
                best=f"{reports['best']['automatic_passed']}/{reports['best']['automatic_total']}",
                final=f"{reports['final']['automatic_passed']}/{reports['final']['automatic_total']}",
                automatic_gain=candidate['automatic_passed']-base['automatic_passed'],
                candidate=selected, candidate_path=str(folder / ('instruction.best.npz' if selected == 'best' else 'instruction.npz')),
                manual_review_required=True, deployed=False,
                limitation='Only nine narrow automatic checks; seven writing/grounding/relevance cases remain ungraded. No broad capability claim.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial-pid', type=int, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=12)
    args = parser.parse_args()
    if not 0 < args.hours <= 12:
        parser.error('Use a positive budget of at most 12 hours')
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = time.monotonic() + args.hours*3600
    stopped, child = False, None

    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        if child and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        elif trainer_alive(args.initial_pid):
            os.kill(args.initial_pid, signal.SIGINT)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def state(stage, **extra):
        write_json(run / 'status.json', dict(stage=stage,pid=os.getpid(),updated_at=time.time(),
                   initial_pid=args.initial_pid,deadline_hours=args.hours,**extra))

    def check():
        if stopped or (run/'STOP').exists() or time.monotonic() >= deadline:
            raise InterruptedError('User stop or overnight time limit; saved checkpoints retained')

    def disk():
        free = shutil.disk_usage(ROOT).free
        if free < 6.5*2**30:
            raise OSError(f'Checkpoint safety requires 6.5 GiB free; available {free/2**30:.2f} GiB')

    def execute(command, log):
        nonlocal child
        check()
        env = {**os.environ,'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'}
        with log.open('a') as stream:
            child = subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                     stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                try:
                    check()
                except InterruptedError:
                    try:
                        os.killpg(child.pid,signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(child.pid,signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        child.wait()
                    raise
                time.sleep(2)
            code = child.returncode
        child = None
        return code

    latest = ROOT / 'checkpoints/kani-big-v2-chat.npz'
    best = ROOT / 'checkpoints/kani-big-v2-chat.best.npz'
    try:
        state('waiting_for_big_kani', saved_step=checkpoint_step(latest), target_steps=9000)
        while trainer_alive(args.initial_pid):
            check()
            state('waiting_for_big_kani', saved_step=checkpoint_step(latest), target_steps=9000)
            time.sleep(5)
        step = checkpoint_step(latest)
        if step < 9000:

            disk()
            state('recovering_big_kani', saved_step=step,target_steps=9000)
            command=[sys.executable,'-u','-m','kelus.gpu_train','instruct','--resume',str(latest),
                     '--out',str(latest),'--data',str(ROOT/'data/instruct-v12.jsonl'),
                     '--steps',str(9000-step),'--batch','2','--lr','0.0001','--log-every','250',
                     '--validation-examples','600','--report-every','50','--group-lengths',
                     '--rust-tokenizer','--memory-gb','9','--checkpoint-compression-level','1','--weights-only-best']
            code=execute(command,ROOT/'training-runs/kani-big-v2-chat.log')
            if code or checkpoint_step(latest)<9000:
                raise RuntimeError(f'Big Kani recovery did not finish; exit {code}')
        disk()

        parent=run/'big-kani-parent.npz'
        if not parent.exists():
            result=subprocess.run(['/bin/cp','-c',str(best),str(parent)],capture_output=True)
            if result.returncode:
                shutil.copy2(best,parent)
        focused=run/'topic'
        focused.mkdir(exist_ok=True)
        command=[sys.executable,'-u','-m','kelus.local_chat_train','--run',str(focused),
                 '--parent',str(parent),'--data',str(ROOT/'training-runs/kani-topic-v1/data/instructions.jsonl'),
                 '--steps','1500','--cases',str(ROOT/'evaluation/topic_v1.json'),'--last-reply-only']
        state('topic_training_and_evaluation',big_kani_steps=checkpoint_step(latest),focused_run=str(focused))
        code=execute(command,run/'topic-supervisor.log')
        focused_state=json.loads((focused/'status.json').read_text()) if (focused/'status.json').exists() else {}
        if code or focused_state.get('state')!='complete':
            disk()
            if not (focused/'instruction.npz').exists():
                raise RuntimeError(f'Topic stage failed before a resumable checkpoint: exit {code}')
            state('recovering_topic',focused_run=str(focused))
            code=execute(command+['--resume'],run/'topic-supervisor.log')
            focused_state=json.loads((focused/'status.json').read_text())
            if code or focused_state.get('state')!='complete':
                raise RuntimeError(f'Topic recovery did not finish: exit {code}')
        result=outcome(focused)
        write_json(run/'results.json',result)
        lines=['# Overnight Kani results','',f"Big Kani completed {checkpoint_step(latest):,} updates.",
               f"Automatic probes: baseline {result['baseline']}; best {result['best']}; final {result['final']}.",
               '', 'The seven manual writing/grounding/relevance cases require review in topic/baseline.json, topic/best.json and topic/final.json.',
               'No website/CLI model was changed automatically. Narrow probe gains do not establish general intelligence.']
        (run/'RESULTS.md').write_text('\n'.join(lines)+'\n')
        state('complete',**result)
    except InterruptedError as error:
        if trainer_alive(args.initial_pid):
            os.kill(args.initial_pid,signal.SIGTERM)
        state('stopped',reason=str(error))
    except BaseException as error:
        state('failed',reason=str(error))
        raise
    finally:
        lock.close()


if __name__=='__main__':
    main()
