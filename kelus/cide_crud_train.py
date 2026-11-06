"""Focused Cide repair curriculum after the completed v2 run.

Trains varied SQLite contracts, then executes the unchanged v2 development
checks and a separate set of schema/identifier cases. Does not deploy a model.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from .corpus import digest
from .pretrain import write_json

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,default=6000);p.add_argument('--resume',action='store_true')
    p.add_argument('--evaluate',type=Path,help='Evaluate a checkpoint without training')
    p.add_argument('--out',type=Path,help='Evaluation report path')
    a=p.parse_args();data=ROOT/'data/cide-crud-v3';run=ROOT/'training-runs/cide-crud-v3'
    if a.evaluate:
        if not a.out:p.error('--evaluate needs --out')
        from .crud_tasks import evaluate
        report=evaluate(a.evaluate,data/'test-cases.json')
        a.out.parent.mkdir(parents=True,exist_ok=True);write_json(a.out,report)
        print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True);return
    if a.steps<1:p.error('Steps must be positive')
    run.mkdir(parents=True,exist_ok=True);lock=(run/'run.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:p.error('Cide CRUD training is already running')
    parent=ROOT/'training-runs/cide-saas-v2/cide-saas-v2-candidate.npz'
    out=run/'instruction.npz'
    recipe=dict(steps=a.steps,parent_sha256=digest(parent),data_sha256=digest(data/'instructions.jsonl'),
                cases_sha256=digest(data/'test-cases.json'),batch=4,lr=0.0001,seed=9135)
    if (run/'recipe.json').exists():
        if not a.resume:p.error('Run exists; use --resume')
        if json.loads((run/'recipe.json').read_text())!=recipe:p.error('Resume recipe changed')
    else:write_json(run/'recipe.json',recipe)
    child=None;stopping=False
    def stop(signum,frame):
        nonlocal stopping
        stopping=True
        if child and child.poll() is None:child.send_signal(signal.SIGINT)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    env={**os.environ,'VECLIB_MAXIMUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONUNBUFFERED':'1','PYTHONFAULTHANDLER':'1'}
    def state(value,**extra):write_json(run/'status.json',dict(model='Cide CRUD v3',state=value,pid=os.getpid(),total_steps=a.steps,**extra))
    def execute(command,log):
        nonlocal child
        if stopping:raise InterruptedError('Stopped')
        with log.open('a') as stream:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            result=child.wait()
        child=None
        if stopping:raise InterruptedError('Stopped; periodic checkpoints retained')
        if result:raise RuntimeError(f'Command failed ({result}); see {log}')
    from .model import Model
    try:
        completed=0
        if out.exists():
            model,header,_=Model.load(out);completed=header['metadata']['steps'];del model
        state('training',completed_steps=completed)
        if completed<a.steps:
            command=[sys.executable,'-m','kelus.train','--data',str(data/'instructions.jsonl'),'--out',str(out),
                     '--steps',str(a.steps-completed),'--batch','4','--lr','0.0001','--patience','0','--last-reply-only',
                     '--log-every','250','--seed','9135']
            command+=['--resume',str(out)] if out.exists() else ['--init-from',str(parent)]
            execute(command,run/'instruction.log')
        model,header,_=Model.load(out);del model
        if header['metadata']['steps']<a.steps:raise InterruptedError('Training stopped before its budget')
        model,header,_=Model.load(run/'instruction.best.npz')
        candidate=run/'cide-crud-v3-candidate.npz'
        model.save(candidate,{**header['metadata'],'specialist':'coding','display_name':'Cide CRUD v3 candidate',
                              'release_status':'requires capability review','completed_steps':a.steps,'recipe':recipe})
        del model
        state('evaluating',completed_steps=a.steps)
        execute([sys.executable,'-m','kelus.cide_saas_eval','--checkpoint',str(candidate),'--out',str(run/'development.json')],run/'evaluation.log')
        execute([sys.executable,'-m','kelus.cide_eval','--checkpoint',str(candidate),'--role','coding','--out',str(run/'basic-code.json')],run/'evaluation.log')
        execute([sys.executable,'-m','kelus.cide_crud_train','--evaluate',str(candidate),'--out',str(run/'heldout.json')],run/'evaluation.log')
        state('complete',completed_steps=a.steps,candidate=str(candidate))
    except InterruptedError as error:state('stopped',reason=str(error))
    except BaseException as error:state('failed',reason=str(error));raise
    finally:lock.close()


if __name__=='__main__':main()
