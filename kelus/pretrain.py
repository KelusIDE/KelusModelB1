"""From-random-weights next-token pretraining on the audited local corpus.

NumPy arithmetic only. Stable total-step schedule, resumable RNG/Adam state,
CPU microbatch accumulation, exclusive run lock, atomic checkpoints/status.
"""
import argparse
import fcntl
import json
import math
import os
import resource
import sys
from pathlib import Path
import signal
import time
import numpy as np
from .corpus import digest
from .model import Config, Model
from .optim import AdamW
from .tensor import cross_entropy, no_grad
from .tokenizer import from_dict


def write_json(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

class Corpus:
    def __init__(self,root):
        self.root=Path(root);path=self.root/'manifest.json'
        self.manifest=json.loads(path.read_text());self.fingerprint=digest(path)
        if self.manifest.get('format')!='kelus-raw-corpus-v1':raise ValueError('Unknown corpus format')
        if digest(self.root/'tokenizer.json')!=self.manifest['tokenizer_sha256']:raise ValueError('Tokenizer changed')
        self.tokenizer=from_dict(json.loads((self.root/'tokenizer.json').read_text()))
        self.streams={}
        for split in ('train','validation'):
            file=self.root/(split+'.bin')
            if digest(file)!=self.manifest['files'][split]:raise ValueError('Corpus tokens changed: '+split)
            stream=np.memmap(file,mode='r',dtype='<u2')
            if len(stream)!=self.manifest['tokens'][split] or int(stream.max())>=self.tokenizer.vocab_size:raise ValueError('Invalid token stream')
            self.streams[split]=stream
    def batch(self,split,batch,context,rng):
        stream=self.streams[split]
        if context>=len(stream):raise ValueError('Context exceeds token stream')
        starts=rng.integers(0,len(stream)-context,size=batch)
        blocks=np.stack([stream[s:s+context+1] for s in starts]).astype(np.int64)
        return blocks[:,:-1],blocks[:,1:]


def validation(model,corpus,batches=8,batch=1,seed=2401):
    # Fixed windows, fixed seed on every check; explicitly a sampled estimate.
    rng=np.random.default_rng(seed);total=0.0
    with no_grad():
        for _ in range(batches):
            x,y=corpus.batch('validation',batch,model.config.context,rng)
            total+=float(cross_entropy(model(x),y).data)
    return total/batches


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data/foundation'))
    p.add_argument('--run',type=Path,default=Path('training-runs/b1-foundation'))
    p.add_argument('--resume',action='store_true')
    p.add_argument('--steps',type=int,default=4000,help='TOTAL optimizer steps, including previous invocations')
    p.add_argument('--context',type=int,default=1024);p.add_argument('--width',type=int,default=320)
    p.add_argument('--layers',type=int,default=6);p.add_argument('--heads',type=int,default=8)
    p.add_argument('--batch',type=int,default=1);p.add_argument('--accumulate',type=int,default=4)
    p.add_argument('--lr',type=float,default=0.0005);p.add_argument('--seed',type=int,default=919)
    p.add_argument('--checkpoint-every',type=int,default=50);p.add_argument('--validation-batches',type=int,default=8)
    p.add_argument('--max-seconds',type=float,default=0,help='Stop cleanly at the next completed step after this time budget; 0 means total steps only')
    a=p.parse_args(argv)
    if not (a.steps>0 and 1<=a.batch<=16 and 1<=a.accumulate<=64 and a.lr>0 and math.isfinite(a.lr) and a.checkpoint_every>0 and a.validation_batches>0 and math.isfinite(a.max_seconds) and a.max_seconds>=0):p.error('Invalid training budget or optimizer settings')
    a.run.mkdir(parents=True,exist_ok=True)
    lock=(a.run/'run.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:p.error('This run is already active')
    latest=a.run/'latest.npz';best=a.run/'best.npz'
    if latest.exists() and not a.resume:p.error('Run already has a checkpoint. Use --resume or a separate --run directory.')
    if a.resume and not latest.exists():p.error('No checkpoint to resume')
    corpus=Corpus(a.data)
    recipe={'steps':a.steps,'batch':a.batch,'accumulate':a.accumulate,'lr':a.lr,'seed':a.seed,
            'validation_batches':a.validation_batches,'validation_seed':2401}
    metadata={};prior_elapsed=0.0
    if a.resume:
        model,header,states=Model.load(latest);metadata=header['metadata']
        if metadata.get('stage')!='pretraining' or metadata.get('corpus_sha256')!=corpus.fingerprint:p.error('Checkpoint corpus/stage mismatch')
        if metadata.get('recipe')!=recipe:p.error('Resume must use the original steps/batch/accumulation/LR/seed/validation recipe')
        if model.tokenizer.to_dict()!=corpus.tokenizer.to_dict():p.error('Checkpoint tokenizer mismatch')
        optimizer=AdamW(model.parameters);optimizer.restore(header['optimizer'],states)
        rng=np.random.default_rng();rng.bit_generator.state=metadata['rng_state']
        prior_elapsed=metadata.get('elapsed_seconds',0.0)
        best_loss=metadata['best_validation_loss']
    else:
        model=Model(Config(a.context,a.width,a.heads,a.layers,a.seed,corpus.tokenizer.vocab_size,'rope',True,0.0),corpus.tokenizer)
        optimizer=AdamW(model.parameters,learning_rate=a.lr,weight_decay=0.1)
        rng=np.random.default_rng(a.seed+1);best_loss=float('inf')
    if optimizer.step_count>=a.steps:
        print('Run already reached its total-step budget.',flush=True);return
    stop=False
    stop_file=a.run/'stop.request'
    stop_file.unlink(missing_ok=True)
    def request_stop(signum,frame):
        nonlocal stop
        stop=True
    old_handlers={s:signal.signal(s,request_stop) for s in (signal.SIGINT,signal.SIGTERM)}
    started=time.monotonic();first_step=optimizer.step_count;last_loss=None;last_val=metadata.get('validation_loss')
    history=a.run/'metrics.jsonl'
    def status(state,grad=None):
        elapsed=prior_elapsed+time.monotonic()-started
        completed=optimizer.step_count-first_step;rate=completed/max(time.monotonic()-started,1e-9)
        value={'stage':'pretraining','state':state,'pid':os.getpid(),'step':optimizer.step_count,'total_steps':a.steps,
               'parameters':model.parameter_count,'context':model.config.context,'tokens_per_step':a.batch*a.accumulate*model.config.context,
               'tokens_seen':optimizer.step_count*a.batch*a.accumulate*model.config.context,'train_loss':last_loss,'validation_loss':last_val,
               'elapsed_seconds':round(elapsed,2),'steps_per_second':round(rate,5),
               'eta_seconds':round((a.steps-optimizer.step_count)/rate) if rate else None,'updated_at':time.time(),
               'checkpoint':str(latest.resolve()),'data':str(a.data.resolve()),'recipe':recipe,
               'peak_memory_mib':round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024 if sys.platform=='darwin' else 1024),1)}
        if grad is not None:value['gradient_norm']=grad
        write_json(a.run/'status.json',value);return value
    def save(state):
        nonlocal best_loss,last_val,metadata
        last_val=validation(model,corpus,a.validation_batches,a.batch)
        improved=last_val<best_loss;best_loss=min(best_loss,last_val)
        metadata={'stage':'pretraining','initialization':'random weights; no pretrained model',
            'corpus_sha256':corpus.fingerprint,'steps':optimizer.step_count,'recipe':recipe,
            'rng_state':rng.bit_generator.state,'train_loss':last_loss,'validation_loss':last_val,'best_validation_loss':best_loss,
            'validation_method':f'Fixed {a.validation_batches*a.batch*model.config.context} held-out tokens, seed 2401; sampled loss, not coding ability',
            'elapsed_seconds':prior_elapsed+time.monotonic()-started,'tokens_seen':optimizer.step_count*a.batch*a.accumulate*model.config.context}
        model.save(latest,metadata,optimizer)
        if improved:model.save(best,metadata,optimizer)
        event=status(state)
        with history.open('a') as stream:stream.write(json.dumps(event)+'\n')
        print(json.dumps(event),flush=True)
    try:
        if not a.resume:save('running')
        else:status('running')
        while optimizer.step_count<a.steps and not stop and not stop_file.exists():
            if a.max_seconds and time.monotonic()-started>=a.max_seconds:break
            optimizer.zero_grad();loss_sum=0.0
            for _ in range(a.accumulate):
                x,y=corpus.batch('train',a.batch,model.config.context,rng)
                loss=cross_entropy(model(x,training=True,rng=rng),y)
                value=float(loss.data)
                if not math.isfinite(value):raise FloatingPointError('Nonfinite loss; previous atomic checkpoint retained')
                (loss/a.accumulate).backward();loss_sum+=value
            step=optimizer.step_count+1;warmup=min(100,max(1,a.steps//20))
            progress=max(0,step-warmup)/max(1,a.steps-warmup)
            optimizer.learning_rate=a.lr*min(1,step/warmup)*(0.1+0.9*0.5*(1+math.cos(math.pi*progress)))
            grad=optimizer.step();last_loss=loss_sum/a.accumulate
            if optimizer.step_count%a.checkpoint_every==0:save('running')
            elif optimizer.step_count%5==0:status('running',grad)
        state='complete' if optimizer.step_count>=a.steps else 'stopped' if stop or stop_file.exists() else 'time_budget'
        save(state)
    except BaseException:
        status('failed');raise
    finally:
        for s,handler in old_handlers.items():signal.signal(s,handler)
        lock.close()

if __name__=='__main__':main()
