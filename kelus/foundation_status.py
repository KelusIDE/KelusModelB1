"""Read foundation training status, or request a stop at a complete optimizer step."""
import argparse
import json
from pathlib import Path
import time

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=Path('training-runs/b1-foundation'));p.add_argument('--watch',action='store_true');p.add_argument('--stop',action='store_true');a=p.parse_args()
    path=a.run/'status.json'
    if not path.exists():p.error('This foundation run has not started')
    if a.stop:
        (a.run/'stop.request').write_text('Stop after the current complete optimizer step.\n')
        print('Stop requested. The trainer will save its complete step and exit.');return
    try:
        while True:
            value=json.loads(path.read_text());eta=value.get('eta_seconds');stale=max(0,int(time.time()-value['updated_at']))
            print(f"B1 foundation — {value['state']} | step {value['step']:,}/{value['total_steps']:,} | tokens {value['tokens_seen']:,} | loss {value['train_loss']} | validation {value['validation_loss']} | ETA {eta/3600:.1f}h" if eta else f"B1 foundation — {value['state']} | step {value['step']:,}/{value['total_steps']:,}",flush=True)
            if stale>120:print(f'Last status update {stale}s ago; this is recorded status, not proof the process is alive.')
            if not a.watch or value['state']!='running':break
            time.sleep(5)
    except KeyboardInterrupt:print('Status viewer closed; training continues.')

if __name__=='__main__':main()
