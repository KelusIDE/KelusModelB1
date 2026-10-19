"""Disclosed application regression replay, separate neural and helper results."""
import argparse
import hashlib
import json
from pathlib import Path
from .model import Model
from .runtime import prepare_reply
from .evaluate import cases, code_result


# Fresh probe wording: not in the curriculum or its validation set.
PROBES=[('Can you help me program something?',('Python','code','build')),
    ('nah that is not what I meant',('misunderstood','restate')),
    ('could we write a little code together?',('code','language','build')),
    ('could you tell me who you are?',('Kelus',)),
    ('thank you, that was helpful!',('welcome',)),
    ('can you explain what a variable is?',('name','value'))]


def evaluate(checkpoint, data='data/conversations-v3.jsonl'):
    model,header,_=Model.load(checkpoint)
    def answer(history):
        kept,omitted,tool=prepare_reply(model,history)
        return (tool[0],tool[1],omitted) if tool else (''.join(model.generate(kept,temperature=0)),'model',omitted)
    history=[];results=[]
    sequence=[('1+1?',lambda r:r=='1+1 = 2'),('hi',lambda r:'Hi!' in r),
        ('can you code?',lambda r:'Python' in r and 'JavaScript' in r and 'Yes' in r),
        ('nooo',lambda r:'misunderstood' in r.lower()),('what date is today',lambda r:"this computer's clock" in r),
        ('Write a Python square function.',lambda r:code_result(r,'square',[((3,),9),((-4,),16)])),
        ('What is a variable?',lambda r:'name' in r and 'value' in r),('what',lambda r:'misunderstood' in r.lower())]
    for prompt,check in sequence:
        history.append({'role':'user','content':prompt});reply,source,omitted=answer(history)
        history.append({'role':'assistant','content':reply})
        results.append({'prompt':prompt,'reply':reply,'source':source,'omitted':omitted,'passed':bool(check(reply))})
    training=set()
    for line in Path(data).read_text().splitlines():
        row=json.loads(line)
        training.update(m['content'] for m in row['messages'] if m['role']=='user')
    probe_results=[]
    for prompt,words in PROBES:
        assert prompt not in training, f'Probe overlap: {prompt}'
        reply,source,_=answer([{'role':'user','content':prompt}])
        probe_results.append({'prompt':prompt,'reply':reply,'source':source,'passed':any(w.casefold() in reply.casefold() for w in words)})
    return {'checkpoint':str(checkpoint),'sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        'metadata':header['metadata'],'scope':'Small disclosed development probes, not a general intelligence benchmark. Screenshot prompts deliberately covered by training or explicit helpers. Probe checks are keyword checks; read the raw outputs.',
        'sequence':results,'sequence_passed':sum(r['passed'] for r in results),'sequence_total':len(results),
        'probes':probe_results,'probes_passed':sum(r['passed'] for r in probe_results),'probes_total':len(probe_results)}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--data',default='data/conversations-v3.jsonl',help='Training file checked for probe overlap');a=p.parse_args()
    result=evaluate(a.checkpoint,a.data);Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('metadata','scope','sha256')},indent=2))
