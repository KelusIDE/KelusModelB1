"""Prepare verified, varied CRUD requests; keep completed v2 data/checkpoints intact."""
from collections import Counter
import json
from pathlib import Path
import random

from kelus.corpus import digest, identity, rows
from kelus.crud_tasks import OPERATIONS, prompt, reference, verify
from kelus.model import Model
from kelus.tokenizer import conversation

ROOT=Path(__file__).resolve().parents[1]
ENTITIES={
 'train':'task note project ticket article invoice contact board comment event bookmark document account folder message order product team user schedule post item file customer payment report alert card category label group photo request lesson job meeting page subscription review link'.split(),
 'validation':['reminder','workspace','memo','appointment'],
 'test':['shipment','submission','journal','delivery'],
}
PREFIXES={'list':['list','fetch','read'], 'get':['get','find','lookup'], 'count':['count','total','summarize'],
          'create':['create','add','insert'], 'update':['update','edit','rename'], 'delete':['delete','remove','erase']}


def specs(split):
    rng=random.Random({'train':9131,'validation':9132,'test':9133}[split])
    for entity in ENTITIES[split]:
        for operation in OPERATIONS:
            for variant in range(12 if split=='train' else 2 if split=='validation' else 1):
                yield dict(operation=operation,table=entity+'s',owner=rng.choice(['owner_id','account_id','user_id']),
                           column=rng.choice(['title','name','label']),db=rng.choice(['db','conn','connection']),
                           user=rng.choice(['user_id','actor','principal','owner']),item=rng.choice(['item_id','record_id','key']),
                           value=rng.choice(['title','text','value','new_title']),limit=rng.choice([32,64,80,120,200]),
                           direction=rng.choice(['ASC','DESC']),name=rng.choice(PREFIXES[operation])+'_'+entity+('s' if operation in ('list','count') else ''))


def main():
    root=ROOT/'data/cide-crud-v3'
    if root.exists(): raise FileExistsError('Prepared output exists; preserve it before rebuilding')
    parent=ROOT/'training-runs/cide-saas-v2/cide-saas-v2-candidate.npz'
    model,_,_=Model.load(parent);tokenizer=model.tokenizer;del model
    records=[];seen=set();verified=0
    for split in ('train','validation','test'):
        cases=[]
        for spec in specs(split):
            code=reference(spec);verify(code,spec);verified+=1
            if split=='test':
                cases.append(dict(spec=spec,prompt=prompt(spec,3)));continue
            for style in ([0,1,2] if split=='train' else [3]):
                question=prompt(spec,style);answer='```python\n'+code+'```'
                key=identity(question+'\n'+answer)
                if key in seen:continue
                seen.add(key)
                row=dict(split=split,family='crud_'+spec['operation'],messages=[dict(role='user',content=question),dict(role='assistant',content=answer)])
                assert len(conversation(row['messages'],tokenizer=tokenizer)[0])<=1000
                records.append(row)
        if split=='test': test_cases=cases

    rng=random.Random(9134);old=list(rows(ROOT/'data/cide-v1.jsonl'))
    for split,n in [('train',800),('validation',12)]:
        pool=[r for r in old if r['split']==split]
        records.extend(rng.sample(pool,n))
    rng.shuffle(records);root.mkdir(parents=True)
    (root/'instructions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    (root/'test-cases.json').write_text(json.dumps(test_cases,indent=2)+'\n')
    report=dict(parent_sha256=digest(parent),instructions_sha256=digest(root/'instructions.jsonl'),test_cases_sha256=digest(root/'test-cases.json'),
                reference_contracts_executed=verified,counts=dict(Counter(r['split'] for r in records)),
                families=dict(Counter(r['family']+':'+r['split'] for r in records)),entity_splits=ENTITIES,
                purpose='Correct the observed failure to follow CRUD contracts with new identifiers and wording.',
                limitations='Procedural variations of six small contracts, not thousands of independent apps. Test entities are excluded from training/validation. Existing five v2 checks are now development checks, not untouched tests.')
    (root/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
