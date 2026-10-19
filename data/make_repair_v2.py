"""Balance short clarifications and concepts; clock facts stay in the clock helper."""
import json
from pathlib import Path
import random
from make_repair import build_repair
from make_conversations import build

def build_v2():
    rng=random.Random(915)
    records=[r for r in build_repair() if not any('current date or time from my training' in m['content'] for m in r['messages'])]
    base=build()
    singles=[r for r in base if r['split']=='train' and len(r['messages'])==2]
    chat=[r for r in singles if r['messages'][0]['content'].lower().rstrip('!.?') in ('what','huh','hi','hello','thanks','who are you','what can you do')]
    concepts=[r for r in singles if r['messages'][0]['content'].startswith('explain ') and not r['messages'][0]['content'].startswith('explain a ')]
    for row in concepts:
        topic=row['messages'][0]['content'].removeprefix('explain ')
        answer=row['messages'][1]['content']
        for prompt in [f'Can you describe a {topic}?', f'Tell me what a {topic} is please',f'How would you explain {topic}?', f'Please explain {topic} to me', f'I want to understand {topic}', f'Explain what {topic} means to a beginner']:
            added={'split':'train','messages':[{'role':'user','content':prompt},{'role':'assistant','content':answer}]}
            records.append(added);chat.append(added)
    for i in range(1800):
        target=rng.choice(chat)
        prior=rng.choice(singles)
        records.append({'split':'train','messages':prior['messages']+target['messages']})
    # Keep only one preceding exchange, matching the deployed runtime.
    for row in records: row['messages']=row['messages'][-4:]
    unique={json.dumps(r['messages'],sort_keys=True):r for r in records}
    return list(unique.values())

if __name__=='__main__':
    rows=build_v2();Path(__file__).with_name('repair-v2.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    print({s:sum(r['split']==s for r in rows) for s in ('train','validation')})
