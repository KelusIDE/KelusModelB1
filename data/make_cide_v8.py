"""Verified paraphrases and repairs for Cide; evaluation wording stays out of training.

This is a focused SQLite/component curriculum, not evidence of general SaaS skill.
Existing source splits and the production checkpoint are preserved.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from kelus.crud_tasks import OPERATIONS, reference, signature, verify
from kelus.tokenizer import conversation, from_dict

ROOT = Path(__file__).resolve().parents[1]
ENTITIES = {
    'train': ['task', 'note', 'ticket', 'invoice', 'bookmark', 'project', 'contact', 'event'],
    'validation': ['rental', 'parcel'],
    'test': ['voucher', 'reservation'],
}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def question(s, style):
    d, u, i, v = (s[k] for k in ('db', 'user', 'item', 'value'))
    t, o, c = (s[k] for k in ('table', 'owner', 'column'))
    action = {
        'list': f'Give me every (id, {c}) tuple for {u}, sorted by id {s["direction"]}.',
        'get': f'Look up id {i} belonging to {u}; return (id, {c}) or None.',
        'count': f'Count the rows belonging to {u} and return an integer.',
        'create': f'Add a row with {o} set to {u} and {c} set to stripped {v}. Commit and return lastrowid.',
        'update': f'Set {c} to stripped {v} only for id {i} belonging to {u}. Commit; return True for one changed row, otherwise False.',
        'delete': f'Remove id {i} only if it belongs to {u}. Commit; return True for one removed row, otherwise False.',
    }[s['operation']]
    validation = (f' Reject non-string {v}, empty stripped text, or stripped text longer than {s["limit"]} characters with ValueError.'
                  if s['operation'] in ('create', 'update') else '')
    schema = f'{d} is a SQLite connection. Table {t} has id INTEGER PRIMARY KEY, {o} INTEGER, {c} TEXT. Ownership is {o}={u}.'
    safe = ' Bind values using ? placeholders. Return Python code only.'
    task = action + validation
    sig = signature(s)

    styles = [
        f'Implement Python {sig}. {task} {schema}{safe}',
        f'Can you write {sig} in Python? {schema} {task}{safe}',
        f'{schema}\n{task}\nUse this exact Python signature: {sig}.{safe}',
        f'I need a database helper called {sig}. {task}\n{schema}{safe}',
        f'Finish this Python function:\n```python\ndef {sig}:\n    pass\n```\n{schema} {task}{safe}',
        f'Write Python code for this request. {task} The function must be {sig}. {schema}{safe}',
        f'Fix the syntax and implement the behavior in this Python stub:\n```python\ndef python {sig}:\n    pass\n```\n{schema} {task}{safe}',
        f'{task}\n{schema}\nPlease use {sig}, keeping every argument name exactly as written.{safe}',

        f'Python task: {schema} Requirements: {task} Function signature: {sig}.{safe}',
        f'Please implement {sig}. Here is what it should do: {task} Database details: {schema}{safe}',

        f'Build a Python helper with the interface {sig}. Expected behavior: {task} Storage: {schema}{safe}',
        f'For {sig}, use these database details: {schema} The desired result is: {task}{safe}',
    ]
    return styles[style]


def specs(split, per_operation):
    rng = random.Random({'train': 481, 'validation': 482, 'test': 483}[split])
    for index in range(per_operation):
        for op in OPERATIONS:
            entity = rng.choice(ENTITIES[split])
            yield dict(operation=op, table=entity + '_records', name=op + '_' + entity + '_' + str(index),
                       owner=rng.choice(['owner_id', 'account_id']), column=rng.choice(['title', 'label']),
                       db=rng.choice(['db', 'conn']), user=rng.choice(['actor', 'member']),
                       item=rng.choice(['record_id', 'key']), value=rng.choice(['text', 'value']),
                       limit=rng.choice([40, 72, 120]), direction=rng.choice(['ASC', 'DESC']))


def build(out, parent):
    if out.exists():
        raise FileExistsError(f'{out} exists; refusing to overwrite prepared data')
    with np.load(parent, allow_pickle=False) as archive:
        header = json.loads(str(archive['header']))
    tokenizer = from_dict(header['tokenizer'])
    context = header['config']['context']
    records, cases, identities = [], [], set()
    checked = 0

    def add(row):
        key = json.dumps(row['messages'], sort_keys=True)
        if key in identities:
            return
        if len(conversation(row['messages'], tokenizer=tokenizer)[0]) > context:
            raise ValueError('Example would be truncated during training')
        identities.add(key)
        records.append(row)

    for split, count, styles in [('train', 60, range(8)), ('validation', 3, range(8, 10)), ('test', 1, range(10, 12))]:
        for spec in specs(split, count):
            code = reference(spec)
            verify(code, spec)
            checked += 1
            for style in styles:
                prompt = question(spec, style)
                if split == 'test':
                    cases.append({'spec': spec, 'prompt': prompt})
                else:
                    add({'split': split, 'family': 'paraphrase_' + spec['operation'], 'style': style,
                         'messages': [{'role': 'user', 'content': prompt},
                                      {'role': 'assistant', 'content': '```python\n' + code + '```'}]})



    source = ROOT / 'data/cide-v7.jsonl'
    pools = {}
    for line in source.open():
        row = json.loads(line)
        if row['split'] not in ('train', 'validation'):
            continue
        kind = 'website' if row.get('source') == 'make_websites' else 'code'
        pools.setdefault((row['split'], kind), []).append(row)
    rng = random.Random(484)
    for (split, kind), pool in sorted(pools.items()):
        count = 1440 if split == 'train' else 24
        for row in rng.sample(pool, min(count, len(pool))):
            add(row)
    rng.shuffle(records)
    out.mkdir(parents=True)
    data = out / 'instructions.jsonl'
    data.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records))
    (out / 'test-cases.json').write_text(json.dumps(cases, indent=2) + '\n')
    report = {'parent': str(parent), 'parent_sha256': digest(parent), 'source_sha256': digest(source),
              'data_sha256': digest(data), 'counts': dict(Counter(r['split'] for r in records)),
              'reference_contracts_executed': checked, 'test_cases': len(cases),
              'styles': {'train': list(range(8)), 'validation': [8, 9], 'test': [10, 11]},
              'entities': ENTITIES, 'context': context,
              'scope': 'Procedural SQLite components and existing replay; not general app-building proof. Test wording/entities excluded from newly generated training and validation.'}
    (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'data/cide-v8')
    parser.add_argument('--parent', type=Path, default=ROOT / 'checkpoints/cide-2-release.npz')
    args = parser.parse_args()
    print(json.dumps(build(args.out, args.parent), indent=2))
