"""Cide v5: exact identifiers and varied CRUD contracts, with SaaS replay.

Only local, generated reference functions are executed. Original reserved CRUD
cases are never used as training examples. This is component training, not a
collection of complete SaaS applications.
"""
from collections import Counter
import json
from pathlib import Path
import random
import re

from kelus.corpus import digest, rows
from kelus.crud_tasks import OPERATIONS, prompt, reference, verify
from kelus.cide_saas_eval import functions
from kelus.model import Model
from kelus.tokenizer import conversation
from .make_cide_crud import ENTITIES, PREFIXES
from .make_cide_continuation import VALIDATION

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/cide-contracts-v5'
PARENT = ROOT / 'training-runs/cide-continuation-v4/instruction.npz'


def build():
    if OUT.exists():
        raise FileExistsError('Prepared v5 data already exists; preserve it')
    model, _, _ = Model.load(PARENT)
    tokenizer, context = model.tokenizer, model.config.context
    del model
    records, seen, checked = [], set(), 0
    signatures = {'train': set(), 'validation': set()}

    def add(split, family, question, code):
        messages = [{'role': 'user', 'content': question},
                    {'role': 'assistant', 'content': '```python\n' + code + '```'}]
        key = json.dumps(messages, sort_keys=True)
        if key in seen:
            return
        if len(conversation(messages, tokenizer=tokenizer)[0]) > context:
            raise ValueError('Generated reference exceeds context')
        seen.add(key)
        records.append({'split': split, 'family': family, 'messages': messages})

    for split, count, seed in [('train', 4800, 9261), ('validation', 96, 9262)]:
        rng = random.Random(seed)
        entities = ENTITIES['train'] if split == 'train' else VALIDATION
        for index in range(count):
            operation = OPERATIONS[index % len(OPERATIONS)]
            entity = rng.choice(entities)

            name = rng.choice(PREFIXES[operation]) + '_' + rng.choice(entities) + '_' + str(rng.randrange(10000))
            spec = dict(operation=operation, table=entity + rng.choice(['s', '_records']),
                        owner=rng.choice(['owner_id', 'account_id', 'user_id']),
                        column=rng.choice(['title', 'name', 'label']),
                        db=rng.choice(['db', 'conn', 'connection']),
                        user=rng.choice(['actor', 'principal', 'member']),
                        item=rng.choice(['item_id', 'record_id', 'key']),
                        value=rng.choice(['text', 'value', 'new_title']),
                        limit=rng.choice([24, 48, 80, 120, 200]),
                        direction=rng.choice(['ASC', 'DESC']), name=name)
            code = reference(spec)
            verify(code, spec)
            checked += 1
            signatures[split].add(name)
            add(split, 'identifier_crud_' + operation, prompt(spec, index % 4), code)
        for index in range(count // 2):
            name = 'read_' + rng.choice(entities) + '_' + str(rng.randrange(100000))
            arg = rng.choice(['payload', 'value', 'record', 'entry'])
            key = rng.choice(['title', 'name', 'status', 'owner_id'])
            if index % 2:
                task, expression, sample, expected = 'Return its argument unchanged.', arg, {'title': 'test'}, {'title': 'test'}
            else:
                task = f'Return the value at key "{key}" in the dictionary argument. Let a missing key raise KeyError.'
                expression, sample, expected = f'{arg}["{key}"]', {key: 'test'}, 'test'
            code = f'def {name}({arg}):\n    return {expression}\n'
            fn = functions(code, [name])[name]
            assert fn(sample) == expected
            if index % 2 == 0:
                try:
                    fn({})
                except KeyError:
                    pass
                else:
                    raise AssertionError('Missing-key contract failed')
            checked += 1
            signatures[split].add(name)
            add(split, 'exact_signature', f'Write Python function {name}({arg}). {task} Preserve the exact function and argument names. Return only code.', code)
    assert not signatures['train'] & signatures['validation']


    source = ROOT / 'data/cide-continuation-v4/instructions.jsonl'
    reserved = re.compile(r'\b(?:' + '|'.join(ENTITIES['test'] + VALIDATION) + r')(?:s|_entries|_records|_items)?\b', re.I)
    old = [r for r in rows(source) if not reserved.search(json.dumps(r['messages']))]
    rng = random.Random(9263)
    for split, count in [('train', 6000), ('validation', 100)]:
        pool = [r for r in old if r['split'] == split]
        for row in rng.sample(pool, min(count, len(pool))):
            key = json.dumps(row['messages'], sort_keys=True)
            if key not in seen:
                seen.add(key)
                records.append(row)
    rng.shuffle(records)
    OUT.mkdir()
    data = OUT / 'instructions.jsonl'
    data.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    report = dict(parent=str(PARENT), parent_sha256=digest(PARENT),
                  replay_sha256=digest(source), data_sha256=digest(data),
                  counts=dict(Counter(r['split'] for r in records)),
                  families=dict(Counter(r.get('family', 'replay') for r in records)),
                  reference_contracts_executed=checked, validation_entities=VALIDATION,
                  excluded_test_entities=ENTITIES['test'],
                  limitation='Procedural identifier/CRUD exercises with prior SaaS/code replay; not complete apps. Prior reserved cases are reused evaluation, not a fresh benchmark.')
    (OUT / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    build()
