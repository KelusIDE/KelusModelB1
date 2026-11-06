"""Expand Cide's local coding curriculum without importing held-out contracts."""
from collections import Counter
import json
from pathlib import Path
import random
import re

from kelus.corpus import digest, identity, rows
from kelus.crud_tasks import OPERATIONS, prompt, reference, verify
from kelus.model import Model
from kelus.tokenizer import conversation
from .make_cide_crud import ENTITIES, PREFIXES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/cide-continuation-v4'
PARENT = ROOT / 'training-runs/cide-crud-v3/instruction.npz'
VALIDATION = ['milestone', 'receipt', 'poll', 'reminder']


def specs(split, variants=40):
    rng = random.Random(9241 if split == 'train' else 9242)
    entities = ENTITIES['train'] if split == 'train' else VALIDATION
    for entity in entities:
        for operation in OPERATIONS:
            for variant in range(variants if split == 'train' else 4):
                table = entity + rng.choice(['s', '_entries', '_records', '_items'])
                yield dict(operation=operation, table=table,
                           owner=rng.choice(['owner_id', 'account_id', 'user_id', 'creator_id', 'member_id']),
                           column=rng.choice(['title', 'name', 'label', 'summary', 'description']),
                           db=rng.choice(['db', 'conn', 'connection', 'database', 'store']),
                           user=rng.choice(['user_id', 'actor', 'principal', 'owner', 'member']),
                           item=rng.choice(['item_id', 'record_id', 'key', 'entry_id', 'target_id']),
                           value=rng.choice(['title', 'text', 'value', 'new_title', 'content']),
                           limit=rng.choice([16, 24, 32, 48, 64, 80, 120, 160, 200, 240]),
                           direction=rng.choice(['ASC', 'DESC']),
                           name=rng.choice(PREFIXES[operation]) + '_' + table + '_' + str(variant))


def build(out=OUT):
    if out.exists():
        raise FileExistsError('Curriculum already exists; preserve the prepared data')
    model, _, _ = Model.load(PARENT)
    tokenizer = model.tokenizer
    del model
    records, seen, checked = [], set(), 0
    def add(row):
        key = identity(json.dumps(row['messages'], sort_keys=True))
        if key not in seen:
            seen.add(key)
            records.append(row)
    for split in ('train', 'validation'):
        for index, spec in enumerate(specs(split)):
            code = reference(spec)
            verify(code, spec)
            checked += 1
            messages = [{'role': 'user', 'content': prompt(spec, index % 3 if split == 'train' else 3)},
                        {'role': 'assistant', 'content': '```python\n' + code + '```'}]
            if len(conversation(messages, tokenizer=tokenizer)[0]) > 1000:
                raise ValueError('A generated contract exceeds the model context')
            add({'split': split, 'family': 'varied_crud_' + spec['operation'], 'messages': messages})


    excluded = ENTITIES['test'] + VALIDATION
    reserved = re.compile(r'\b(?:' + '|'.join(excluded) + r')(?:s|_entries|_records|_items)?\b', re.I)
    rng = random.Random(9243)
    sources = {}
    for relative, count in [('data/cide-saas-v2/instructions.jsonl', 6000), ('data/cide-v1.jsonl', 800)]:
        path = ROOT / relative
        source = [row for row in rows(path) if not reserved.search(json.dumps(row['messages']))]
        for split, limit in [('train', count), ('validation', 24)]:
            pool = [row for row in source if row['split'] == split]
            for row in rng.sample(pool, min(limit, len(pool))):
                add(row)
        sources[relative] = digest(path)
    rng.shuffle(records)
    out.mkdir(parents=True)
    data = out / 'instructions.jsonl'
    data.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records))
    report = {'parent': str(PARENT), 'parent_sha256': digest(PARENT), 'sources': sources,
              'data_sha256': digest(data), 'counts': dict(Counter(row['split'] for row in records)),
              'families': dict(Counter(row.get('family', 'replay') for row in records)),
              'reference_contracts_executed': checked, 'validation_entities': VALIDATION,
              'excluded_test_entities': ENTITIES['test'],
              'limitations': 'Procedural CRUD variations plus existing SaaS/code replay, not independent complete applications. Validation is developmental; reserved behavior checks are excluded from training.'}
    (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    print(json.dumps(build(), indent=2))
