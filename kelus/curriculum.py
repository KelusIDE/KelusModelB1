"""Prepare attributed multilingual instruction records with task-family splits.

Does not download data, generate answers, train a model, or execute code.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from .languages import canonical, scope, scope_sha256
from .tokenizer import conversation

SPLITS = ('train', 'validation', 'test')
FIELDS = ('code_language', 'reply_language', 'family', 'source', 'reviewed')


def family_split(family):
    bucket = int(hashlib.sha256(('kelus-instruction-v1:' + family).encode()).hexdigest()[:8], 16) % 10
    return 'test' if bucket == 0 else 'validation' if bucket == 1 else 'train'


def normalize(records, assign_splits=False):
    """Keep every translation/variant of a family in the same split."""
    output, families, seen = [], {}, {}
    for number, source in enumerate(records, 1):
        try:
            row = dict(source)
            if any(key not in row for key in FIELDS):
                raise ValueError('Missing language/family/provenance fields')
            row['family'] = str(row['family']).strip().casefold()
            if not row['family'] or row['family'] in ('none', 'null'):
                raise ValueError('Task family must be a nonempty semantic group')
            row['code_language'] = canonical(row['code_language'], 'programming') if row['code_language'] is not None else None
            row['reply_language'] = canonical(row['reply_language'], 'chat')
            if row['reviewed'] is not True:
                raise ValueError('Unreviewed example; inspect its answer and provenance first')
            provenance = row['source']
            if not isinstance(provenance, dict) or any(not isinstance(provenance.get(k), str) or not provenance[k].strip() for k in ('id', 'license', 'reference')):
                raise ValueError('Source needs nonempty id, license and reference')
            messages = row['messages']
            conversation(messages)  # Existing role/UTF-8 validator, no generated-code execution.
            if len(messages) % 2 or messages[-1]['role'] != 'assistant' or any(not m['content'].strip() for m in messages):
                raise ValueError('Training conversations must end with a nonempty assistant answer')
            split = row.get('split')
            if split is None and assign_splits:
                split = family_split(row['family'])
            if split not in SPLITS:
                raise ValueError('Explicit train/validation/test split required')
            row['split'] = split
            old_split = families.setdefault(row['family'], split)
            if old_split != split:
                raise ValueError('Task family crosses splits (including translations/variants)')
            key = hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            label = (row['family'], split, row['code_language'], row['reply_language'])
            if key in seen:
                if seen[key] != label:
                    raise ValueError('Identical conversation has conflicting family, split or language labels')
                continue
            seen[key] = label
            output.append(row)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f'Instruction record {number}: {error}') from error
    return output


def coverage(records):
    targets = scope()
    counts = {s: Counter() for s in SPLITS}
    families = {s: set() for s in SPLITS}
    answers = {s: set() for s in SPLITS}
    cells = {s: Counter() for s in SPLITS}
    for row in records:
        split = row['split']
        counts[split]['examples'] += 1
        counts[split]['chat:' + row['reply_language']] += 1
        if row['code_language']:
            counts[split]['code:' + row['code_language']] += 1
        cells[split][(row['code_language'] or 'chat') + '/' + row['reply_language']] += 1
        families[split].add(row['family'])
        answers[split].add(row['messages'][-1]['content'])
    return {'scope_sha256': scope_sha256(), 'status': 'data_coverage_only_not_model_ability',
            'splits': {s: {'counts': dict(counts[s]), 'families': len(families[s]),
                           'distinct_final_answers': len(answers[s]), 'cells': dict(cells[s]),
                           'missing_code': [k for k in targets['programming'] if not counts[s]['code:' + k]],
                           'missing_chat': [k for k in targets['chat'] if not counts[s]['chat:' + k]]}
                       for s in SPLITS}}


def read_records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def prepare(inputs, destination, require_all=False):
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError('Output already exists; use a new directory to preserve the prepared split')
    records = normalize([r for path in inputs for r in read_records(path)], assign_splits=True)
    report = coverage(records)
    if not records:
        raise ValueError('No reviewed instruction examples supplied')
    if require_all and any(report['splits'][s]['missing_code'] or report['splits'][s]['missing_chat'] for s in SPLITS):
        raise ValueError('Requested languages are missing from train, validation or test; inspect coverage before training')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.curriculum-', dir=destination.parent))
    try:
        for filename, splits in [('instruction.jsonl', {'train', 'validation'}), ('heldout.jsonl', {'test'})]:
            selected = sorted((r for r in records if r['split'] in splits), key=lambda r: (r['family'], r['reply_language'], r['code_language'] or '', json.dumps(r['messages'], sort_keys=True)))
            (temporary / filename).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected), encoding='utf-8')
        report['inputs'] = [{'path': str(Path(p).resolve()), 'sha256': hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in inputs]
        report['files'] = {name: hashlib.sha256((temporary / name).read_bytes()).hexdigest() for name in ('instruction.jsonl', 'heldout.jsonl')}
        (temporary / 'coverage.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', nargs='+', help='Reviewed JSONL instruction records')
    parser.add_argument('--out', help='New prepared dataset directory')
    parser.add_argument('--require-all', action='store_true', help='Require every target language in each split; coverage is not evidence of quality')
    parser.add_argument('--targets', action='store_true')
    args = parser.parse_args()
    if args.targets:
        print(json.dumps(scope(), ensure_ascii=False, indent=2))
    elif not args.input or not args.out:
        parser.error('Specify --input and --out, or --targets')
    else:
        print(json.dumps(prepare(args.input, args.out, args.require_all), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
