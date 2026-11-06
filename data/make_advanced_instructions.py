"""Prepare diverse local human-written instruction data; never mix in legacy reply templates."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re

from kelus.tokenizer import conversation, from_dict

ROOT = Path(__file__).resolve().parents[1]
REJECT = re.compile(r'chatgpt|openai|open ?assistant|LAION|as an ai language model|I do not have enough training|I have not learned to write', re.I)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(text):
    return ' '.join(text.lower().split())


def candidates(source):
    for index, line in enumerate((source / 'databricks-dolly-15k.jsonl').open()):
        row = json.loads(line)
        question = row['instruction'].strip()
        if row['context'].strip():
            question += '\n\n' + row['context'].strip()
        yield 'dolly/' + str(index), 'dolly/' + row['category'], [
            {'role': 'user', 'content': question}, {'role': 'assistant', 'content': row['response'].strip()}]
    with gzip.open(source / '2023-11-05_oasst2_ready.trees.jsonl.gz', 'rt') as stream:
        for index, line in enumerate(stream):
            node = json.loads(line)['prompt']
            if node.get('lang') != 'en':
                continue

            messages = []
            while node and len(messages) < 6:
                if node.get('deleted') or node.get('synthetic') or node.get('lang') != 'en':
                    break
                messages.append({'role': 'user' if node['role'] == 'prompter' else 'assistant', 'content': node['text'].strip()})
                replies = [r for r in node.get('replies', []) if not r.get('deleted') and not r.get('synthetic')]
                if node['role'] == 'prompter':
                    ranked = [r for r in replies if r.get('rank') is not None]
                    node = min(ranked, key=lambda r: r['rank']) if ranked else None
                else:
                    node = next((r for r in replies if r.get('replies')), None)
            if messages and messages[-1]['role'] == 'user':
                messages.pop()
            if messages:
                yield 'oasst2/' + str(index), 'oasst2/english', messages


def build(source, out, tokenizer_path, context):
    if out.exists():
        raise FileExistsError('Dataset exists; choose a new output directory')
    tokenizer = from_dict(json.loads(tokenizer_path.read_text()))
    seen_prompts, seen_answers, kept, skipped = set(), set(), [], Counter()
    for source_id, family, messages in candidates(source):
        if any(REJECT.search(m['content']) for m in messages):
            skipped['identity_or_stock_reply'] += 1
            continue
        answers = [normalized(m['content']) for m in messages if m['role'] == 'assistant']
        prompts = [normalized(m['content']) for m in messages if m['role'] == 'user']
        if not answers or any(len(a) < 40 for a in answers):
            skipped['short_answer'] += 1
            continue
        if any(p in seen_prompts for p in prompts) or any(a in seen_answers for a in answers):
            skipped['duplicate_prompt_or_answer'] += 1
            continue
        tokens, mask = conversation(messages, tokenizer=tokenizer)
        if len(tokens) > context + 1:
            skipped['too_long_no_truncation'] += 1
            continue

        key = hashlib.sha256(prompts[0].encode()).hexdigest()
        split = 'validation' if int(key[:8], 16) % 100 < 5 else 'train'
        kept.append(dict(split=split, family=family, source_id=source_id, messages=messages,
                         tokens=len(tokens), target_tokens=sum(mask)))
        seen_prompts.update(prompts)
        seen_answers.update(answers)
    if not any(r['split'] == 'validation' for r in kept):
        raise ValueError('No validation examples')
    out.mkdir(parents=True)
    data = out / 'instructions.jsonl'
    data.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in kept))
    manifest = dict(data_sha256=sha(data), tokenizer_sha256=sha(tokenizer_path), context=context,
                    counts=dict(Counter(r['split'] for r in kept)),
                    families=dict(Counter(r['family'] for r in kept)),
                    total_tokens=sum(r['tokens'] for r in kept),
                    assistant_target_tokens=sum(r['target_tokens'] for r in kept),
                    skipped=dict(skipped), exact_prompt_and_answer_overlap=0,
                    sources=[dict(path=str(p), sha256=sha(p), license=license, url=url) for p, license, url in [
                        (source / 'databricks-dolly-15k.jsonl', 'CC BY-SA 3.0', 'https://huggingface.co/datasets/databricks/databricks-dolly-15k'),
                        (source / '2023-11-05_oasst2_ready.trees.jsonl.gz', 'Apache-2.0', 'https://huggingface.co/datasets/OpenAssistant/oasst2')]],
                    limitations='Human dataset labels are not a guarantee of correctness. Exact deduplication is not semantic deduplication. First pilot is English only. Validation loss does not certify general intelligence.')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (out / 'ATTRIBUTION.md').write_text('# Instruction data sources\n\n'
        'Databricks Dolly 15k contributors — CC BY-SA 3.0: https://huggingface.co/datasets/databricks/databricks-dolly-15k\n\n'
        'OpenAssistant OASST2 contributors — Apache-2.0: https://huggingface.co/datasets/OpenAssistant/oasst2\n\n'
        'This filtered derivative preserves source identifiers, removes exact repeated prompts/replies and overly long examples, '
        'and formats retained conversations for Kani. Dolly-derived records remain subject to CC BY-SA 3.0. '
        'Source hashes and selection rules are in manifest.json and data/make_advanced_instructions.py.\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/everyday-sources')
    parser.add_argument('--out', type=Path, default=ROOT / 'training-runs/kani-advanced-instruction-pilot/data')
    parser.add_argument('--tokenizer', type=Path, default=ROOT / 'kaggle/kani_tokenizer.json')
    parser.add_argument('--context', type=int, default=1024)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.out, args.tokenizer, args.context), indent=2))
