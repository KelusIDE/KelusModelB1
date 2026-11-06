"""Select varied paragraph targets from the existing attributed instruction corpus.

This is format-focused curation, not a factual review of every community answer.
Original train/validation assignments and source IDs are retained. Short-answer
replay helps avoid teaching the model to be verbose for every request.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

from kelus.tokenizer import conversation, from_dict

ROOT = Path(__file__).resolve().parents[1]


def build(source, out, cases, tokenizer_path):
    if out.exists():
        raise FileExistsError('Choose a fresh dataset directory')
    tokenizer = from_dict(json.loads(tokenizer_path.read_text()))
    probes = json.loads(cases.read_text())
    reserved = {' '.join((c.get('prompt') or c['messages'][0]['content']).lower().split()) for c in probes}
    paragraphs, replay, seen = [], [], set()
    for line in source.read_text().splitlines():
        row = json.loads(line)
        messages = row['messages']
        if len(messages) != 2:
            continue
        question, answer = (m['content'].strip() for m in messages)
        key = ' '.join(answer.lower().split())
        if key in seen or ' '.join(question.lower().split()) in reserved:
            continue
        words = len(answer.split())
        paragraph = (65 <= words <= 160 and '\n' not in answer and '```' not in answer
                     and len(re.findall(r'[.!?](?:\s|$)', answer)) >= 4
                     and not re.search(r'\b(?:list|bullets?|code|poem|haiku|translate|equation|step.by.step)\b', question, re.I))
        short = 8 <= words <= 45 and '```' not in answer
        if not (paragraph or short):
            continue
        if paragraph:
            low, high = max(50, (words // 20) * 20), (words // 20 + 2) * 20
            question = f'Write your answer as one clear paragraph of {low}–{high} words.\n\n' + question
        changed = [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}]
        ids, _ = conversation(changed, tokenizer=tokenizer)
        if len(ids) > 1024:
            continue
        seen.add(key)
        (paragraphs if paragraph else replay).append(dict(split=row['split'], source_id=row['source_id'],
            family=row['family'], task='paragraph' if paragraph else 'short_replay', messages=changed))
    rng = random.Random(4291)
    rng.shuffle(paragraphs); rng.shuffle(replay)

    selected = paragraphs + replay[:max(1, len(paragraphs) // 4)]
    rng.shuffle(selected)
    if len(paragraphs) < 100 or not any(r['split'] == 'validation' for r in selected):
        raise ValueError('Insufficient varied paragraphs or validation coverage')
    out.mkdir(parents=True)
    data = out / 'instructions.jsonl'
    data.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = dict(source=str(source), source_sha256=sha(source), data_sha256=sha(data),
        cases_sha256=sha(cases), counts=dict(Counter(r['split'] for r in selected)),
        tasks=dict(Counter(r['task'] for r in selected)), families=dict(Counter(r['family'] for r in selected)),
        exact_answer_duplicates=0, tokenizer_sha256=sha(tokenizer_path),
        limitation='Original dataset splits preserved; parent may already have seen related examples. Format filtering is not factual review or proof of generalization.')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (out / 'ATTRIBUTION.md').write_text((source.parent / 'ATTRIBUTION.md').read_text()
        + '\nParagraph selection and prompt formatting: data/make_paragraphs.py. Original source identifiers and licenses are retained.\n')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT / 'training-runs/kani-advanced-instruction-pilot/data/instructions.jsonl')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cases', type=Path, default=ROOT / 'evaluation/paragraph_v1.json')
    p.add_argument('--tokenizer', type=Path, default=ROOT / 'kaggle/kani_tokenizer.json')
    a = p.parse_args()
    print(json.dumps(build(a.source, a.out, a.cases, a.tokenizer), indent=2))
