"""Latest-request following: diverse instruction replay plus verifiable minimal contrasts.

Reported failures are development probes, excluded verbatim from training.
Generated operands/nouns and validation groups differ. These component exercises
do not establish open-domain conversation or general reasoning capability.
"""
from collections import Counter
import json
from pathlib import Path
import random

from kelus.corpus import digest
from kelus.tokenizer import conversation, from_dict

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'training-runs/kani-topic-v1/data'
REPLAY = ROOT / 'training-runs/kani-advanced-instruction-pilot/data/instructions.jsonl'


def number(n):
    small = 'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split()
    if n < 20:
        return small[n]
    return ['','','twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety'][n//10] + (' ' + small[n%10] if n%10 else '')


def build():
    if OUT.exists():
        raise FileExistsError('Prepared curriculum exists; preserve it')
    tokenizer = from_dict(json.loads((ROOT / 'kaggle/kani_tokenizer.json').read_text()))
    records = [json.loads(line) for line in REPLAY.open()]
    seen = {json.dumps(r['messages'], sort_keys=True) for r in records}
    checked = Counter()
    train_nouns = ['vase','lamp','statue','parcel','painting','clock','bottle','basket','figurine','plant']
    valid_nouns = ['lantern','teapot','ornament']
    containers = ['crate','carton','locker','cabinet','chest']
    for split, count, seed in [('train', 2400, 2811), ('validation', 120, 2812)]:
        rng = random.Random(seed)
        nouns = train_nouns if split == 'train' else valid_nouns
        for i in range(count):
            family = i % 3
            if family == 0:

                a = rng.randrange(0, 70) if split == 'train' else rng.randrange(70, 100)
                b = rng.randrange(0, 30)
                sign = rng.choice(['+', '-'])
                operation = 'plus' if sign == '+' else 'minus'
                answer = a+b if sign == '+' else a-b
                questions = [f'whats {a}{sign}{b} haha', f'what is {number(a)} {operation} {number(b)}',
                             f'Can you calculate {a} {sign} {b}? Just the number please.', f'{a} {sign} {b} = ?']
                q = questions[(i//3) % len(questions)]
                messages = [{'role':'user','content':q}, {'role':'assistant','content':str(answer)}]
                assert int(messages[-1]['content']) == (a+b if sign == '+' else a-b)
                label = 'word_and_casual_arithmetic'
            elif family == 1:
                item, box = rng.choice(nouns), rng.choice(containers)
                modifier = rng.choice(['blue','red','green','silver','wooden','striped'])
                big = bool(rng.randrange(2))
                size = 'big' if big else 'small'
                q = f'The {modifier} {item} cannot fit inside the {box} because it is too {size}. What is too {size}? Answer only with the object name.'
                answer = item if big else box
                messages = [{'role':'user','content':q}, {'role':'assistant','content':answer}]
                assert answer == (item if size == 'big' else box)
                label = 'reference_minimal_contrasts'
            else:
                item = rng.choice(nouns)
                old, new = rng.sample(['blue','red','green','silver','white','black','purple','orange'], 2)
                q = rng.choice([f'Correction: the {item} is {new}, not {old}. What color is it now? Answer with one word.',
                                f'I made a mistake. My {item} is actually {new}. Tell me the corrected color only.'])
                messages = [{'role':'user','content':f'My {item} is {old}.'},
                            {'role':'assistant','content':f'Your {item} is {old}.','supervise':False},
                            {'role':'user','content':q}, {'role':'assistant','content':new}]
                label = 'latest_turn_correction'
            key = json.dumps(messages, sort_keys=True)
            if key in seen:
                continue
            ids, _ = conversation(messages, tokenizer=tokenizer)
            assert len(ids) <= 1025
            seen.add(key)
            checked[label] += 1
            records.append(dict(split=split, family=label, source='procedural_verified', messages=messages))

    random.Random(2813).shuffle(records)
    OUT.mkdir(parents=True)
    path = OUT / 'instructions.jsonl'
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records))
    report = dict(data_sha256=digest(path), replay_sha256=digest(REPLAY),
                  counts=dict(Counter(r['split'] for r in records)), generated_reference_checks=dict(checked),
                  validation_nouns=valid_nouns, validation_first_operands='70..99',
                  limitation='Generated component exercises plus varied human-written instruction replay. Prior reported failures are development tests, not unseen benchmarks. No claim of broad factual correctness.')
    (OUT / 'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    build()
