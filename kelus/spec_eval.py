"""Score function-writing on held-out specifications from an instruction curriculum.

Uses the curriculum's validation-split python_spec records: names, operations and
phrasings the model never trained on. Each reply is checked by the bounded AST
interpreter in kelus.evaluate on three inputs; generated code is never executed.

    python3 -m kelus.spec_eval --checkpoint checkpoints/kani-b2-2f.best.npz
"""
import argparse
import ast
import json
from pathlib import Path
import random
from .evaluate import code_result
from .model import Model
import re


def signature_prefix(prompt):
    """The function header a user stated in the request, as a decoding prefix."""
    stated = re.search(r'\b([A-Za-z_]\w*)\(([^)]*)\)', prompt)
    if stated:
        return f'```python\ndef {stated[1]}({stated[2]}):'
    named = re.search(r'\b(?:named|called)\s+([A-Za-z_]\w*)', prompt)
    if not named:
        return ''
    takes = re.search(r'\btakes\s+([A-Za-z_]\w*(?:(?:,\s*|\s+and\s+)[A-Za-z_]\w*)*)\s+and\s+(?:returns|adds|subtracts|multiplies|divides|squares|doubles)\b', prompt)
    if takes:
        params = re.split(r',\s*|\s+and\s+', takes[1])
        return f'```python\ndef {named[1]}({", ".join(params)}):'
    return f'```python\ndef {named[1]}('


def cases(data, limit, seed=7, family='python_spec'):
    rows = [json.loads(line) for line in Path(data).read_text(encoding='utf-8').splitlines() if line.strip()]
    picked = [r for r in rows if r.get('family') == family and r['split'] == 'validation']
    random.Random(seed).shuffle(picked)
    out = []
    for row in picked[:limit]:
        if 'accept' in row:  # chat families: exact reply after the conversation so far
            out.append((row['messages'][:-1], row.get('situation'), row['accept']))
            continue
        prompt, answer = row['messages'][0]['content'], row['messages'][1]['content']
        function = ast.parse(answer.strip('`').removeprefix('python').strip()).body[0]
        params = [a.arg for a in function.args.args]
        expression = ast.unparse(function.body[0].value)
        rng, trials = random.Random(prompt), []
        while len(trials) < 3:
            inputs = tuple(rng.randint(-9, 12) for _ in params)
            try:
                trials.append((inputs, eval(expression, {}, dict(zip(params, inputs)))))  # reference answer from our own generator
            except ZeroDivisionError:
                continue
        out.append(([row['messages'][0]], function.name, trials))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', default='data/instruct-v1.jsonl')
    parser.add_argument('--limit', type=int, default=200)
    parser.add_argument('--family', default='python_spec', help='python_spec, python_compose, python_power, or a chat family with accepted replies such as feelings')
    parser.add_argument('--prefix', action='store_true', help='start replies with the stated signature (constrained decoding)')
    parser.add_argument('--out')
    args = parser.parse_args()
    model, header, _ = Model.load(args.checkpoint)
    results = []
    for messages, name, expected in cases(args.data, args.limit, family=args.family):
        prompt = messages[-1]['content']
        chat = isinstance(expected[0], str)
        reply = ''.join(model.generate(messages, max_tokens=120, temperature=0,
                                       prefix=signature_prefix(prompt) if args.prefix and not chat else ''))
        passed = reply.strip() in expected if chat else code_result(reply, name, expected)
        results.append({'prompt': prompt, 'function': name, 'reply': reply, 'passed': passed})
    passed = sum(r['passed'] for r in results)
    report = {'checkpoint': args.checkpoint, 'family': args.family, 'prefix': args.prefix, 'steps': header.get('metadata', {}).get('steps'), 'passed': passed, 'total': len(results),
              'accuracy': round(passed / max(1, len(results)), 3),
              'scope': 'Held-out Python specifications (unseen operation templates and names) from the curriculum validation split.',
              'results': results}
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('family', 'prefix', 'steps', 'passed', 'total', 'accuracy')}))
    for r in [r for r in results if not r["passed"]][:5]:
        print('FAIL', r['prompt'], '->', repr(r['reply'][:120]))


if __name__ == '__main__':
    main()
