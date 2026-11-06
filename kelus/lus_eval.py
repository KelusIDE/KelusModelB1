"""Can Cide write Kel (.lus, the user's own language)? Held-out tasks, none of them a kind that data/make_lus.py trains on.

    .venv/bin/python -m kelus.lus_eval --checkpoint checkpoints/cide-4-5.npz --out reports/lus.json

The real compiler (~/Downloads/kel, through data/check_lus.mjs) checks each reply: a ```lus block that parses and passes
Kel's type checker. Nothing is run, so a program that compiles can still give the wrong answer; replies are kept for reading.
The tasks must differ from make_lus.py's templates, so that this measures writing something new.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time

from .model import Model

ROOT = Path(__file__).resolve().parent.parent
FENCE = re.compile(r'```\s*([A-Za-z+#]*)[^\n]*\n(.*?)(?:```|$)', re.S)
ASKS = ['Write a Kel program that {}.', 'write a .lus program that {}', 'In Kel, write a program that {}.']
TASKS = [
    'prints the triangular number for 12', 'adds up the squares of the numbers from 1 to 8', 'counts how many divisors 36 has',
    'checks whether 28 is a perfect number', 'finds the least common multiple of 4 and 6', 'checks whether 153 is an Armstrong number',
    'checks whether 49 is a perfect square', 'prints the odd numbers from 15 down to 1', 'converts 26 miles to kilometers',
    'works out the speed of a car that drives 150 km in 2.5 hours', 'works out a 20% tip on a 45.50 bill', 'prints a 3 by 4 grid of the number 7',
    'prints a pyramid of numbers with 4 rows', 'prints whether -5 is positive, negative or zero', 'prints the season for month number 7',
    'prints the next even number after 13', 'adds up the even digits of 2846', 'counts how many times 64 can be halved before it reaches 1',
    'prints the average of the numbers from 1 to 9', 'multiplies 6 by 7 using only addition', 'prints 17 divided by 5 with its remainder',
    'prints the numbers from 1 to 6 on one line', 'counts down from 50 by fives', 'prints the value of 2x squared plus 3x plus 1 when x is 4',
    'converts 25 degrees Celsius to Kelvin', 'works out the pay for 45 hours at 12.0 an hour with time and a half after 40 hours',
    'prints the larger of the text lengths of "apple" and "kiwi"', 'prints "yes" if a stored name equals "Ana" and "no" otherwise',
    'has a function that says hello to a name, and calls it twice', 'prints how many days are in 3 weeks and 4 days',
    # Lists, text, your own types and asking. Added when Kel grew them (2026-11-04).
    'finds the second largest number in the list [4, 9, 2, 7]', 'adds the lists [1, 2, 3] and [4, 5, 6] item by item into a new list',
    'keeps only the words longer than 3 letters from a list of words', 'moves the first item of a list to the end',
    'checks whether the word "level" reads the same backwards', 'reverses the word "kelus"', 'counts the words in "the cat sat on the mat"',
    'makes the first letter of "kel" a capital', 'removes the spaces from "a b c"', 'prints a list of the lengths of three words',
    'has a type Rectangle with a width and a height and a method that gives its area', 'has a type Book with a title and a number of pages, and prints whether a book is long',
    'has a type Dog with a name and a method speak that prints a bark with its name', 'keeps a list of three Book values with pages and adds up the pages',
    'has a type Counter and a method that returns a new Counter counted up by one', 'asks for a number and prints its square',
    'asks for a name and prints it in capital letters', 'asks for a word until the user types "stop"', 'keeps asking for numbers until 0 and prints their total',
    'rounds 3.7 and prints the square root of 81.0',
]


def prompts():
    return [ASKS[i % len(ASKS)].format(task) for i, task in enumerate(TASKS)]


def compile_only(programs):
    """{id: (ok, error)} from the real compiler, which does not run them."""
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as handle:
        json.dump([{'id': i, 'code': code} for i, code in enumerate(programs)], handle)
    try:
        out = subprocess.run(['node', str(ROOT / 'data/check_lus.mjs'), handle.name], capture_output=True, text=True, timeout=120, check=True).stdout
    finally:
        Path(handle.name).unlink(missing_ok=True)
    return {r['id']: (r['ok'], r.get('error')) for r in json.loads(out)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--max-tokens', type=int, default=500)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    model, header, _ = Model.load(args.checkpoint)
    started, results = time.monotonic(), []
    for prompt in prompts():
        reply = ''.join(model.generate([{'role': 'user', 'content': prompt}], max_tokens=args.max_tokens, temperature=0)).strip()
        blocks = FENCE.findall(reply)
        code = next((c for lang, c in blocks if lang.lower() in ('lus', 'kel')), blocks[0][1] if blocks else reply)
        results.append({'prompt': prompt, 'reply': reply, 'code': code, 'labelled': any(lang.lower() in ('lus', 'kel') for lang, _ in blocks)})
    checked = compile_only([r['code'] for r in results])
    for i, r in enumerate(results):
        r['compiles'], r['error'] = checked[i]
        r['passed'] = bool(r['labelled'] and r['compiles'])
    count = lambda name: f"{sum(r[name] for r in results)}/{len(results)}"
    report = {'checkpoint': args.checkpoint, 'steps': header.get('metadata', {}).get('steps'), 'passed': count('passed'), 'labelled_lus': count('labelled'),
              'compiles': count('compiles'), 'behavior_tested': False, 'seconds': round(time.monotonic() - started, 1), 'results': results}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('results', 'checkpoint')}))


if __name__ == '__main__':
    main()
