"""Does Cide write well-formed Python, JavaScript and HTML? Held-out requests from data/cide-v13.jsonl, per part.

    python3 -m kelus.code_quality_eval --checkpoint checkpoints/cide-4.npz --out reports/code-quality.json

Each reply is checked without running it: a code block in the right language; Python parses (ast.parse), and when the
request names the function ("a function `add`") the code defines it; JavaScript parses (`node --check` reads it; nothing
executes); HTML tags balance. Replies get more room than kelus.code_language_eval's 300 tokens, so a cut-off answer
is not mistaken for broken code. Parsing is not correctness: only running the code against tests can show that.
"""
import argparse
import ast
import collections
import json
from pathlib import Path
import re
import time

from .model import Model
from .website_eval import Structure, script_parses

FENCE = re.compile(r'```\s*([A-Za-z+#]*)[^\n]*\n(.*?)(?:```|$)', re.S)
ALIASES = {'js': 'javascript', 'jsx': 'javascript', 'py': 'python', 'python3': 'python', 'htm': 'html'}
PARTS = ('python-self-oss', 'python-magicoder', 'javascript', 'html')
NAMED = re.compile(r'\bfunction\s+(?:named\s+|called\s+)?`([A-Za-z_][A-Za-z0-9_]*)')


def judge(prompt, reply, language):
    blocks = [(ALIASES.get(lang.lower(), lang.lower()), code) for lang, code in FENCE.findall(reply)]
    code = '\n'.join(c for lang, c in blocks if lang == language)
    checks = {'right_language': bool(code.strip())}
    if language == 'python':
        try:
            tree = ast.parse(code)
            checks['parses'] = bool(code.strip())
        except (SyntaxError, ValueError):
            tree, checks['parses'] = None, False
        called = NAMED.search(prompt)
        if called:
            defined = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))} if tree else set()
            checks['defines_function'] = called.group(1) in defined
    elif language == 'javascript':
        checks['parses'] = bool(code.strip()) and script_parses(code) is True
    else:
        parser = Structure()
        parser.feed(code)
        checks['parses'] = bool(code.strip()) and parser.balanced and not parser.open
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', default='data/cide-v13.jsonl')
    parser.add_argument('--per-part', type=int, default=30)
    parser.add_argument('--max-tokens', type=int, default=800)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    model, header, _ = Model.load(args.checkpoint)
    rows = collections.defaultdict(list)
    for line in Path(args.data).read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if row['split'] == 'validation' and row.get('part') in PARTS and len(rows[row['part']]) < args.per_part:
            rows[row['part']].append(row)
    results, started = [], time.monotonic()
    for part in PARTS:
        for row in rows[part]:
            prompt = row['messages'][0]['content']
            try:
                reply = ''.join(model.generate(row['messages'][:1], max_tokens=args.max_tokens, temperature=0)).strip()
            except ValueError:
                continue
            checks = judge(prompt, reply, row['language'])
            results.append({'part': part, 'prompt': prompt[:300], 'reply': reply[:1500], 'checks': checks, 'passed': all(checks.values())})
    by = collections.defaultdict(list)
    for r in results:
        by[r['part']].append(r)
    report = {'checkpoint': args.checkpoint, 'steps': header.get('metadata', {}).get('steps'),
              'passed': f"{sum(r['passed'] for r in results)}/{len(results)}",
              'by_part': {part: f"{sum(r['passed'] for r in rs)}/{len(rs)}" for part, rs in by.items()},
              'by_check': {name: f"{sum(r['checks'].get(name, False) for r in results)}/{sum(name in r['checks'] for r in results)}"
                           for name in ('right_language', 'parses', 'defines_function')},
              'behavior_tested': False, 'seconds': round(time.monotonic() - started, 1), 'results': results}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('results', 'checkpoint')}))


if __name__ == '__main__':
    main()
