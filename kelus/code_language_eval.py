"""Does Cide answer coding requests in the language asked for? Held-out prompts from data/cide-v6.jsonl, per language.

    python3 -m kelus.code_language_eval --checkpoint checkpoints/cide-2.npz --out reports/code-languages.json

For each of the 19 languages: replies with a code block, replies whose first block is labeled with that language
(```go for Go ...), and, for Python, blocks that parse (ast.parse; generated code is never run). Held-out prompts come
from the validation split, so they were not trained on. Replies are kept for reading; parsing is not correctness.
"""
import argparse
import ast
import collections
import json
from pathlib import Path
import re
import time

from .model import Model

FENCE = re.compile(r'```\s*([A-Za-z+#]*)\s*\n(.*?)(?:```|$)', re.S)
ALIASES = {'c++': 'cpp', 'c#': 'csharp', 'cs': 'csharp', 'js': 'javascript', 'jsx': 'javascript', 'ts': 'typescript',
           'sh': 'shell', 'bash': 'shell', 'zsh': 'shell', 'py': 'python', 'python3': 'python', 'golang': 'go',
           'rb': 'ruby', 'rs': 'rust', 'mysql': 'sql', 'postgresql': 'sql', 'sqlite': 'sql'}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', default='data/cide-v6.jsonl')
    parser.add_argument('--per-language', type=int, default=8)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    model, header, _ = Model.load(args.checkpoint)
    rows = collections.defaultdict(list)
    for line in Path(args.data).read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if row['split'] == 'validation' and row.get('language') and len(rows[row['language']]) < args.per_language:
            rows[row['language']].append(row)
    results, started = [], time.monotonic()
    for lang, chosen in sorted(rows.items()):
        for row in chosen:
            try:
                reply = ''.join(model.generate(row['messages'][:1], max_tokens=300, temperature=0)).strip()
            except ValueError:
                continue
            block = FENCE.search(reply)
            label = ALIASES.get(block.group(1).lower(), block.group(1).lower()) if block else None
            parses = None
            if lang == 'python' and block:
                try:
                    ast.parse(block.group(2))
                    parses = True
                except SyntaxError:
                    parses = False
            results.append({'language': lang, 'prompt': row['messages'][0]['content'][:300], 'reply': reply[:1200],
                            'code': bool(block), 'right_language': label == lang, 'python_parses': parses})
    by = collections.defaultdict(list)
    for r in results:
        by[r['language']].append(r)
    languages = {lang: f"{sum(r['right_language'] for r in rs)}/{len(rs)}" for lang, rs in sorted(by.items())}
    python = [r['python_parses'] for r in results if r['python_parses'] is not None]
    report = {'checkpoint': args.checkpoint, 'steps': header.get('metadata', {}).get('steps'),
              'code_blocks': f"{sum(r['code'] for r in results)}/{len(results)}",
              'right_language': f"{sum(r['right_language'] for r in results)}/{len(results)}",
              'python_parses': f"{sum(python)}/{len(python)}", 'by_language': languages,
              'seconds': round(time.monotonic() - started, 1), 'results': results}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('results', 'checkpoint')}))


if __name__ == '__main__':
    main()
