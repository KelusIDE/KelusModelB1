"""Score reading: answers to questions about held-out Wikipedia articles, from the passages shown.

    python3 -m kelus.reading_eval --checkpoint checkpoints/kani-b2-2f-v5.npz --out reports/reading.json

Validation records come from articles never used in training. For each one the model sees
the search results and question (kelus.reading.prompt) and its reply is checked:
  exact      the reply equals the reference (copied sentence or date, and citation)
  cited      an answer ends with the citation of the passage that answers it
  grounded   the answer's text (citation removed) appears in the passages shown
  no_answer  questions the passages don't answer get kelus.reading.NO_ANSWER
"""
import argparse
import json
from pathlib import Path
import re
import time

from .model import Model
from .reading import NO_ANSWER


def score(row, reply):
    reference = row['accept'][0]
    shown = row['messages'][0]['content']
    if row['kind'] == 'no_answer':
        return {'exact': reply == reference, 'no_answer': reply == NO_ANSWER}
    cite = re.search(r'\[\d+\]$', reference).group(0)
    text = re.sub(r'\s*\[\d+\]$', '', reply).rstrip('.')
    dated = re.match(r'.+ (?:was born|died) (?:on|in) (.+)$', text)
    return {'exact': reply == reference, 'cited': reply.endswith(cite), 'refused': reply == NO_ANSWER,
            'grounded': bool(text) and (text in shown or bool(dated and dated.group(1) in shown))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', default='data/reading-v1.jsonl')
    parser.add_argument('--limit', type=int, default=400)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    model, header, _ = Model.load(args.checkpoint)
    rows = [json.loads(line) for line in Path(args.data).read_text(encoding='utf-8').splitlines() if line.strip()]
    rows = [r for r in rows if r['split'] == 'validation' and r.get('family') == 'reading'][:args.limit]
    results, started = [], time.monotonic()
    for row in rows:
        reply = ''.join(model.generate(row['messages'][:1], max_tokens=200, temperature=0)).strip()
        results.append({'kind': row['kind'], 'title': row['title'], 'question': row['messages'][0]['content'].rsplit('Question: ', 1)[1],
                        'reply': reply, 'reference': row['accept'][0], **score(row, reply)})
    answers = [r for r in results if r['kind'] == 'answer']
    unanswerable = [r for r in results if r['kind'] == 'no_answer']
    rate = lambda items, key: f"{sum(r[key] for r in items)}/{len(items)}"
    report = {'checkpoint': args.checkpoint, 'steps': header.get('metadata', {}).get('steps'), 'total': len(results),
              'exact': rate(results, 'exact'), 'cited': rate(answers, 'cited'), 'grounded': rate(answers, 'grounded'),
              'refused_answerable': rate(answers, 'refused'), 'no_answer': rate(unanswerable, 'no_answer'),
              'seconds': round(time.monotonic() - started, 1),
              'scope': 'Held-out Wikipedia articles; pattern-made questions; the answer must come from the passages shown.',
              'results': results}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('results', 'scope', 'checkpoint')}))


if __name__ == '__main__':
    main()
