"""Score deciding to search: fact questions should get "SEARCH: <query>", everything else a direct reply.

    python3 -m kelus.decide_eval --checkpoint checkpoints/kani-b2-2f-v6.npz --out reports/decide.json

Positives are the held-out "decide" questions in data/instruct-v6.jsonl (questions about articles
never trained on). Negatives are held-out chat, feelings and practiced-concept conversations whose
reference reply is not a search; a search there is a false search.
"""
import argparse
import json
from pathlib import Path
import time

from .model import Model

NEGATIVE = ('chat_pairs', 'feelings', 'practiced')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', default='data/instruct-v6.jsonl')
    parser.add_argument('--limit', type=int, default=300)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    model, header, _ = Model.load(args.checkpoint)
    rows = [json.loads(line) for line in Path(args.data).read_text(encoding='utf-8').splitlines() if line.strip()]
    rows = [r for r in rows if r['split'] == 'validation']
    positives = [r for r in rows if r.get('family') == 'decide'][:args.limit]
    negatives = [r for r in rows if r.get('family') in NEGATIVE and not r['messages'][-1]['content'].startswith('SEARCH: ')][:args.limit]
    results, started = [], time.monotonic()
    for row in positives + negatives:
        reply = ''.join(model.generate(row['messages'][:-1], max_tokens=80, temperature=0)).strip()
        reference = row['messages'][-1]['content']
        results.append({'family': row['family'], 'prompt': row['messages'][-2]['content'], 'reply': reply[:300], 'reference': reference[:300],
                        'searched': reply.startswith('SEARCH: '), 'exact': reply == reference})
    pos = [r for r in results if r['family'] == 'decide']
    neg = [r for r in results if r['family'] != 'decide']
    report = {'checkpoint': args.checkpoint, 'steps': header.get('metadata', {}).get('steps'),
              'searched_when_needed': f"{sum(r['searched'] for r in pos)}/{len(pos)}", 'exact_query': f"{sum(r['exact'] for r in pos)}/{len(pos)}",
              'false_searches': f"{sum(r['searched'] for r in neg)}/{len(neg)}", 'seconds': round(time.monotonic() - started, 1), 'results': results}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('results', 'checkpoint')}))


if __name__ == '__main__':
    main()
