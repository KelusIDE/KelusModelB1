"""Score deciding to look up places: "where should I go" questions should get "PLACES: <what> @ <where>".

    python3 -m kelus.places_eval --checkpoint checkpoints/kani-otter-4.npz --out reports/places.json

Positives are the held-out place questions in data/places-v2.jsonl: each has a thing, a phrasing or a place
never trained on. Contrasts are held-out questions that mention places but want a fact (a search), code or
chat, and messages that need a chat reply ("I need a pep talk"). Other negatives are held-out chat, feelings and
practiced conversations from instruct-v9, and fact questions (decide), which must still search. A places call on
any of those is a false places call.
"""
import argparse
import json
from pathlib import Path
import time

from .model import Model

OTHER = ('chat_pairs', 'feelings', 'practiced', 'decide')


def parse(reply):
    """(what, where) of a places call, or None."""
    body = reply[len('PLACES: '):] if reply.startswith('PLACES: ') else ''
    if ' @ ' not in body:
        return None
    what, where = body.rsplit(' @ ', 1)
    return what.strip().lower(), where.strip().lower()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--places', default='data/places-v2.jsonl')
    parser.add_argument('--data', default='data/instruct-v9.jsonl')
    parser.add_argument('--limit', type=int, default=400)
    parser.add_argument('--others', type=int, default=200)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    model, header, _ = Model.load(args.checkpoint)
    load = lambda path: [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    places = [r for r in load(args.places) if r['split'] == 'validation']
    positives = [r for r in places if r['family'] == 'places'][:args.limit]
    contrasts = [r for r in places if r['family'] != 'places']
    others = [r for r in load(args.data) if r['split'] == 'validation' and r.get('family') in OTHER]
    others = [r for family in OTHER for r in [o for o in others if o['family'] == family][:args.others // len(OTHER)]]
    results, started = [], time.monotonic()
    for row in positives + contrasts + others:
        reply = ''.join(model.generate(row['messages'][:-1], max_tokens=48 if row['family'] == 'places' else 80, temperature=0)).strip()
        reference = row['messages'][-1]['content']
        got, wanted = parse(reply), parse(reference)
        results.append({'family': row['family'], 'prompt': row['messages'][-2]['content'], 'reply': reply[:300], 'reference': reference[:300],
                        'called': got is not None, 'exact': reply == reference, 'where': bool(got and wanted and got[1] == wanted[1]),
                        'searched': reply.startswith('SEARCH: ')})
    pos = [r for r in results if r['family'] == 'places']
    con = [r for r in results if r['family'].startswith('places_')]
    oth = [r for r in results if r['family'] in OTHER]
    facts = [r for r in con + oth if r['reference'].startswith('SEARCH: ')]
    report = {'checkpoint': args.checkpoint, 'steps': header.get('metadata', {}).get('steps'),
              'called_when_needed': f"{sum(r['called'] for r in pos)}/{len(pos)}", 'exact_call': f"{sum(r['exact'] for r in pos)}/{len(pos)}",
              'right_where': f"{sum(r['where'] for r in pos)}/{len(pos)}",
              'false_places': f"{sum(r['called'] for r in con + oth)}/{len(con) + len(oth)}",
              'facts_still_searched': f"{sum(r['searched'] for r in facts)}/{len(facts)}",
              'seconds': round(time.monotonic() - started, 1), 'results': results}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('results', 'checkpoint')}))


if __name__ == '__main__':
    main()
