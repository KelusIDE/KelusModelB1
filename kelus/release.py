"""Release the best tested snapshot of a training run as a named Kani build.

    python -m kelus.release --eval training-runs/kani-big-chat-v1-eval --name "Kani Otter 4" \\
        --codename "Kani B3:88M" --checkpoint kani-otter-4 --previous-id seal-3 --previous-label "Kani Seal 3 · v7"

Every snapshot sN.npz in --eval with reading, decide and benchmark results is scored. Gates (all must hold):
multi-turn context is perfect, no more false searches and no fewer needed searches than the baseline (small
slack), and "the results don't say" stays as reliable. Among those, the highest reading exact + cited + grounded
+ benchmark wins, and it must beat the baseline (the current latest build). Nothing is released otherwise.
Snapshots with places results (kelus.places_eval) also add their exact places calls to the score, and must call
places for at least 90% of the held-out place questions, make at most 3 false places calls, and keep reading
and the benchmark close to the baseline (a new skill may not cost an old one). Everyday results (kelus.everyday_eval)
add their recipe and learning questions sent to search and exact phrase translations to the score; food chat must
stay chat (at least 80%).

A release writes checkpoints/<checkpoint>.npz (weights only, display name and codename in its metadata), exports
it for the terminal app (training-runs/build-exports/<checkpoint>) and makes it the "latest" build in
kani-builds.json; the previous latest stays available under --previous-id. The website is not changed.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def fraction(text):
    passed, total = (int(n) for n in str(text).split('/'))
    return passed, total


def results(directory, snapshot):
    """Scores of one tested snapshot, or None when a test has not finished."""
    files = {kind: directory / f'{kind}-{snapshot}.json' for kind in ('reading', 'decide', 'bench')}
    if not all(path.exists() for path in files.values()):
        return None
    reading, decide, bench = (json.loads(path.read_text()) for path in files.values())
    places = directory / f'places-{snapshot}.json'
    places = json.loads(places.read_text()) if places.exists() else None
    everyday = directory / f'everyday-{snapshot}.json'
    everyday = json.loads(everyday.read_text()) if everyday.exists() else None
    return {'snapshot': snapshot, 'steps': bench.get('steps'),
            **({'places_called': fraction(places['called_when_needed'])[0], 'places_total': fraction(places['called_when_needed'])[1],
                'places_exact': fraction(places['exact_call'])[0], 'places_where': fraction(places['right_where'])[0],
                'false_places': fraction(places['false_places'])[0]} if places else {}),
            **({'recipes_searched': fraction(everyday['recipes_searched'])[0], 'learning_searched': fraction(everyday['learning_searched'])[0],
                'food_chat_kept': fraction(everyday['food_chat_kept'])[0], 'food_chat_total': fraction(everyday['food_chat_kept'])[1],
                'phrases_exact': fraction(everyday['phrases_exact'])[0]} if everyday else {}),
            'exact': fraction(reading['exact'])[0], 'cited': fraction(reading['cited'])[0],
            'grounded': fraction(reading['grounded'])[0], 'no_answer': fraction(reading['no_answer'])[0],
            'searched': fraction(decide['searched_when_needed'])[0], 'exact_query': fraction(decide['exact_query'])[0],
            'false_searches': fraction(decide['false_searches'])[0], 'benchmark': bench['overall']['passed'],
            'context': bench['context_regression']['passed'], 'context_total': bench['context_regression']['total']}


def score(row):
    return (row['exact'] + row['cited'] + row['grounded'] + row['benchmark'] + row.get('places_exact', 0)
            + row.get('recipes_searched', 0) + row.get('learning_searched', 0) + row.get('phrases_exact', 0))


def gates(row, baseline):
    checks = {'context perfect': row['context'] == row['context_total'],
              'false searches': row['false_searches'] <= baseline['false_searches'] + 2,
              'searches when needed': row['searched'] >= baseline['searched'] - 6,
              "says when results don't answer": row['no_answer'] >= baseline['no_answer'] - 2}
    if 'places_called' in row:
        checks.update({'places when asked': row['places_called'] >= 0.9 * row['places_total'], 'false places': row['false_places'] <= 3,
                       'keeps reading': all(row[k] >= baseline[k] - 12 for k in ('exact', 'cited', 'grounded')),
                       'keeps benchmark': row['benchmark'] >= baseline['benchmark'] - 5})
    if 'food_chat_kept' in row:
        checks['food chat stays chat'] = row['food_chat_kept'] >= 0.8 * row['food_chat_total']
    return checks


def choose(eval_dir, baseline):
    rows = []
    for path in sorted(eval_dir.glob('s*.npz'), key=lambda p: int(p.stem[1:])):
        row = results(eval_dir, path.stem)
        if row:
            row['gates'] = gates(row, baseline)
            row['score'] = score(row)
            rows.append(row)
    passing = [row for row in rows if all(row['gates'].values())]
    best = max(passing, key=lambda row: (row['score'], row['steps'] or 0), default=None)
    return rows, best


def release(args, best, baseline):
    source = args.eval / f"{best['snapshot']}.npz"
    destination = ROOT / 'checkpoints' / f'{args.checkpoint}.npz'
    with np.load(source) as archive:
        values = {name: archive[name] for name in archive.files if name.startswith('parameter.')}
        header = json.loads(str(archive['header']))
    header.pop('optimizer', None)
    header['metadata'] = {**header.get('metadata', {}), 'display_name': args.name, 'codename': args.codename,
                          'release': {'released_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'source': str(source.relative_to(ROOT)),
                                      'scores': {k: best[k] for k in ('exact', 'cited', 'grounded', 'no_answer', 'searched', 'exact_query', 'false_searches', 'benchmark', 'context',
                                                                      'places_called', 'places_exact', 'places_where', 'false_places',
                                                                      'recipes_searched', 'learning_searched', 'food_chat_kept', 'phrases_exact') if k in best},
                                      'baseline': args.baseline_name, 'baseline_scores': {k: baseline[k] for k in ('exact', 'cited', 'grounded', 'benchmark')}}}
    temporary = destination.with_suffix('.tmp.npz')
    np.savez(temporary, header=np.array(json.dumps(header)), **values)
    temporary.replace(destination)
    export = ROOT / 'training-runs' / 'build-exports' / args.checkpoint
    subprocess.run([sys.executable, '-m', 'kelus.export_web', '--checkpoint', str(destination), '--out', str(export)], cwd=ROOT, check=True,
                   capture_output=True)

    builds_path = ROOT / 'kani-builds.json'
    builds = json.loads(builds_path.read_text())
    latest = next(build for build in builds if build['id'] == 'latest')
    previous = {**latest, 'id': args.previous_id, 'label': args.previous_label}
    old_checkpoint = latest['checkpoint']
    builds = [build for build in builds if build['id'] != args.previous_id]
    for build in builds:
        if build['id'] in ('latest', 'pixels') and build['checkpoint'] == old_checkpoint:
            build.update(checkpoint=args.checkpoint, export=str(export.relative_to(ROOT)),
                         label=f"{args.name} · latest" + (' + Pixels (experimental)' if build['id'] == 'pixels' else ''))
    builds.insert(next(i for i, build in enumerate(builds) if build['id'] == 'pixels') + 1, previous)
    builds_path.write_text('[\n' + ',\n'.join('  ' + json.dumps(build, ensure_ascii=False, separators=(',', ':')) for build in builds) + '\n]\n')
    return destination, export


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--eval', type=Path, required=True, help='folder with sN.npz snapshots and their test results')
    parser.add_argument('--baseline', type=Path, default=ROOT / 'training-runs/kani-b2-2f-v7-eval', help='test results of the current latest build')
    parser.add_argument('--baseline-snapshot', default='s10000')
    parser.add_argument('--baseline-name', default='Kani Seal 3 (v7)')
    parser.add_argument('--name', required=True)
    parser.add_argument('--codename', required=True)
    parser.add_argument('--checkpoint', required=True, help='name in checkpoints/ for the release')
    parser.add_argument('--previous-id', required=True, help='build id that keeps the current latest model available')
    parser.add_argument('--previous-label', required=True)
    parser.add_argument('--dry-run', action='store_true', help='only print the scores and the choice')
    args = parser.parse_args()
    args.eval = args.eval.resolve()
    baseline = results(args.baseline, args.baseline_snapshot)
    if baseline is None:
        sys.exit(f'No complete test results for {args.baseline_snapshot} in {args.baseline}')
    rows, best = choose(args.eval, baseline)
    print(f"baseline {args.baseline_name}: score {score(baseline)} (exact {baseline['exact']}, cited {baseline['cited']}, grounded {baseline['grounded']}, benchmark {baseline['benchmark']})")
    for row in rows:
        failed = [name for name, ok in row['gates'].items() if not ok]
        print(f"{row['snapshot']:>7}: score {row['score']} (exact {row['exact']}, cited {row['cited']}, grounded {row['grounded']}, benchmark {row['benchmark']}, "
              f"context {row['context']}/{row['context_total']}, searched {row['searched']}, false {row['false_searches']}"
              + (f", places {row['places_called']}/{row['places_total']} exact {row['places_exact']} where {row['places_where']} false {row['false_places']}" if 'places_called' in row else '')
              + (f", recipes searched {row['recipes_searched']} learning searched {row['learning_searched']} food chat {row['food_chat_kept']}/{row['food_chat_total']} phrases {row['phrases_exact']}"
                 if 'recipes_searched' in row else '')
              + ")" + (f"  fails: {', '.join(failed)}" if failed else ''))
    if best is None or best['score'] <= score(baseline):
        sys.exit('No snapshot passes the gates and beats the baseline; nothing released.')
    print(f"best: {best['snapshot']} (score {best['score']} vs {score(baseline)})")
    if args.dry_run:
        return
    destination, export = release(args, best, baseline)
    print(json.dumps({'released': args.name, 'checkpoint': str(destination.relative_to(ROOT)), 'export': str(export.relative_to(ROOT)),
                      'from': best['snapshot']}))


if __name__ == '__main__':
    main()
