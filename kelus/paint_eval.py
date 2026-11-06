"""Score Kani Painter on held-out drawing requests (data/paint-v2.jsonl validation).

Validation scenes combine settings, times and weather that never appear together
in training. For each request the painter writes a program; this checks that it
renders through the safe interpreter and that the picture contains what was asked:
the setting, time of day, weather, requested object counts and colors. Checks read
the program's structure (the generator's distinctive calls), not pixels, so they
measure following the request, not artistic quality. Renders are saved for review.

    python3 -m kelus.paint_eval --checkpoint checkpoints/kani-painter-v1.npz --limit 60 --out reports/painter
"""
import argparse
import html
import json
from pathlib import Path
import re
import time

from .model import Model
from .paint import PaintError, code_block, render

SETTING = {'ocean': r"gradient\(0, 260, 640, 140", 'lake': r"mountains\(262, 120", 'mountains': r"mountains\(240, 150",
           'desert': r"'sand', 'tan'\)", 'forest': r"mountains\(250, 110", 'city': r"rect\(i \* 30 - 10",
           'meadow': r"roughness=0\.25", 'snow': r"gradient\(0, 290, 640, 110, 'snow'", 'space': r"'#02010a', '#120a2e'"}
WEATHER = {'cloudy': r"'white/90'\)", 'rainy': r"'sky/50'\)", 'snowing': r"rand\(0\.8, 2\.2\), 'snow/90'\)", 'foggy': r"'silver/0'"}
TIME_SKY = {'day': ('#3b82c4', '#4a90d9'), 'sunset': ('#1b2a5a', '#2b1b4a'), 'dawn': ('#3a4a7a', '#4b5a8f'), 'night': ('#04071a', '#02010a')}
OBJECT = {'sailboat': r"'charcoal', 3\)", 'lighthouse': r"'gold'\)", 'pine': r"'forest'\)", 'tree': r"'green'\)\n.*'forest'\)",
          'house': r"'crimson'\)", 'cactus': r"rect\(x \+ \d+, y - \d+, \d+, \d+, 'forest'\)", 'pyramid': r"'sand'\)\n.*'tan'\)",
          'snowman': r"'orange'\)$", 'balloon': r"'white/35'\)", 'tent': r"'orange'\)\n.*'dark_brown'\)", 'rocket': r"'sky'\)$",
          'planet': r"glow\(x, y, \d+, '\w+/30'\)", 'person': r"circle\(x, y - \d+, \d+, '#(?:ffdbac|f1c27d|e0ac69|c68642|8d5524)'\)",
          'cat': r"'lime'\)\n.*'lime'\)\n.*'pink'\)", 'dog': r"'black'\)\n.*'dark_brown'\)\n.*'black'\)",
          'car': r"'charcoal'\)\n.*'silver'\)\n.*'charcoal'\)\n.*'silver'\)", 'flower': r"circle\(x, y - \d+, \d+, 'gold'\)",
          'bird': r"line\(x - \d+, y - \d+, x, y, 'charcoal', \d+\)", 'castle': r"rect\(x - \d+, y - \d+, \d+, \d+, 'silver'\)\n.*'gray'\)",

          'apple': r"'white/30'\)", 'pear': r"'white/25'\)", 'banana': r"'cream/60'\)", 'orange': r"'white/22'\)", 'lemon': r"'cream/50'\)",
          'strawberry': r"'white/15'\)", 'cherries': r"'white/40'\)", 'grapes': r"'white/45'\)", 'watermelon': r"'white/10'\)"}
TABLE = r"rect\(0, 290, 640, 110, '\w+'\)"


def check(program, scene):
    """Which parts of the request the program contains."""
    found = {}
    if 'setting' in scene:
        found['setting'] = bool(re.search(SETTING[scene['setting']], program))
        if scene['setting'] != 'space':
            sky = re.search(r"gradient\(0, 0, 640, 400, '(#[0-9a-f]{6})'", program)
            if scene['weather'] in ('rainy', 'foggy'):
                found['time'] = bool(sky)
            else:
                found['time'] = bool(sky) and sky[1] in TIME_SKY[scene['time']]
            expected = scene['weather']
            present = {w for w, pattern in WEATHER.items() if re.search(pattern, program, re.M)}
            found['weather'] = present == ({expected} if expected != 'clear' else set())
        objects = scene['objects']
    else:
        if scene.get('anime'):

            found['portrait'] = 'face(x, y' in program and 'anime_eye(' in program
            for key in ('hair', 'eyes'):
                if scene.get(key):
                    found[key] = f"'{scene[key]}'" in program
            if scene.get('extra'):
                found['extra'] = ('cat_ears(' if scene['extra'] == 'cat ears' else 'hair_bow(') in program
            return found
        objects = scene['objects'] if 'objects' in scene else [[scene['object'], 1, scene['color']]]
        if scene.get('still_life'):
            found['setting'] = bool(re.search(TABLE, program))
    for name, count, color in objects:
        ok = bool(re.search(OBJECT[name], program, re.M))
        if count > 1:
            ok = ok and f'range({count})' in program
        if color:
            ok = ok and f"'{color}'" in program
        found[f'object:{name}'] = ok
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', default='data/paint-v4.jsonl')
    parser.add_argument('--limit', type=int, default=60)
    parser.add_argument('--out', type=Path, help='directory for renders, contact sheet and report.json')
    args = parser.parse_args()
    model, header, _ = Model.load(args.checkpoint)
    rows = [json.loads(line) for line in Path(args.data).read_text(encoding='utf-8').splitlines() if line.strip()]
    rows = [r for r in rows if r['split'] == 'validation'][:args.limit]
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
    results, started = [], time.monotonic()
    for index, row in enumerate(rows):
        request = row['messages'][0]['content']

        reply = ''.join(model.generate([{'role': 'user', 'content': request}], max_tokens=900, temperature=0, prefix='```python\n'))
        program = code_block(reply + ('' if reply.rstrip().endswith('```') else '\n```'))
        error = None
        try:
            image = render(program, seed=index)
            if args.out:
                image.save(args.out / f'{index:03d}.png')
        except PaintError as problem:
            error = str(problem)
        found = check(program, row['scene']) if error is None else {}
        results.append({'request': request, 'scene': row['scene'], 'renders': error is None, 'error': error, 'checks': found,
                        'follows_request': error is None and all(found.values()), 'program': program})
    total = len(results)
    parts = {}
    for r in results:
        for key, ok in r['checks'].items():
            kind = key.split(':')[0]
            parts.setdefault(kind, []).append(ok)
    report = {'checkpoint': args.checkpoint, 'steps': header.get('metadata', {}).get('steps'), 'total': total,
              'renders': sum(r['renders'] for r in results), 'follows_request': sum(r['follows_request'] for r in results),
              'parts': {k: f'{sum(v)}/{len(v)}' for k, v in sorted(parts.items())}, 'seconds': round(time.monotonic() - started, 1),
              'scope': 'Held-out scene combinations; structural checks of the generated program, not image quality.', 'results': results}
    if args.out:
        (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        cells = ''.join(f'<figure><img src="{i:03d}.png" width="320"><figcaption>{"✓" if r["follows_request"] else "✗"} {html.escape(r["request"])}'
                        f'{"<br><b>" + html.escape(r["error"]) + "</b>" if r["error"] else ""}</figcaption></figure>' for i, r in enumerate(results))
        (args.out / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Kani Painter</title><style>body{background:#111;color:#ddd;'
                                             'font:12px sans-serif;display:flex;flex-wrap:wrap;gap:8px}figure{margin:0;width:320px}</style>' + cells)
    print(json.dumps({k: report[k] for k in ('steps', 'total', 'renders', 'follows_request', 'parts', 'seconds')}))


if __name__ == '__main__':
    main()
