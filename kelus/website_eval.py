"""Can Cide build a complete, working single-file website? Held-out requests from data/websites-v1.jsonl
(or --data data/web-basics-v1.jsonl: one plain request per page kind, in wordings never trained on; or
--data data/cide-followups-v1.jsonl: a request later in a chat, after a page, sent with the whole chat as the site does).

    python3 -m kelus.website_eval --checkpoint checkpoints/cide-2.npz --out reports/websites.json

The requests name businesses whose themes were never trained on (data/make_websites.py HELD_OUT_THEMES) and ask for one
or two interactive features. Each reply's page is checked without running it: a complete document with balanced tags,
CSS and a script, JavaScript that parses (`node --check` reads it; nothing executes), every element the script looks up
by id present, each requested feature's controls present and wired, and for an edit ("make the buttons red") the
change it asked for (`expect` in the row). Pages are saved beside the report so a person
can open them; only a browser test proves behavior, and generated code is never run here.
"""
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time

from .model import Model


FEATURE_IDS = {
    'counter': ['count', 'increase', 'decrease', 'reset'], 'todo': ['todo-form', 'todo-input', 'todo-list'],
    'tip': ['bill', 'tip-percent', 'people', 'tip-amount', 'per-person'], 'converter': ['first-unit', 'second-unit'],
    'quiz': ['question', 'answers', 'quiz-result'], 'stopwatch': ['time', 'start', 'stop', 'clear'],
    'contact': ['contact-form', 'contact-name', 'contact-email', 'contact-message', 'form-success'],
    'dark_mode': ['theme-toggle'], 'faq': [], 'tabs': ['about', 'prices', 'location'], 'modal': ['open-modal', 'modal', 'close-modal'],
    'slideshow': ['slide', 'caption', 'previous', 'next'], 'quotes': ['quote', 'new-quote'], 'guess': ['guess-form', 'guess-input', 'guess-hint'],
    'cart': ['cart', 'cart-total'], 'filter_list': ['search', 'items'], 'char_count': ['note', 'chars-left'],
    # data/make_web_basics.py pages
    'login': ['login-form', 'login-user', 'login-password', 'login-error', 'login-success'],
    'signup': ['signup-form', 'signup-name', 'signup-email', 'signup-password', 'signup-confirm', 'signup-error', 'signup-success'],
    'calculator': ['display', 'keys'], 'countdown': ['timer-minutes', 'timer-display', 'timer-start', 'timer-pause', 'timer-reset'],
    'landing': ['features'], 'navbar': ['menu-toggle', 'menu'], 'button': ['message-button', 'message'],
}
# plain pages from data/make_web_basics.py that need no script: the text each must contain instead
STATIC = {'hello': 'hello', 'template': '<main', 'table': '<table', 'profile': '<section'}
VOID = {'meta', 'input', 'br', 'img', 'link', 'hr', 'source'}


class Structure(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.tags, self.open, self.balanced = set(), set(), [], True

    def handle_starttag(self, tag, attrs):
        self.tags.add(tag)
        self.ids.update(value for name, value in attrs if name == 'id' and value)
        if tag not in VOID:
            self.open.append(tag)

    def handle_endtag(self, tag):
        if self.open and self.open[-1] == tag:
            self.open.pop()
        else:
            self.balanced = False


def script_parses(script):
    """`node --check` parses the script without running it."""
    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False) as handle:
        handle.write(script)
    try:
        return subprocess.run(['node', '--check', handle.name], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return None
    finally:
        Path(handle.name).unlink(missing_ok=True)


def judge(reply, features, expect=None):
    match = re.search(r'```html\s*\n(.*?)(?:```|$)', reply, re.S)
    html = match.group(1) if match else reply
    parser = Structure()
    parser.feed(html)
    script = '\n'.join(re.findall(r'<script>(.*?)</script>', html, re.S))
    looked_up = set(re.findall(r"getElementById\(\s*['\"]([^'\"]+)['\"]\s*\)", script))
    static = all(f in STATIC for f in features)
    checks = {
        'complete_document': bool(re.search(r'<!doctype html', html, re.I)) and {'html', 'head', 'body'} <= parser.tags and '</html>' in html,
        'balanced_tags': parser.balanced and not parser.open,
        'css_and_script': static or ('style' in parser.tags and bool(script.strip())),
        'script_parses': (static and not script.strip()) or (bool(script.strip()) and script_parses(script) is True),
        'ids_exist': (static or bool(looked_up) or 'querySelector' in script) and looked_up <= parser.ids,
        'features_wired': (all(STATIC[f] in html.lower() for f in features) if static else
                           all(set(FEATURE_IDS[f]) <= parser.ids for f in features) and 'addEventListener' in script),
    }
    if expect is not None:
        checks['edit_applied'] = all(text in html for text in expect)
    return html, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', default='data/websites-v1.jsonl')
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--max-tokens', type=int, default=1900, help='longest page to generate (Cide 3 writes up to 4,000 tokens)')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    model, header, _ = Model.load(args.checkpoint)
    rows = [json.loads(line) for line in Path(args.data).read_text(encoding='utf-8').splitlines()]
    rows = [r for r in rows if r['split'] == 'validation'][:args.count]
    folder = (args.out.with_suffix('') if args.out else Path(tempfile.mkdtemp())).with_name((args.out.stem if args.out else 'websites') + '-pages')
    folder.mkdir(parents=True, exist_ok=True)
    results, started = [], time.monotonic()
    for index, row in enumerate(rows):
        messages = row['messages'][:-1]
        tokens, _ = model.prepare_prompt(messages, reserve=args.max_tokens)
        budget = model.config.context - len(tokens) - 8
        reply = ''.join(model.generate(messages, max_tokens=max(64, min(args.max_tokens, budget)), temperature=0))
        html, checks = judge(reply, row['features'], row.get('expect') if row.get('edits') else None)
        page = folder / f'site-{index}.html'
        page.write_text(html, encoding='utf-8')
        results.append({'prompt': messages[-1]['content'], 'turn': len(messages) // 2 + 1, 'edits': row.get('edits'),
                        'features': row['features'], 'checks': checks, 'working_structure': all(checks.values()),
                        'page': str(page), 'reply': reply[:6000]})
    report = {'checkpoint': args.checkpoint, 'steps': header.get('metadata', {}).get('steps'), 'context': model.config.context,
              'working_structure': f"{sum(r['working_structure'] for r in results)}/{len(results)}",
              'by_check': {name: f"{sum(r['checks'].get(name, False) for r in results)}/{sum(name in r['checks'] for r in results)}"
                           for name in dict.fromkeys(n for r in results for n in r['checks'])},
              'behavior_tested': False, 'pages': str(folder), 'seconds': round(time.monotonic() - started, 1), 'results': results}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('results', 'checkpoint')}))


if __name__ == '__main__':
    main()
