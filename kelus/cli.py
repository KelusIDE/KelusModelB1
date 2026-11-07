"""The kelus command: chat, one-shot answers, an editor protocol, and the project tools.

    kelus chat                      interactive terminal chat
    kelus ask "hello!"              one answer; reads stdin when the prompt is - or omitted
    kelus stdio                     JSON lines on stdin/stdout, used by the VS Code extension
    kelus models                    list local checkpoints
    kelus status                    foundation pretraining and the release watcher
    kelus serve|train|eval|readme   run kelus.server, kelus.train, kelus.benchmark, kelus.readme
                                    from the repository directory, passing the remaining arguments

Install with `python3 -m pip install -e .`, or run `python3 -m kelus.cli` without installing.
Models are named by checkpoint file: --model b1-v3 loads checkpoints/b1-v3.best.npz.
The default is the model live on kani.kelus.org.
"""
import argparse
import json
import os
from pathlib import Path
import queue
import runpy
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent.parent
def _latest_build(fallback='kani-b2-2f-v7-s10000'):
    """The checkpoint of the "latest" build in kani-builds.json (kelus.release updates it), so every app agrees."""
    try:
        return next(b['checkpoint'] for b in json.loads((ROOT / 'kani-builds.json').read_text()) if b['id'] == 'latest')
    except (OSError, ValueError, StopIteration, KeyError):
        return fallback


DEFAULT_MODEL = _latest_build()
PAINTER = ROOT / 'checkpoints' / 'kani-painter-v4b.npz'
TOOLS = {'serve': 'server', 'train': 'train', 'eval': 'benchmark', 'readme': 'readme'}


def checkpoint_path(model=None, checkpoint=None):
    if checkpoint:
        return Path(checkpoint).expanduser().resolve()
    name = model or DEFAULT_MODEL
    for candidate in (ROOT / 'checkpoints' / f'{name}.best.npz', ROOT / 'checkpoints' / f'{name}.npz'):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'No checkpoint named "{name}" in {ROOT / "checkpoints"}. '
                            'Run `kelus models`, train one with `kelus train`, or pass --checkpoint.')


def load(args):
    from .model import Model
    path = checkpoint_path(args.model, args.checkpoint)
    model, header, _ = Model.load(path)
    return model, header, path


def reply(model, messages, max_tokens, temperature):
    """Return (source, omitted turns, text pieces). Local helper answers keep their label."""
    from .runtime import prepare_reply
    retained, omitted, tool = prepare_reply(model, messages, max_tokens)
    if tool:
        return tool[1], omitted, iter([tool[0]])
    return 'model', omitted, model.generate(retained, max_tokens=max_tokens, temperature=temperature)


SKETCH = ROOT / 'training-runs' / 'kani-sketch-v1' / 'best.npz'
_sketch = {}


def sketched(description):
    """(subject, base64 PNG) of a Kani Sketch drawing of the subject named in `description`, or None."""
    import base64
    import io
    if not SKETCH.exists():
        return None
    from .sketch import find_subject, render, sample, tidy
    if 'model' not in _sketch:
        from .model import Model
        _sketch['model'] = Model.load(SKETCH)[0]
    model = _sketch['model']
    subject = find_subject(description, model.tokenizer.categories)
    if subject is None:
        return None
    for seed in range(5):
        strokes = tidy(model.tokenizer.decode(sample(model, subject, seed=seed)))
        if sum(len(stroke) for stroke in strokes) >= 6:
            break
    buffer = io.BytesIO()
    render(strokes).save(buffer, format='PNG')
    return subject, base64.b64encode(buffer.getvalue()).decode('ascii')


PIXELS = ROOT / 'training-runs' / 'kani-pixels-v2'
MLX_PYTHON = ROOT / '.venv' / 'bin' / 'python'


def pixels_drawn(description):
    """(subject, base64 PNG) of four Kani Pixels v2 images of the subject named in `description`, or None."""
    import base64
    import subprocess
    import tempfile
    if not (PIXELS / 'latest.npz').exists():
        raise ValueError('Kani Pixels v2 is not trained yet; choose Kani Painter instead')
    python = MLX_PYTHON if MLX_PYTHON.exists() else Path(sys.executable)
    with tempfile.TemporaryDirectory() as folder:
        out = Path(folder) / 'pixels.png'
        try:
            done = subprocess.run([str(python), '-m', 'kelus.pixel_dit', 'sample', '--run', str(PIXELS), '--prompt', description,
                                   '--out', str(out), '--count', '4', '--columns', '2', '--scale', '4'],
                                  cwd=ROOT, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            raise ValueError('Kani Pixels took longer than 2 minutes; try again or choose Kani Painter') from None
        if done.returncode:
            raise ValueError('Kani Pixels failed: ' + (done.stderr.strip().splitlines() or ['no output'])[-1])
        subject = json.loads(done.stdout.strip().splitlines()[-1])['subject']
        return (subject, base64.b64encode(out.read_bytes()).decode('ascii')) if subject else None


def painting(program):
    """Base64 PNG of a Kani Paint program, rendered by the checked interpreter (never exec)."""
    import base64
    import io
    from .paint import code_block, render
    buffer = io.BytesIO()
    render(code_block(program)).save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('ascii')


def ask(args):
    from .server import validate_chat
    prompt = sys.stdin.read() if args.prompt in ([], ['-']) else ' '.join(args.prompt)
    request = {'messages': [{'role': 'user', 'content': prompt.strip()}],
               'max_tokens': args.max_tokens, 'temperature': args.temperature}
    try:
        if not prompt.strip():
            raise ValueError('Nothing to ask. Pass a prompt or pipe one in.')
        messages, max_tokens, temperature = validate_chat(request)
        _, coding_max_tokens, _ = validate_chat({**request, 'max_tokens': args.cide_max_tokens})
        model, header, _ = load(args)
        if header.get('metadata', {}).get('specialist') == 'coding':
            from .cide import context
            source, output = 'cide', model.generate(context(model, messages, coding_max_tokens), max_tokens=coding_max_tokens, temperature=temperature)
        else:
            source, _, output = reply(model, messages, max_tokens, temperature)
        if source == 'model':
            from .cide import handoff
            events = list(handoff(output, messages, max_tokens, temperature, coding_max_tokens=coding_max_tokens))
            if any(e.get('source') == 'cide' for e in events):
                source = 'cide'
            output = (e['text'] for e in events if e['type'] == 'token')
        if args.json:
            print(json.dumps({'source': source, 'text': ''.join(output)}))
            return
        if source != 'model':
            print(f'[{source}] ', end='')
        for piece in output:
            print(piece, end='', flush=True)
        print()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f'kelus ask: {error}')


def stdio(args):
    """Serve one JSON request or event per line until stdin closes.

    Requests:  {"type": "chat", "id": ..., "messages": [...], "max_tokens": 240, "temperature": 0}
               "reading": true marks a search-results prompt (site/reading.js): generated directly,
               without local helpers, which would otherwise react to words inside the results.
               {"type": "cancel", "id": ...}
    Events:    ready, then start / token... / done (or error) for each chat id.
               Drawing requests go to Kani Painter (source "painter"): its program streams as
               tokens, then an image event carries the rendered PNG as base64.
    """
    from .paint import PaintError, draw_request, fix_spelling, unknown_subject
    from .server import validate_chat
    protocol, lock = sys.stdout, threading.Lock()
    sys.stdout = sys.stderr

    def send(**event):
        with lock:
            protocol.write(json.dumps(event) + '\n')
            protocol.flush()
    try:
        model, header, path = load(args)
    except (ValueError, OSError) as error:
        send(type='error', message=str(error))
        return 1
    painter_path = Path(args.painter) if args.painter else PAINTER if PAINTER.exists() else None
    send(type='ready', name=header.get('metadata', {}).get('display_name', model.name), checkpoint=path.name.removesuffix('.npz'),
         parameters=model.parameter_count, steps=header.get('metadata', {}).get('steps', 0),
         codename=header.get('metadata', {}).get('codename', model.codename), config=header['config'], tokenizer_spec=model.tokenizer.to_dict(),
         painter=painter_path is not None)
    requests, cancelled, painter = queue.Queue(), set(), None

    def read():
        for line in sys.stdin:
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                send(type='error', message='Invalid JSON request')
                continue
            if isinstance(request, dict) and request.get('type') == 'cancel':
                cancelled.add(request.get('id'))
            else:
                requests.put(request)
        requests.put(None)
    threading.Thread(target=read, daemon=True).start()
    while (request := requests.get()) is not None:
        rid = request.get('id') if isinstance(request, dict) else None
        if not isinstance(request, dict) or request.get('type') != 'chat':
            send(type='error', id=rid, message='Unknown request; expected type "chat" or "cancel".')
            continue
        started, chunks = time.monotonic(), 0
        try:
            messages, max_tokens, temperature = validate_chat(request)
            _, coding_max_tokens, _ = validate_chat({**request, 'max_tokens': request.get('cide_max_tokens', 4096)})
            expert = request.get('expert')
            if expert not in (None, 'cide'):
                raise ValueError('Unknown expert; expected cide')
            direct_cide = expert == 'cide' or messages[-1]['content'].startswith('/cide ')
            if messages[-1]['content'].startswith('/cide '):
                messages = messages[:-1] + [{'role': 'user', 'content': messages[-1]['content'][6:].strip()}]
            image_model = request.get('image_model', 'painter')
            if image_model not in ('pixels', 'painter'):
                raise ValueError('Image model must be pixels or painter')
            drawing = None if direct_cide else draw_request(messages[-1]['content'])
            drawing = drawing and fix_spelling(drawing)


            pixel = drawing and image_model == 'pixels' and unknown_subject(drawing) and pixels_drawn(drawing)
            if direct_cide:
                from .cide import answer
                for event in answer(messages, coding_max_tokens, temperature, cancelled=lambda: rid in cancelled):
                    send(id=rid, **event)
                    chunks += event['type'] == 'token'
            elif pixel:
                subject, png = pixel
                send(type='start', id=rid, source='pixels', omitted=0)
                send(type='token', id=rid, text=f'Kani Pixels v2 painted {subject}: four tries at 64×64 pixels, from a diffusion model '
                                                'trained from scratch on Open Images photos. Experimental and blurry.')
                if rid not in cancelled:
                    send(type='image', id=rid, png=png)
            elif drawing and not painter_path:
                raise ValueError('No Painter checkpoint installed; choose Kani Pixels instead')
            elif drawing and (refusal := unknown_subject(drawing)):

                drawn = sketched(drawing)
                if drawn:
                    subject, png = drawn
                    send(type='start', id=rid, source='sketch', omitted=0)
                    send(type='token', id=rid, text=f"Kani Sketch drew {'an' if subject[0].lower() in 'aeiou' else 'a'} {subject}: a doodle, learned from 414,000 Quick, Draw! drawings.")
                    send(type='image', id=rid, png=png)
                else:
                    send(type='start', id=rid, source='painter', omitted=0)
                    send(type='token', id=rid, text=refusal + ' Kani Sketch, which doodles 345 everyday subjects from Quick, Draw!, has not learned it either.')
            elif drawing:
                if painter is None:
                    from .model import Model
                    painter = Model.load(painter_path)[0]
                send(type='start', id=rid, source='painter', omitted=0)
                program = ''
                for piece in painter.generate([{'role': 'user', 'content': drawing}], max_tokens=900, temperature=0, prefix='```python\n'):
                    if rid in cancelled:
                        break
                    program += piece
                    send(type='token', id=rid, text=piece)
                    chunks += 1
                if rid not in cancelled:
                    if not program.rstrip().endswith('```'):
                        program += '\n```'
                        send(type='token', id=rid, text='\n```')
                    try:
                        send(type='image', id=rid, png=painting(program))
                    except PaintError as problem:
                        send(type='token', id=rid, text=f'\n\nThe program did not render: {problem}')
                    except ImportError:
                        send(type='token', id=rid, text='\n\nRendering needs Pillow for this Python: python3 -m pip install pillow')
            else:
                if header.get('metadata', {}).get('specialist') == 'coding' and not request.get('reading'):
                    max_tokens = coding_max_tokens
                if request.get('reading') is True:
                    source, omitted, output = 'model', 0, model.generate(messages[-1:], max_tokens=max_tokens, temperature=temperature)
                else:
                    source, omitted, output = reply(model, messages, max_tokens, temperature)
                is_cide = header.get('metadata', {}).get('specialist') == 'coding'
                send(type='start', id=rid, source='cide' if is_cide and source == 'model' else source, omitted=omitted)
                from .cide import handoff
                if source == 'model' and not request.get('reading') and not is_cide:
                    events = handoff(output, messages, max_tokens, temperature, cancelled=lambda: rid in cancelled, coding_max_tokens=coding_max_tokens)
                else:
                    events = ({'type': 'token', 'text': piece} for piece in output)
                for event in events:
                    if rid in cancelled:
                        break
                    send(id=rid, **event)
                    chunks += event['type'] == 'token'
        except (ValueError, OSError) as error:
            send(type='error', id=rid, message=str(error))
            continue
        finally:
            stopped = rid in cancelled
            cancelled.discard(rid)
        send(type='done', id=rid, chunks=chunks, seconds=round(time.monotonic() - started, 2), cancelled=stopped)
    return 0


def models(args):
    import numpy as np
    paths = sorted((ROOT / 'checkpoints').glob('*.npz'))
    if not paths:
        print('No checkpoints yet. Train one with `kelus train`.')
    default = checkpoint_path() if any(p.name.startswith(DEFAULT_MODEL + '.') for p in paths) else None
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            header = json.loads(str(archive['header']))
        c, steps = header['config'], header.get('metadata', {}).get('steps', '?')
        mark = '*' if path == default else ' '
        print(f"{mark} {path.name.removesuffix('.npz'):28} {c['layers']} layers · width {c['width']} · "
              f"context {c['context']} · {steps} steps · {path.stat().st_size / 1e6:.1f} MB")
    if default:
        print(f'\n* default. Use one with: kelus chat --model {DEFAULT_MODEL}')


def status(args):
    foundation = ROOT / 'training-runs/b1-foundation/status.json'
    if foundation.exists():
        value = json.loads(foundation.read_text())
        line = f"Foundation pretraining: {value['state']} · step {value['step']:,}/{value['total_steps']:,}"
        if value.get('validation_loss') is not None:
            line += f" · validation loss {value['validation_loss']:.3f}"
        if value['state'] == 'running' and value.get('eta_seconds'):
            line += f" · about {value['eta_seconds'] / 3600:.1f} h left"
        age = time.time() - value.get('updated_at', time.time())
        print(line + (f' (status written {age / 60:.0f} min ago)' if age > 120 else ''))
    else:
        print('Foundation pretraining: not started')
    release = ROOT / 'training-runs/auto-deploy/status.json'
    if release.exists():
        value = json.loads(release.read_text())
        print(f"Release watcher: {value['state'].replace('_', ' ')} · {value.get('detail', '')}")
    else:
        print('Release watcher: not started')


def parser():
    p = argparse.ArgumentParser(prog='kelus', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = p.add_subparsers(dest='command', metavar='command')

    def model_options(command):
        command.add_argument('--model', help=f'checkpoint name in checkpoints/ (default {DEFAULT_MODEL})')
        command.add_argument('--checkpoint', help='path to a .npz checkpoint; overrides --model')
        command.add_argument('--max-tokens', type=int, default=240)
        command.add_argument('--cide-max-tokens', type=int, default=4096, help='Coding specialist output limit (1–4096)')
        command.add_argument('--temperature', type=float, default=0)
    model_options(commands.add_parser('chat', help='interactive terminal chat'))
    command = commands.add_parser('ask', help='answer one prompt')
    command.add_argument('prompt', nargs='*', help='prompt text; - or nothing reads stdin')
    command.add_argument('--json', action='store_true', help='print {"source", "text"} as JSON')
    model_options(command)
    command = commands.add_parser('stdio', help='JSON-lines protocol for editor extensions')
    command.add_argument('--model')
    command.add_argument('--checkpoint')
    command.add_argument('--painter', help=f'Kani Painter checkpoint for drawing requests (default {PAINTER.relative_to(ROOT)} if present)')
    commands.add_parser('models', help='list local checkpoints')
    commands.add_parser('status', help='foundation pretraining and release watcher')
    for name, module in TOOLS.items():
        commands.add_parser(name, help=f'run kelus.{module} (arguments pass through)')
    return p


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in TOOLS:
        os.chdir(ROOT)
        sys.argv = [f'kelus {argv[0]}'] + argv[1:]
        runpy.run_module(f'kelus.{TOOLS[argv[0]]}', run_name='__main__', alter_sys=True)
        return
    args = parser().parse_args(argv)
    if args.command == 'chat':
        try:
            path = checkpoint_path(args.model, args.checkpoint)
        except FileNotFoundError as error:
            raise SystemExit(f'kelus chat: {error}')
        sys.argv = ['kelus chat', '--checkpoint', str(path), '--max-tokens', str(args.max_tokens), '--temperature', str(args.temperature)]
        runpy.run_module('kelus.chat', run_name='__main__', alter_sys=True)
    elif args.command == 'ask':
        ask(args)
    elif args.command == 'stdio':
        sys.exit(stdio(args))
    elif args.command == 'models':
        models(args)
    elif args.command == 'status':
        status(args)
    else:
        parser().print_help()


if __name__ == '__main__':
    main()
