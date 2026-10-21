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
DEFAULT_MODEL = 'b1-v3'
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


def ask(args):
    from .server import validate_chat
    prompt = sys.stdin.read() if args.prompt in ([], ['-']) else ' '.join(args.prompt)
    request = {'messages': [{'role': 'user', 'content': prompt.strip()}],
               'max_tokens': args.max_tokens, 'temperature': args.temperature}
    try:
        if not prompt.strip():
            raise ValueError('Nothing to ask. Pass a prompt or pipe one in.')
        messages, max_tokens, temperature = validate_chat(request)
        model, _, _ = load(args)
        source, _, output = reply(model, messages, max_tokens, temperature)
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
               {"type": "cancel", "id": ...}
    Events:    ready, then start / token... / done (or error) for each chat id.
    """
    from .server import validate_chat
    protocol, lock = sys.stdout, threading.Lock()
    sys.stdout = sys.stderr  # stray prints must never corrupt the protocol stream

    def send(**event):
        with lock:
            protocol.write(json.dumps(event) + '\n')
            protocol.flush()
    try:
        model, header, path = load(args)
    except (ValueError, OSError) as error:
        send(type='error', message=str(error))
        return 1
    send(type='ready', name=model.name, checkpoint=path.name.removesuffix('.npz'),
         parameters=model.parameter_count, steps=header.get('metadata', {}).get('steps', 0),
         codename=model.codename, config=header['config'], tokenizer_spec=model.tokenizer.to_dict())
    requests, cancelled = queue.Queue(), set()

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
            source, omitted, output = reply(model, messages, max_tokens, temperature)
            send(type='start', id=rid, source=source, omitted=omitted)
            for piece in output:
                if rid in cancelled:
                    break
                send(type='token', id=rid, text=piece)
                chunks += 1
        except ValueError as error:
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
        command.add_argument('--temperature', type=float, default=0)
    model_options(commands.add_parser('chat', help='interactive terminal chat'))
    command = commands.add_parser('ask', help='answer one prompt')
    command.add_argument('prompt', nargs='*', help='prompt text; - or nothing reads stdin')
    command.add_argument('--json', action='store_true', help='print {"source", "text"} as JSON')
    model_options(command)
    command = commands.add_parser('stdio', help='JSON-lines protocol for editor extensions')
    command.add_argument('--model')
    command.add_argument('--checkpoint')
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
