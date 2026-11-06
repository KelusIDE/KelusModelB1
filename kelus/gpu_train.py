"""Kani training on the Mac's GPU (MLX), writing the same files as the CPU trainers.

    instruct   like kelus.train --init-from: chat records (kelus.data.Dataset), log lines and
               checkpoints in kelus.train's format, so kelus.progress, the evaluations and
               kelus.train --resume all work on the result
    pretrain   like kelus.pretrain: next-token windows from a kelus-raw-corpus-v1 directory, with
               status.json, metrics.jsonl, latest.npz and best.npz in the run directory

Learning-rate schedules, loss, AdamW and clipping match the CPU trainers (kelus.gpu checks the math).

    .venv/bin/python -m kelus.gpu_train instruct --init-from A.npz --data data/instruct-v7.jsonl --out checkpoints/B.npz --steps 10000
    .venv/bin/python -m kelus.gpu_train pretrain --data data/sketch-v1 --run training-runs/kani-sketch-v1 --steps 12000 --context 256 --width 256
"""
import argparse
import dataclasses
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time

import mlx.core as mx
import numpy as np

from .gpu import AdamW, forward_fast, limit_memory, loss, parameters, rope_tables, save
from .model import Config, Model
from .tokenizer import PAD

BUCKET = 128


def compiled_step(config, cos, sin, checkpoint_blocks=False):
    gradient = mx.value_and_grad(loss)
    return mx.compile(lambda params, x, y: gradient(params, x, y, config, cos, sin, True, checkpoint_blocks))


def mean_loss(params, config, cos, sin, batches):
    """Token-weighted validation loss, as kelus.train.evaluate computes it."""
    total, count = 0.0, 0
    for x, y in batches:
        targets = int((y >= 0).sum())
        if targets:
            total += float(loss(params, mx.array(x), mx.array(y), config, cos, sin, True)) * targets
            count += targets
    return total / max(count, 1)


def restore(optimizer, header, states):
    meta = header.get('optimizer', {})
    for prefix, destination in (('m', optimizer.m), ('v', optimizer.v)):
        for name in destination:
            destination[name] = mx.array(states[f'adam.{prefix}.{name}'])
    optimizer.step_count = int(meta.get('step_count', 0))


def instruct(args):
    from .data import Dataset
    best_path = Path(args.out).with_suffix('.best.npz')
    if Path(args.out).exists() and not args.overwrite and not args.resume:
        raise SystemExit(f'{args.out} exists. Resume it or pass --overwrite.')
    source = args.resume or args.init_from
    model, header, states = Model.load(source)
    config = model.config
    if config.dropout:
        raise SystemExit('GPU training does not implement dropout; this checkpoint was trained with it')
    if args.context and args.context != config.context:
        if args.resume or config.positions != 'rope':
            raise SystemExit('A new context length needs --init-from and a checkpoint with rotary positions')


        longer = Model(dataclasses.replace(config, context=args.context), model.tokenizer)
        for name, parameter in model.parameters.items():
            longer.parameters[name].data = parameter.data
        longer.name, model, config = model.name, longer, longer.config
    params, (cos, sin) = parameters(model), rope_tables(model)
    optimizer = AdamW(params, weight_decay=args.weight_decay)
    if args.resume:
        metadata = dict(header.get('metadata', {}))
        args.seed = metadata.get('seed', args.seed)
        restore(optimizer, header, states)
    else:
        metadata = {'parent_steps': header.get('metadata', {}).get('steps', 0), 'parent_sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest()}
    del states
    last_reply_only = args.last_reply_only or (args.resume and metadata.get('training_targets') == 'last assistant reply, earlier turns provide context')
    dataset = Dataset(args.data, seed=args.seed, tokenizer=model.tokenizer, rust_tokenizer=args.rust_tokenizer,
                      last_reply_only=last_reply_only)
    if args.resume and metadata.get('data_sha256') != dataset.fingerprint:
        raise SystemExit('Resume data differs from the checkpoint.')
    rng = np.random.default_rng(args.seed + optimizer.step_count)
    if args.validation_examples and args.validation_examples < len(dataset.validation):

        keep = np.sort(np.random.default_rng(2401).choice(len(dataset.validation), args.validation_examples, replace=False))
        dataset.validation = [dataset.validation[i] for i in keep]
    validation = lambda: mean_loss(params, config, cos, sin, dataset.validation_batches(args.batch, config.context))
    initial = validation()
    print(json.dumps({'name': model.name, 'parameters': model.parameter_count, 'training_examples': len(dataset.training),
                      'validation_examples': len(dataset.validation), 'dataset_tokens': dataset.token_count,
                      'initial_validation_loss': initial, 'device': 'gpu (mlx)', 'tokenizer_backend': dataset.tokenizer_backend}), flush=True)
    metadata.update({'data_sha256': dataset.fingerprint, 'seed': args.seed, 'base_learning_rate': args.lr, 'training_examples': len(dataset.training),
                     'validation_examples': len(dataset.validation), 'dataset_tokens': dataset.token_count, 'educational_demo': True,
                     'validation_method': 'all assistant targets, overlap scored once', 'tokenizer': model.tokenizer.kind,
                     'training_targets': 'last assistant reply, earlier turns provide context' if last_reply_only else 'all assistant replies',
                     'balanced_languages': False, 'trained_on': 'gpu (mlx)',
                     'tokenizer_backend': dataset.tokenizer_backend})
    best = initial
    if not best_path.exists() or args.overwrite:
        save(model, params, optimizer, best_path, {**metadata, 'steps': optimizer.step_count, 'validation_loss': initial,
                                                  'selection': 'lowest full-validation loss'}, args.lr, weights_only=args.weights_only_best,
             compression_level=args.checkpoint_compression_level)
    step_fn = compiled_step(config, cos, sin, args.gradient_checkpointing)
    log_path = Path(args.out).with_suffix('.training.jsonl')
    started = time.monotonic()
    warmup = min(50, max(1, args.steps // 10))
    with log_path.open('a', encoding='utf-8') as log:
        for local in range(args.steps):
            x, y = dataset.batch('train', args.batch, config.context, rng, trim=True, group_width=BUCKET if args.group_lengths else 0)
            width = min(config.context, -(-x.shape[1] // BUCKET) * BUCKET)
            if width > x.shape[1]:
                x = np.pad(x, ((0, 0), (0, width - x.shape[1])), constant_values=PAD)
                y = np.pad(y, ((0, 0), (0, width - y.shape[1])), constant_values=-1)
            value, grads = step_fn(params, mx.array(x), mx.array(y))
            progress = max(0, local - warmup) / max(1, args.steps - warmup)
            learning_rate = args.lr * min(1, (local + 1) / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
            params, norm = optimizer.update(params, grads, learning_rate)
            mx.eval(params, optimizer.m, optimizer.v, value, norm)
            if not math.isfinite(float(value)):
                raise FloatingPointError('Nonfinite loss; the last checkpoint is preserved')
            step = optimizer.step_count
            if args.report_every and step % args.report_every == 0 and step % args.log_every:

                report = {'step': step, 'train_loss': round(float(value), 4), 'width': int(x.shape[1]),
                          'elapsed_seconds': round(time.monotonic() - started, 1)}
                print(json.dumps(report), flush=True)
            if local == 0 or step % args.log_every == 0 or local == args.steps - 1:
                current = validation()
                metadata.update({'steps': step, 'train_loss': float(value), 'validation_loss': current,
                                 'trained_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'context_tokens': config.context,
                                 'rng_state': rng.bit_generator.state})
                report = {'step': step, 'train_loss': round(float(value), 4), 'validation_loss': round(current, 4),
                          'gradient_norm': round(float(norm), 4), 'elapsed_seconds': round(time.monotonic() - started, 1)}
                print(json.dumps(report), flush=True)
                log.write(json.dumps(report) + '\n'); log.flush()
                save(model, params, optimizer, args.out, metadata, learning_rate,
                     compression_level=args.checkpoint_compression_level)
                if current < best:
                    best = current
                    save(model, params, optimizer, best_path, {**metadata, 'selection': 'lowest full-validation loss'}, learning_rate,
                         weights_only=args.weights_only_best, compression_level=args.checkpoint_compression_level)
    print(f'Saved {args.out} · {args.steps} new training steps', flush=True)


def pretrain(args):
    from .pretrain import Corpus
    corpus = Corpus(args.data)
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    latest, best_path = run / 'latest.npz', run / 'best.npz'
    lock = (run / 'run.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise ValueError('This training run is already active')
    try:
        return _pretrain_locked(args, corpus, run, latest, best_path)
    finally:
        lock.close()


def _pretrain_locked(args, corpus, run, latest, best_path):
    from .corpus import digest
    from .pretrain import write_json
    if args.resume and args.init_from:
        raise ValueError('Choose resume or init-from')
    if args.resume and not latest.exists():
        raise ValueError('No checkpoint to resume')
    if args.init_from and Path(args.init_from).resolve() in (latest.resolve(), best_path.resolve()):
        raise ValueError('Parent checkpoint must be outside the destination run')
    if args.resume:
        model, header, states = Model.load(latest)
        metadata = dict(header.get('metadata', {}))
        if metadata.get('corpus_sha256') != corpus.fingerprint:
            raise ValueError('Resume corpus differs from the checkpoint')
    else:
        if latest.exists() and not args.overwrite:
            raise SystemExit(f'{latest} exists. Pass --resume or --overwrite.')
        if args.init_from:
            model, parent, states = Model.load(args.init_from)
            del states
            metadata = {'initialization': 'Continued from our own Kani weights; fresh optimizer',
                        'parent_sha256': digest(Path(args.init_from)), 'parent_metadata': parent.get('metadata', {})}
        else:
            model = Model(Config(args.context, args.width, args.heads, args.layers, args.seed, corpus.tokenizer.vocab_size, 'rope', True, 0.0), corpus.tokenizer)
            metadata = {'initialization': 'random weights; no pretrained model'}
        header, states = {}, {}
    config = model.config
    if model.tokenizer.to_dict() != corpus.tokenizer.to_dict():
        raise ValueError('Checkpoint tokenizer differs from the corpus')
    recipe = {key: getattr(args, key) for key in ('steps', 'batch', 'accumulate', 'lr', 'weight_decay', 'seed', 'validation_batches', 'gradient_checkpointing')}
    if args.resume and metadata.get('recipe') and metadata['recipe'] != recipe:
        raise ValueError('Resume must keep the original training recipe')
    if config.dropout:
        raise ValueError('GPU training does not implement dropout')
    params, (cos, sin) = parameters(model), rope_tables(model)
    optimizer = AdamW(params, weight_decay=args.weight_decay)
    if args.resume and states:
        restore(optimizer, header, states)
    elif args.resume:
        raise ValueError('Resume checkpoint has no optimizer state; use init-from in a new run')
    del states
    if optimizer.step_count >= args.steps:
        print('Run already reached its total-step budget.', flush=True)
        return
    rng = np.random.default_rng(args.seed + optimizer.step_count)
    if metadata.get('rng_state'):
        rng.bit_generator.state = metadata['rng_state']
    validation_rng = np.random.default_rng(2401)
    windows = [corpus.batch('validation', args.batch, config.context, validation_rng) for _ in range(args.validation_batches)]
    validation = lambda: mean_loss(params, config, cos, sin, windows)
    best = float(metadata.get('best_validation_loss', math.inf))
    step_fn = compiled_step(config, cos, sin, args.gradient_checkpointing)
    started, first = time.monotonic(), optimizer.step_count
    last_loss, last_val = None, metadata.get('validation_loss')
    warmup = min(100, max(1, args.steps // 20))
    tokens_per_step = args.batch * args.accumulate * config.context
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    backup = Path(args.backup_dir) if args.backup_dir else None
    if backup:
        backup.mkdir(parents=True, exist_ok=True)
        if backup.resolve() == run.resolve():
            raise ValueError('Backup directory must differ from the working run')
    handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}

    def status(state):
        rate = (optimizer.step_count - first) / max(time.monotonic() - started, 1e-9)
        value = {'stage': 'pretraining', 'state': state, 'pid': os.getpid(), 'step': optimizer.step_count, 'total_steps': args.steps,
                 'parameters': model.parameter_count, 'context': config.context, 'tokens_per_step': tokens_per_step,
                 'tokens_seen': optimizer.step_count * tokens_per_step, 'train_loss': last_loss, 'validation_loss': last_val,
                 'steps_per_second': round(rate, 5), 'eta_seconds': round((args.steps - optimizer.step_count) / rate) if rate else None,
                 'updated_at': time.time(), 'checkpoint': str(latest.resolve()), 'data': str(Path(args.data).resolve()), 'device': 'gpu (mlx)'}
        value['device'] = str(mx.default_device()) + ' (mlx)'
        value['recipe'] = recipe
        write_json(run / 'status.json', value)
        return value

    def checkpoint(state, learning_rate):
        nonlocal best, last_val, metadata
        last_val = validation()
        metadata = {**metadata, 'stage': 'pretraining', 'corpus_sha256': corpus.fingerprint, 'steps': optimizer.step_count,
                    'rng_state': rng.bit_generator.state, 'train_loss': last_loss, 'validation_loss': last_val,
                    'best_validation_loss': min(best, last_val), 'tokens_seen': optimizer.step_count * tokens_per_step,
                    'recipe': recipe, 'validation_method': 'Fixed distinct windows; sampled loss, not a capability benchmark'}
        save(model, params, optimizer, latest, metadata, learning_rate)
        if last_val < best:
            best = last_val
            save(model, params, optimizer, best_path, metadata, learning_rate)
        event = status(state)
        with (run / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(event) + '\n')
        if backup:
            for source in (latest, best_path, run / 'status.json', run / 'metrics.jsonl'):
                target = backup / source.name

                if target.exists() and target.stat().st_mtime_ns == source.stat().st_mtime_ns:
                    continue
                temp = target.with_suffix(target.suffix + '.tmp')
                shutil.copy2(source, temp)
                temp.replace(target)
        print(json.dumps(event), flush=True)

    learning_rate = args.lr
    try:
        if not args.resume:
            checkpoint('running', learning_rate)
        else:
            status('running')
        while optimizer.step_count < args.steps:
            if stop_requested or (run / 'stop.request').exists() or (args.max_seconds and time.monotonic() - started >= args.max_seconds):
                break
            grads, loss_sum = None, 0.0
            for _ in range(args.accumulate):
                x, y = corpus.batch('train', args.batch, config.context, rng)
                value, part = step_fn(params, mx.array(x), mx.array(y))
                if grads is None:
                    grads = {name: value / args.accumulate for name, value in part.items()}
                else:
                    grads = {name: grads[name] + value / args.accumulate for name, value in part.items()}
                mx.eval(grads, value)
                loss_sum += float(value) / args.accumulate
            if not math.isfinite(loss_sum):
                raise FloatingPointError('Nonfinite loss; previous checkpoint retained')
            step = optimizer.step_count + 1
            progress = max(0, step - warmup) / max(1, args.steps - warmup)
            learning_rate = args.lr * min(1, step / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
            params, norm = optimizer.update(params, grads, learning_rate)
            mx.eval(params, optimizer.m, optimizer.v, value)
            last_loss = loss_sum
            if not math.isfinite(last_loss):
                raise FloatingPointError('Nonfinite loss; previous checkpoint retained')
            if optimizer.step_count % args.checkpoint_every == 0:
                checkpoint('running', learning_rate)
            elif optimizer.step_count % 10 == 0:
                status('running')
        checkpoint('complete' if optimizer.step_count >= args.steps else 'stopped', learning_rate)
    except BaseException as error:
        event = status('failed')
        event['error'] = str(error)
        write_json(run / 'status.json', event)
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    command = commands.add_parser('instruct')
    command.add_argument('--init-from')
    command.add_argument('--resume')
    command.add_argument('--data', required=True)
    command.add_argument('--out', required=True)
    command.add_argument('--overwrite', action='store_true')
    command.add_argument('--steps', type=int, default=10000)
    command.add_argument('--batch', type=int, default=8)
    command.add_argument('--lr', type=float, default=0.0003)
    command.add_argument('--weight-decay', type=float, default=0.01)
    command.add_argument('--seed', type=int, default=42)
    command.add_argument('--log-every', type=int, default=1000)
    command.add_argument('--validation-examples', type=int, help='validate on this many sampled examples (default: all)')
    command.add_argument('--rust-tokenizer', action='store_true', help='Use the local Rust BPE encoder for dataset preparation')
    command.add_argument('--last-reply-only', action='store_true', help='Train the latest assistant turn while retaining previous turns as context')
    command.add_argument('--weights-only-best', action='store_true', help='Save best weights without optimizer; latest remains resumable')
    command.add_argument('--checkpoint-compression-level', type=int, choices=range(10), default=6,
                         help='Lossless checkpoint ZIP compression; level 1 saves CPU time at a small size cost')
    command.add_argument('--report-every', type=int, default=0, help='print step and loss this often, without validation (0: off)')
    command.add_argument('--group-lengths', action='store_true',
                         help='batch examples of similar length together (random batches of 8 are about two-thirds padding)')
    command.add_argument('--context', type=int, default=0,
                         help='train at this context length (up to 4096) instead of the checkpoint\'s; rotary checkpoints only')
    command.add_argument('--gradient-checkpointing', action='store_true', help='Recompute block activations to reduce memory use')
    command = commands.add_parser('pretrain')
    command.add_argument('--data', required=True)
    command.add_argument('--run', required=True)
    command.add_argument('--resume', action='store_true')
    command.add_argument('--init-from', help='Start a new run from our own checkpoint with a fresh optimizer')
    command.add_argument('--accumulate', type=int, default=1)
    command.add_argument('--gradient-checkpointing', action='store_true', help='Recompute block activations to reduce memory use')
    command.add_argument('--max-seconds', type=float, default=0, help='Save and stop after this session budget; 0 disables the limit')
    command.add_argument('--backup-dir', help='Copy checkpoints and progress to persistent storage after each save')
    command.add_argument('--overwrite', action='store_true')
    command.add_argument('--steps', type=int, required=True, help='total optimizer steps, including earlier runs')
    for name, default in (('context', 1024), ('width', 320), ('heads', 8), ('layers', 6), ('batch', 8), ('seed', 919),
                          ('checkpoint_every', 500), ('validation_batches', 16)):
        command.add_argument('--' + name.replace('_', '-'), type=int, default=default)
    command.add_argument('--lr', type=float, default=0.0005)
    command.add_argument('--weight-decay', type=float, default=0.01)
    for command in commands.choices.values():
        command.add_argument('--memory-gb', type=float, default=10, help='GPU memory cap for this process')
    args = parser.parse_args()
    if args.command == 'pretrain':
        if not (args.steps > 0 and args.batch > 0 and args.accumulate > 0 and args.checkpoint_every > 0
                and args.validation_batches > 0 and args.lr > 0 and math.isfinite(args.lr)
                and math.isfinite(args.max_seconds) and args.max_seconds >= 0):
            parser.error('Invalid training budget')
    limit_memory(args.memory_gb)
    if args.command == 'instruct':
        if bool(args.init_from) == bool(args.resume):
            parser.error('Pass exactly one of --init-from or --resume')
        instruct(args)
    else:
        pretrain(args)


if __name__ == '__main__':
    main()
