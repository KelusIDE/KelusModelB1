"""Bounded instruction pilot for our own Kani .npz weights; never initializes a replacement base model.

Run from the repository root with python -m kaggle.kani_instruction. Uses the same
PyTorch implementation as the existing cloud pretrainer and exports back to Kelus.
The time limit stops this process, NOT Runpod billing; stop the Pod separately.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import signal
import time

import numpy as np
import torch
import torch.nn.functional as F

from kaggle.kani_advanced import Kani, kani_parameters
from kelus.tokenizer import PAD, conversation, from_dict


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_base(path):
    with np.load(path, allow_pickle=False) as archive:
        header = json.loads(str(archive['header']))
        c = header['config']
        if header['format'] != 'kelus-b1' or c.get('positions') != 'rope' or not c.get('tie_embeddings') or c.get('dropout', 0):
            raise ValueError('Expected Kani RoPE/tied-embedding checkpoint without dropout')
        tokenizer = from_dict(header['tokenizer'])
        net = Kani(c['vocab_size'], c['width'], c['layers'], c['heads'], c['context'])
        with torch.no_grad():
            for name, target in net.named_parameters():
                parts = name.split('.')
                transpose = False
                if name == 'token.weight':
                    source = 'token'
                elif parts[0] == 'final':
                    source = 'final.' + ('gain' if parts[-1] == 'weight' else 'bias')
                elif parts[2] in ('norm1', 'norm2'):
                    source = f'block{parts[1]}.{parts[2]}.' + ('gain' if parts[-1] == 'weight' else 'bias')
                else:
                    source, transpose = f'block{parts[1]}.{parts[2]}', True
                value = archive['parameter.' + source]
                value = value.T if transpose else value
                if tuple(value.shape) != tuple(target.shape) or not np.isfinite(value).all():
                    raise ValueError('Invalid weight: ' + source)
                target.copy_(torch.from_numpy(value.copy()))
    return net, tokenizer, header


def records(path, tokenizer, context):
    splits = {'train': [], 'validation': []}
    prompt_split, answer_split = {}, {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        split = row['split']
        if split not in splits:
            continue
        for message in row['messages']:
            key = ' '.join(message['content'].lower().split())
            groups = prompt_split if message['role'] == 'user' else answer_split
            if key in groups and groups[key] != split:
                raise ValueError('Exact prompt/answer leakage across data splits')
            groups[key] = split
        ids, supervised = conversation(row['messages'], tokenizer=tokenizer)
        if len(ids) - 1 > context:
            raise ValueError('Overlong example; refusing silent truncation')
        targets = np.array(ids[1:], dtype=np.int64)
        targets[~np.array(supervised[1:], dtype=bool)] = -100
        if not np.any(targets != -100):
            raise ValueError('Example has no assistant targets')
        splits[split].append((np.array(ids[:-1], dtype=np.int64), targets))
    if not all(splits.values()):
        raise ValueError('Both training and validation data are required')
    return splits


def batch(examples, device):
    length = max(len(x) for x, _ in examples)
    x = np.full((len(examples), length), PAD, dtype=np.int64)
    y = np.full_like(x, -100)
    for index, (ids, targets) in enumerate(examples):
        x[index, :len(ids)], y[index, :len(ids)] = ids, targets
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def amp(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' and torch.cuda.is_bf16_supported() else nullcontext()


def validate(net, examples, device, batch_size):
    net.eval()
    total, count = 0., 0
    with torch.no_grad():
        for offset in range(0, len(examples), batch_size):
            x, y = batch(examples[offset:offset + batch_size], device)
            with amp(device):
                logits = net(x)
                loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100, reduction='sum')
            total += loss.item()
            count += (y != -100).sum().item()
    net.train()
    return total / count


def export(net, header, path, metadata):
    value = {**header, 'metadata': {**header.get('metadata', {}), **metadata}}
    temp = Path(str(path) + '.tmp.npz')
    np.savez(temp, header=np.array(json.dumps(value)), **{'parameter.' + k: v for k, v in kani_parameters(net).items()})
    temp.replace(path)


def train(args):
    import fcntl
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; refusing an accidental CPU cloud run')
    args.run.mkdir(parents=True, exist_ok=True)
    lock = (args.run / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    checkpoint = args.run / 'checkpoint.pt'
    if (args.run / 'recipe.json').exists() != args.resume:
        raise ValueError('Existing run needs --resume; a new run must not use it')
    recipe = dict(parent_sha256=sha(args.checkpoint), data_sha256=sha(args.data), steps=args.steps,
                  batch=args.batch, accumulate=args.accumulate, lr=args.lr, seed=args.seed,
                  validation_examples=args.validation_examples)
    recipe_path = args.run / 'recipe.json'
    if args.resume and json.loads(recipe_path.read_text()) != recipe:
        raise ValueError('Resume recipe changed')
    if args.resume and not checkpoint.exists():
        raise ValueError('No resumable checkpoint was saved')
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    net, tokenizer, header = load_base(args.checkpoint)
    data = records(args.data, tokenizer, net.context)
    selection = np.random.default_rng(7742).permutation(len(data['validation']))[:args.validation_examples]
    validation = [data['validation'][i] for i in selection]
    recipe_path.write_text(json.dumps(recipe, indent=2) + '\n')
    net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    step, best, order, offset = 0, float('inf'), rng.permutation(len(data['train'])).tolist(), 0
    if args.resume:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        net.load_state_dict(saved['net'])
        optimizer.load_state_dict(saved['optimizer'])
        step, best, order, offset = saved['step'], saved['best'], saved['order'], saved['offset']
        rng.bit_generator.state = saved['rng']
        torch.set_rng_state(saved['torch_rng'])
    interrupted = False

    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    started = time.monotonic()

    def save(current):
        nonlocal best
        improved = current < best
        best = min(best, current)
        tmp = checkpoint.with_suffix('.tmp.pt')
        torch.save(dict(net=net.state_dict(), optimizer=optimizer.state_dict(), step=step, best=best,
                        order=order, offset=offset, rng=rng.bit_generator.state, torch_rng=torch.get_rng_state()), tmp)
        tmp.replace(checkpoint)
        metadata = dict(display_name='Kani Advanced instruction pilot', specialist='chat', stage='instruction tuning candidate',
                        steps=step, validation_loss=current, parent_sha256=recipe['parent_sha256'],
                        data_sha256=recipe['data_sha256'], training_examples=len(data['train']),
                        validation_examples=len(validation), trained_on=str(device), auto_deploy=False)
        export(net, header, args.run / 'latest.npz', metadata)
        if improved:
            export(net, header, args.run / 'best.npz', metadata)
    current = validate(net, validation, device, args.batch)
    with (args.run / 'metrics.jsonl').open('a') as stream:
        def report(**extra):
            row = dict(step=step, total_steps=args.steps, validation_loss=current,
                       elapsed_seconds=round(time.monotonic() - started, 1), **extra)
            print(json.dumps(row), flush=True)
            stream.write(json.dumps(row) + '\n'); stream.flush()
            temp = args.run / 'status.tmp.json'
            temp.write_text(json.dumps(row, indent=2) + '\n'); temp.replace(args.run / 'status.json')
        report(state='training', device=str(device), training_examples=len(data['train']))
        if not args.resume:
            save(current)
        while step < args.steps and not interrupted and time.monotonic() - started < args.max_seconds:
            optimizer.zero_grad(set_to_none=True)
            micro_batches = []
            for _ in range(args.accumulate):
                indices = []
                for _ in range(args.batch):
                    if offset >= len(order):
                        order, offset = rng.permutation(len(data['train'])).tolist(), 0
                    indices.append(order[offset]); offset += 1
                micro_batches.append([data['train'][i] for i in indices])
            denominator = sum(int((y != -100).sum()) for examples in micro_batches for _, y in examples)
            train_loss = 0.
            for examples in micro_batches:
                x, y = batch(examples, device)
                with amp(device):
                    logits = net(x)
                    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100, reduction='sum') / denominator
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss; previous checkpoint retained')
                loss.backward(); train_loss += loss.item()
            norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
            warmup = min(25, max(1, args.steps // 10))
            lr = args.lr * min(1., (step + 1) / warmup) * (0.1 + 0.9 * (1 + math.cos(math.pi * step / args.steps)) / 2)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.step(); step += 1
            if step % args.save_every == 0 or step == args.steps or interrupted or time.monotonic() - started >= args.max_seconds:
                current = validate(net, validation, device, args.batch)
                save(current)
            if step == 1 or step % 10 == 0 or step == args.steps:
                report(state='training', train_loss=round(train_loss, 5), gradient_norm=float(norm), lr=lr)

        current = validate(net, validation, device, args.batch)
        save(current)
        report(state='complete' if step == args.steps else 'stopped', reason='budget or signal' if step < args.steps else 'step target')
    lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', choices=['cuda', 'cpu', 'mps'], default='cuda')
    parser.add_argument('--steps', type=int, default=250)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--accumulate', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--seed', type=int, default=7741)
    parser.add_argument('--max-seconds', type=float, default=1800)
    parser.add_argument('--save-every', type=int, default=50)
    parser.add_argument('--validation-examples', type=int, default=64)
    args = parser.parse_args()
    if min(args.steps, args.batch, args.accumulate, args.max_seconds, args.save_every, args.validation_examples) <= 0 or not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.max_seconds):
        parser.error('Budgets, batch sizes, validation count and learning rate must be positive and finite')
    train(args)
