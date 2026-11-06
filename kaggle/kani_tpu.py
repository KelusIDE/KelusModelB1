"""Kani Advanced on a TPU: the same model, checkpoint and text as kani_advanced.py, trained with JAX on Kaggle's TPU
v5e-8 (8 chips) instead of PyTorch on two T4s.

Kaggle's TPU hours are a quota of their own (about 20 a week, beside the GPUs' 30). Both trainers read and write the
same kani-advanced/checkpoint.pt (PyTorch's format, the optimizer's state included) and progress.json, so a run moves
between GPU and TPU sessions freely; only never train one model on both at once.

A TPU trains many times faster than two T4s, so it cannot tokenize a whole session's text before it starts. A feeder
process downloads and tokenizes while the TPU trains, in shards of `shard_steps` steps that mix the sources by their
`share`; each shard is read in a shuffled order without repeats, then deleted. The next session, on either machine,
starts at the files after the last one read (progress 'next_file').

    python kani_tpu.py train --settings settings.json    # kani_advanced.py and the tokenizer sit beside it
"""
import argparse
import functools
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')  # PyTorch only reads and writes checkpoints here: keep it off the TPU
import kani_advanced as ka

TPU_DEFAULTS = {
    'tpu_micro_batch': 8,    # sequences per chip at once; lower it if XLA runs out of memory
    'shard_steps': 250,      # training steps of text in each shard the feeder writes
    'shards_ahead': 4,       # the most shards tokenized ahead of training
    'remat': True,           # recompute attention in the backward pass instead of keeping it (saves most memory)
    'require_tpu': True,     # stop at once when JAX finds no TPU, instead of training on the CPU for hours
}
BETAS, EPS = (0.9, 0.95), 1e-8


def load_settings(path):
    return {**TPU_DEFAULTS, **ka.load_settings(path)}


# --- The feeder: a separate process that never touches the TPU.

class Source:
    """One source's documents as one stream of tokens, file after file from file number `start` of its list."""

    def __init__(self, source, files, start, pool, workers, folder):
        self.source, self.files, self.number = source, files, start - 1
        self.pool, self.workers, self.folder = pool, workers, folder
        self.encode = functools.partial(ka.encode_documents, kind=source.get('filter'))
        self.batches, self.path, self.rest = iter(()), None, np.zeros(0, dtype='<u2')

    def next_file(self):
        if self.path is not None and 'local' not in self.source:
            self.path.unlink(missing_ok=True)
        self.number += 1
        name = self.files[self.number % len(self.files)]
        print(f'feeder: reading {name}', flush=True)
        self.path = ka.fetch(self.source, name, self.folder)
        self.batches = ka.parquet_texts(self.path, 500, self.source.get('column', 'text'))

    def take(self, count):
        parts, size = [self.rest], len(self.rest)
        while size < count:
            group = list(itertools.islice(self.batches, 4 * self.workers))
            if not group:
                self.next_file()
                continue
            for tokens in self.pool.map(self.encode, group):
                parts.append(tokens)
                size += len(tokens)
        joined = np.concatenate(parts)
        self.rest = joined[count:]
        return joined[:count]


def feed(settings):
    """validation.bin, then shard-00000.bin, shard-00001.bin, ... in data_dir, for the job in feed.json. Every shard is
    exactly shard_steps steps of blocks (plus one token), each source's share of it in turn, so the same start always
    gives the same shards and a restarted session finds its place."""
    from multiprocessing import Pool
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    os.nice(10)  # the trainer's own work comes first
    data = Path(settings['data_dir'])
    job = json.loads((data / 'feed.json').read_text())
    lists, validation = ka.data_files(settings)
    shares = [source.get('share', 1.0) for source in settings['sources']]
    size = settings['shard_steps'] * settings['batch_tokens'] + 1
    quotas = [int(size * share / sum(shares)) for share in shares]
    quotas[0] += size - sum(quotas)
    workers = settings['workers'] or max(1, min(32, (os.cpu_count() or 2) - 2))  # each keeps its own word cache
    folder = str(data / 'downloads')
    print(f'feeder: {workers} tokenizer processes', flush=True)
    with Pool(workers, initializer=ka._start_worker, initargs=(settings['tokenizer'], job['contamination'])) as pool:
        ka.validation_tokens(settings, validation, pool, folder).tofile(data / 'validation.tmp')
        (data / 'validation.tmp').replace(data / 'validation.bin')
        sources = [Source(source, files, start, pool, workers, folder)
                   for source, files, start in zip(settings['sources'], lists, job['start'])]
        for shard in itertools.count():
            while shard >= job['first_shard'] and len(list(data.glob('shard-*.bin'))) >= settings['shards_ahead']:
                time.sleep(1)
            started = time.time()
            tokens = np.concatenate([source.take(quota) for source, quota in zip(sources, quotas)])
            if shard < job['first_shard']:
                continue
            name = f'shard-{shard:05d}'
            after = {ka.source_key(s): source.number + 1 for s, source in zip(settings['sources'], sources)}
            (data / f'{name}.json').write_text(json.dumps({'next_file': after}))
            tokens.tofile(data / f'{name}.tmp')
            (data / f'{name}.tmp').replace(data / f'{name}.bin')
            print(f'feeder: {name} ready ({len(tokens):,} tokens in {time.time() - started:.0f}s)', flush=True)


# --- The model and its training step, in JAX.

def build_steps(settings, mesh=None, dtype='bfloat16'):
    """forward(params, ids) -> logits, train_step(state, xs, ys, lr) -> (state, loss, gradient norm) and
    eval_step(params, x, y, weights) -> the weighted sum of each sequence's mean loss.

    The parameters keep kani_advanced.py's PyTorch names and layouts (matrices are output x input), with each block
    weight stacked over the layers (blocks.q.weight is layers x output x input) so that one lax.scan runs the layers:
    the backward pass then recomputes one layer's attention at a time. The residual stream, layer norms, softmax and
    loss stay float32; matrix products run in `dtype`. train_step is
    torch.optim.AdamW (betas 0.9/0.95, weight decay on matrices only) after clipping the gradient norm to 1, over
    xs/ys of [micro steps, sequences, context]."""
    import jax
    import jax.numpy as jnp
    width, heads, context = (settings[key] for key in ('width', 'heads', 'context'))
    head = width // heads
    cos, sin = (table.numpy() for table in ka.rope_tables(context, head))
    mask = np.tril(np.ones((context, context), dtype=bool))
    dtype = jnp.dtype(dtype)
    wide = jnp.promote_types(dtype, jnp.float32)  # float32, or float64 when the tests ask for it

    def norm(x, p, name):
        mean = x.mean(-1, keepdims=True)
        variance = jnp.square(x - mean).mean(-1, keepdims=True)
        return (x - mean) * jax.lax.rsqrt(variance + 1e-5) * p[name + '.weight'] + p[name + '.bias']

    def linear(x, weight):
        return jnp.einsum('...i,oi->...o', x.astype(dtype), weight.astype(dtype))

    def rotate(x, c, s):
        half = x.shape[-1] // 2
        return x * c + jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1) * s

    def block(x, p):
        b, t, _ = x.shape
        c, s = cos[:t, None, :], sin[:t, None, :]
        h = norm(x, p, 'norm1')
        q, k, v = (linear(h, p[name + '.weight']).reshape(b, t, heads, head) for name in ('q', 'k', 'v'))
        q, k = (rotate(z.astype(wide), c, s).astype(dtype) for z in (q, k))
        scores = jnp.einsum('bqhd,bkhd->bhqk', q, k, preferred_element_type=wide) / math.sqrt(head)
        weights = jax.nn.softmax(jnp.where(mask[:t, :t], scores, -jnp.inf), axis=-1)
        mixed = jnp.einsum('bhqk,bkhd->bqhd', weights.astype(dtype), v).reshape(b, t, width)
        x = x + linear(mixed, p['projection.weight']).astype(wide)
        up = jax.nn.gelu(linear(norm(x, p, 'norm2'), p['up.weight']).astype(wide), approximate=True)
        return x + linear(up, p['down.weight']).astype(wide)

    if settings['remat']:
        block = jax.checkpoint(block, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)

    def forward(params, ids):
        x = params['token.weight'][ids]
        blocks = {name[len('blocks.'):]: value for name, value in params.items() if name.startswith('blocks.')}
        x, _ = jax.lax.scan(lambda x, p: (block(x, p), None), x, blocks)
        return linear(norm(x, params, 'final'), params['token.weight']).astype(wide)

    def token_losses(params, x, y):
        logits = forward(params, x)
        return jax.nn.logsumexp(logits, axis=-1) - jnp.take_along_axis(logits, y[..., None], axis=-1)[..., 0]

    def train_step(state, xs, ys, lr):
        params = state['params']

        def micro(total, batch):
            x, y = batch
            if mesh is not None:
                x, y = (jax.lax.with_sharding_constraint(z, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('data', None)))
                        for z in (x, y))
            loss, grads = jax.value_and_grad(lambda p: token_losses(p, x, y).mean())(params)
            return jax.tree_util.tree_map(jnp.add, total, grads), loss

        grads, losses = jax.lax.scan(micro, jax.tree_util.tree_map(jnp.zeros_like, params), (xs, ys))
        grads = jax.tree_util.tree_map(lambda g: g / xs.shape[0], grads)
        total = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)))
        grads = jax.tree_util.tree_map(lambda g: g * jnp.minimum(1.0, 1.0 / (total + 1e-6)), grads)
        count = state['count'] + 1
        m = jax.tree_util.tree_map(lambda m, g: BETAS[0] * m + (1 - BETAS[0]) * g, state['m'], grads)
        v = jax.tree_util.tree_map(lambda v, g: BETAS[1] * v + (1 - BETAS[1]) * g * g, state['v'], grads)
        steps = count.astype(wide)
        step, root = lr / (1 - BETAS[0] ** steps), jnp.sqrt(1 - BETAS[1] ** steps)
        matrix = lambda name, p: p.ndim - name.startswith('blocks.') == 2  # weight decay on matrices only
        params = {name: (p * (1 - lr * settings['weight_decay']) if matrix(name, p) else p)
                        - step * m[name] / (jnp.sqrt(v[name]) / root + EPS) for name, p in params.items()}
        return {'params': params, 'm': m, 'v': v, 'count': count}, losses.mean(), total

    def eval_step(params, x, y, weights):
        return jnp.sum(token_losses(params, x, y).mean(-1) * weights)

    return forward, train_step, eval_step


# --- PyTorch checkpoints, shared with kani_advanced.py.

def optimizer(net, settings):
    """kani_advanced.py's AdamW groups (weight decay on matrices), on the CPU: only to read and write its state."""
    import torch
    decay = [p for p in net.parameters() if p.dim() == 2]
    no_decay = [p for p in net.parameters() if p.dim() < 2]
    return torch.optim.AdamW([{'params': decay, 'weight_decay': settings['weight_decay']}, {'params': no_decay, 'weight_decay': 0.0}],
                             lr=settings['lr'], betas=BETAS, eps=EPS)


def restore(settings, tokenizer, checkpoint):
    """The model (PyTorch, CPU), its optimizer, the fp16 loss scaler's state (kept for the GPUs) and progress, from
    checkpoint.pt, or new ones as kani_advanced.py makes them."""
    import torch
    torch.manual_seed(settings['seed'])
    net = ka.build(settings, tokenizer.vocab_size)
    opt = optimizer(net, settings)
    scaler = {'scale': 65536.0, 'growth_factor': 2.0, 'backoff_factor': 0.5, 'growth_interval': 2000, '_growth_tracker': 0}
    progress = {'session': 0, 'tokens_seen': 0, 'step': 0, 'history': []}
    if checkpoint:
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        for key in ('width', 'layers', 'heads', 'context'):
            if state['settings'][key] != settings[key]:
                raise SystemExit(f'{checkpoint} has {key}={state["settings"][key]}; keep the model settings or use a new run_dir')
        if state['net']['token.weight'].shape[0] != tokenizer.vocab_size:
            raise SystemExit(f'{checkpoint} has a {state["net"]["token.weight"].shape[0]:,}-token vocabulary, the tokenizer '
                             f'{tokenizer.vocab_size:,}: run kani_advanced.py retokenize first')
        net.load_state_dict(state['net'])
        if state['opt'] is not None:
            opt.load_state_dict(state['opt'])
            scaler = state['scaler'] or scaler
        progress = state['progress']
    return net, opt, scaler, progress


def stacked(values, layers):
    """PyTorch's per-layer names (blocks.3.q.weight) to one array over all layers per name (blocks.q.weight)."""
    out = {name: value for name, value in values.items() if not name.startswith('blocks.')}
    for name in sorted({name.split('.', 2)[2] for name in values if name.startswith('blocks.')}):
        out['blocks.' + name] = np.stack([values[f'blocks.{i}.{name}'] for i in range(layers)])
    return out


def unstacked(values):
    out = {}
    for name, value in values.items():
        if name.startswith('blocks.'):
            out.update({f'blocks.{i}.{name[len("blocks."):]}': layer for i, layer in enumerate(value)})
        else:
            out[name] = value
    return out


def to_jax_state(net, opt):
    params = {name: p.detach().numpy().copy() for name, p in net.named_parameters()}
    moments = {key: {name: (opt.state[p][key].numpy().copy() if p in opt.state else np.zeros_like(params[name]))
                     for name, p in net.named_parameters()} for key in ('exp_avg', 'exp_avg_sq')}
    first = next(net.parameters())
    count = float(opt.state[first]['step']) if first in opt.state else 0.0
    layers = len(net.blocks)
    return {'params': stacked(params, layers), 'm': stacked(moments['exp_avg'], layers), 'v': stacked(moments['exp_avg_sq'], layers),
            'count': np.float32(count)}


def from_jax_state(net, opt, state):
    """Copy a (host) JAX state into the PyTorch model and optimizer."""
    import torch
    params, m, v = (unstacked({name: np.asarray(value, dtype=np.float32) for name, value in state[key].items()})
                    for key in ('params', 'm', 'v'))
    with torch.no_grad():
        for name, p in net.named_parameters():
            p.copy_(torch.from_numpy(params[name]))
            opt.state[p] = {'step': torch.tensor(float(state['count'])), 'exp_avg': torch.from_numpy(m[name].copy()),
                            'exp_avg_sq': torch.from_numpy(v[name].copy())}


def save_checkpoint(settings, net, opt, scaler, progress):
    import torch
    run = Path(settings['run_dir'])
    torch.save({'net': net.state_dict(), 'opt': opt.state_dict(), 'scaler': scaler, 'progress': progress, 'settings': settings},
               run / 'checkpoint.tmp')
    (run / 'checkpoint.tmp').replace(run / 'checkpoint.pt')
    (run / 'progress.json').write_text(json.dumps(progress, indent=1))


# --- Training.

def train(settings):
    started = time.time()
    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec
    devices = jax.devices()
    machine = f'{len(devices)} x {devices[0].device_kind}'
    if settings['require_tpu'] and devices[0].platform != 'tpu':
        raise SystemExit(f'No TPU found (JAX sees {machine}): set Settings → Accelerator → TPU v5e-8.')
    mesh = Mesh(np.array(devices), ('data',))
    replicated = NamedSharding(mesh, PartitionSpec())
    split, rows = NamedSharding(mesh, PartitionSpec(None, 'data', None)), NamedSharding(mesh, PartitionSpec('data', None))
    context = settings['context']
    sequences, micro = settings['batch_tokens'] // context, len(devices) * settings['tpu_micro_batch']
    if settings['batch_tokens'] % context or sequences % micro:
        raise SystemExit(f'batch_tokens / context = {sequences} sequences a step must be a multiple of chips x tpu_micro_batch = {micro}')
    accumulate = sequences // micro

    tokenizer = ka.Tokenizer.load(settings['tokenizer'])
    run, data = Path(settings['run_dir']), Path(settings['data_dir'])
    checkpoint = ka.find_file(settings, 'checkpoint.pt')
    print('attached inputs:', ', '.join(ka.attached(settings)[:20]), flush=True)
    if checkpoint:
        print(f'continuing from {checkpoint}', flush=True)
    elif settings['continue_from_previous']:
        raise SystemExit('No earlier checkpoint.pt found, so this session would start over from zero. Attach the latest '
                         'output of this notebook or the GPU one (Add Input → Your Work → Notebooks), or set '
                         'continue_from_previous to False to begin a new run.')
    banned = ka.find_file(settings, 'contamination.npy')
    run.mkdir(parents=True, exist_ok=True)
    if banned:
        print(f'dropping documents that share 13 words with the test sets ({len(np.load(banned)):,} runs, {banned})', flush=True)
        if banned.parent != run:
            banned = Path(shutil.copy(banned, run / 'contamination.npy'))
    elif settings['decontaminate']:
        raise SystemExit('decontaminate is on but no contamination.npy was found: attach the dataset that holds it '
                         '(made by "kani_advanced.py grams") beside the checkpoint.')
    net, opt, scaler, progress = restore(settings, tokenizer, checkpoint)
    session = progress['session']

    # Where this session's text starts. A session this trainer began and was cut off resumes at its next unread
    # step; anything else (a new session, or one a GPU began) starts its own text from the first shard.
    start = [ka.start_file(source, session, progress.get('next_file', {})) for source in settings['sources']]
    session_start = progress['history'][-1]['step'] if progress.get('history') else 0
    resumed = progress.get('feed') == {'session': session, 'start': start}
    first_shard, within = divmod(progress['step'] - session_start, settings['shard_steps']) if resumed else (0, 0)
    progress['feed'] = {'session': session, 'start': start}
    data.mkdir(parents=True, exist_ok=True)
    for old in itertools.chain(data.glob('shard-*'), data.glob('validation.*')):
        old.unlink()
    (data / 'settings.json').write_text(json.dumps(settings))
    (data / 'feed.json').write_text(json.dumps({'start': start, 'first_shard': first_shard, 'contamination': banned and str(banned)}))
    feeder = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'feed', '--settings', str(data / 'settings.json')])

    def wait_for(path):
        waited = time.time()
        while not path.exists():
            if feeder.poll() is not None:
                raise SystemExit(f'The feeder stopped (exit code {feeder.returncode}) before {path.name}: see its message above.')
            time.sleep(0.5)
        return time.time() - waited

    def open_shard(index):
        waited = wait_for(data / f'shard-{index:05d}.bin')
        tokens = np.memmap(data / f'shard-{index:05d}.bin', dtype='<u2', mode='r')
        order = np.random.default_rng([settings['seed'], session, index]).permutation(settings['shard_steps'] * sequences)
        return tokens, order, json.loads((data / f'shard-{index:05d}.json').read_text()), waited

    forward, train_step, eval_step = build_steps(settings, mesh)
    step_fn = jax.jit(train_step, donate_argnums=0, out_shardings=(replicated, replicated, replicated))
    eval_fn = jax.jit(eval_step, out_shardings=replicated)
    state = jax.device_put(to_jax_state(net, opt), replicated)
    print(json.dumps({'parameters': sum(p.numel() for p in net.parameters()), 'machine': machine, 'session': session,
                      'step': progress['step'], 'tokens_seen': progress['tokens_seen'], 'tokens_per_step': settings['batch_tokens'],
                      'accumulate': accumulate, 'resumed_at_shard': first_shard if resumed else None}), flush=True)

    # The same held-out windows kani_advanced.py's validate() scores, so the two report comparable losses.
    wait_for(data / 'validation.bin')
    held_out = np.fromfile(data / 'validation.bin', dtype='<u2')
    rng = np.random.default_rng(0)
    starts = np.concatenate([rng.integers(0, len(held_out) - context - 1, size=settings['micro_batch'])
                             for _ in range(settings['eval_batches'])])
    windows = np.stack([held_out[s:s + context + 1] for s in starts]).astype(np.int32)
    pad = -len(windows) % micro
    windows = np.concatenate([windows, windows[:pad]])
    weights = np.concatenate([np.ones(len(starts)), np.zeros(pad)]).astype(np.float32)

    def validate(params):
        total = sum(eval_fn(params, jax.device_put(windows[i:i + micro, :-1], rows), jax.device_put(windows[i:i + micro, 1:], rows),
                            jax.device_put(weights[i:i + micro], NamedSharding(mesh, PartitionSpec('data'))))
                    for i in range(0, len(windows), micro))
        return float(total) / len(starts)

    def save(state):
        from_jax_state(net, opt, jax.device_get(state))
        save_checkpoint(settings, net, opt, scaler, progress)

    index, (tokens, order, meta, waited) = first_shard, open_shard(first_shard)
    budget, last_checkpoint, stopping = settings['hours'] * 3600, time.time(), False
    losses, count, window, validation_loss, first_status = [], 0, time.time(), None, True
    print('compiling the training step (a minute or two)', flush=True)
    while not stopping:
        if within == settings['shard_steps']:
            for name in (f'shard-{index:05d}.bin', f'shard-{index:05d}.json'):
                (data / name).unlink(missing_ok=True)
            index, within = index + 1, 0
            tokens, order, meta, wait = open_shard(index)
            waited += wait
        blocks = order[within * sequences:(within + 1) * sequences]
        batch = tokens[(blocks[:, None] * context + np.arange(context + 1)).ravel()].reshape(accumulate, micro, context + 1).astype(np.int32)
        lr = ka.learning_rate(settings, progress['tokens_seen'], progress.get('rewarm_from'))
        state, loss, _ = step_fn(state, jax.device_put(batch[..., :-1], split), jax.device_put(batch[..., 1:], split), np.float32(lr))
        losses.append(loss)
        within, count = within + 1, count + 1
        progress['step'] += 1
        progress['tokens_seen'] += settings['batch_tokens']
        stopping = time.time() - started > budget or progress['tokens_seen'] >= settings['target_tokens']
        if progress['step'] % settings['eval_every'] == 0 or stopping:
            validation_loss = validate(state['params'])
        if progress['step'] % 20 == 0 or stopping:
            loss = float(np.mean(jax.device_get(losses)))
            status = {'step': progress['step'], 'tokens': f"{progress['tokens_seen'] / 1e9:.3f}B", 'loss': round(loss, 4),
                      'validation_loss': validation_loss and round(validation_loss, 4), 'lr': f'{lr:.2e}',
                      'tokens_per_second': round(count * settings['batch_tokens'] / (time.time() - window)),
                      'waiting_for_text_seconds': round(waited), 'hours_left': round(max(0.0, budget - (time.time() - started)) / 3600, 2)}
            memory = devices[0].memory_stats() if first_status and hasattr(devices[0], 'memory_stats') else None
            if memory and 'peak_bytes_in_use' in memory:
                status['tpu_memory_gb'] = round(memory['peak_bytes_in_use'] / 2 ** 30, 1)
            first_status = False
            print(json.dumps(status), flush=True)
            (run / 'status.json').write_text(json.dumps({**status, 'state': 'running', 'updated': time.time()}))
            losses, count, window = [], 0, time.time()
        if time.time() - last_checkpoint > settings['checkpoint_minutes'] * 60:
            save(state)
            last_checkpoint = time.time()
            print(f'checkpoint at step {progress["step"]:,}', flush=True)

    feeder.terminate()
    progress['history'].append({'session': session, 'step': progress['step'], 'tokens_seen': progress['tokens_seen'],
                                'validation_loss': validation_loss, 'hours': round((time.time() - started) / 3600, 2), 'machine': machine})
    progress['next_file'] = meta['next_file']
    progress.pop('feed')
    progress['session'] += 1
    save(state)
    ka.export(net, settings, tokenizer, progress, run / 'kani-advanced.npz')
    complete = progress['tokens_seen'] >= settings['target_tokens']
    (run / 'status.json').write_text(json.dumps({'step': progress['step'], 'tokens_seen': progress['tokens_seen'],
                                                 'validation_loss': validation_loss, 'state': 'complete' if complete else 'stopped'}))
    net.eval()
    samples = {prompt: ka.generate(net, tokenizer, prompt) for prompt in (ka.PROMPTS_MULTILINGUAL if tokenizer.version == 3 else ka.PROMPTS)}
    (run / 'samples.json').write_text(json.dumps(samples, indent=1, ensure_ascii=False))
    for text in samples.values():
        print(f'--- {text}', flush=True)
    print(('target reached: ' if complete else 'session done: ') + f'{progress["tokens_seen"] / 1e9:.2f}B of '
          f'{settings["target_tokens"] / 1e9:.1f}B tokens, validation loss {validation_loss:.4f}', flush=True)
    feeder.wait()


def main():
    if 'ipykernel' in sys.modules:
        print('This is the TPU training script, not the notebook: run it with "!python kani_tpu.py train".')
        return
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['train', 'feed'])
    parser.add_argument('--settings', default='settings.json')
    args = parser.parse_args()
    settings = load_settings(args.settings)
    if args.command == 'train':
        train(settings)
    else:
        feed(settings)


if __name__ == '__main__':
    main()
