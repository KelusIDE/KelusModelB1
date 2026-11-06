"""Train Kani models on the Mac's GPU with MLX, Apple's open-source array library.

The architecture, weight names, initialization, loss, AdamW and gradient clipping are the same as
kelus.model and kelus.optim, so a GPU-trained checkpoint is an ordinary Kani .npz: the NumPy engine,
the website (site/b1.js) and the CLI run it unchanged, and kelus.train can resume it. MLX supplies
GPU arrays and automatic gradients only; every model still starts from random weights.

    .venv/bin/python -m kelus.gpu check                    # GPU vs NumPy: loss, gradients, one AdamW step
    .venv/bin/python -m kelus.gpu bench --width 320 --layers 6
"""
import argparse
import json
import math
import time

import mlx.core as mx
import numpy as np

from .model import Config, Model


def limit_memory(gigabytes):
    """Cap this process's GPU memory. MLX's default allows ~19 GB on a 24 GB Mac, so two GPU jobs (plus the
    editor and browser) can exhaust memory and macOS stops apps. Run one GPU job at a time under a cap."""
    mx.set_memory_limit(int(gigabytes * 2 ** 30))
    mx.set_cache_limit(int(min(gigabytes / 4, 2) * 2 ** 30))


def parameters(model):
    """The NumPy model's weights as GPU arrays."""
    return {name: mx.array(parameter.data) for name, parameter in model.parameters.items()}


def rope_tables(model):
    if model.config.positions != 'rope':
        return None, None
    return mx.array(model.rope_cos), mx.array(model.rope_sin)


def _norm(x, gain, bias):
    centered = x - mx.mean(x, axis=-1, keepdims=True)
    return centered * mx.rsqrt(mx.mean(centered * centered, axis=-1, keepdims=True) + 1e-5) * gain + bias


def _rotate(x, cos, sin):
    half = x.shape[-1] // 2
    return x * cos + mx.concatenate([-x[..., half:], x[..., :half]], axis=-1) * sin


def forward(params, ids, config, cos=None, sin=None):
    """Logits [batch, time, vocab], the same computation as kelus.model.Model.__call__ (no dropout)."""
    batch, time = ids.shape
    d, heads = config.width, config.heads
    width = d // heads
    x = params['token'][ids]
    if config.positions == 'learned':
        x = x + params['position'][:time]
    mask = mx.triu(mx.full((time, time), -1e9, dtype=mx.float32), k=1)
    if cos is not None:
        cos, sin = cos[:time], sin[:time]
    for layer in range(config.layers):
        prefix = f'block{layer}.'
        normalized = _norm(x, params[prefix + 'norm1.gain'], params[prefix + 'norm1.bias'])
        q, k, v = ((normalized @ params[prefix + name]).reshape(batch, time, heads, width).transpose(0, 2, 1, 3) for name in 'qkv')
        if cos is not None:
            q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)
        attention = mx.softmax((q @ k.transpose(0, 1, 3, 2)) * (width ** -0.5) + mask, axis=-1)
        x = x + (attention @ v).transpose(0, 2, 1, 3).reshape(batch, time, d) @ params[prefix + 'projection']
        hidden = _norm(x, params[prefix + 'norm2.gain'], params[prefix + 'norm2.bias']) @ params[prefix + 'up']
        x = x + 0.5 * hidden * (1 + mx.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * hidden ** 3))) @ params[prefix + 'down']
    output = params['token'].T if config.tie_embeddings else params['output']
    return _norm(x, params['final.gain'], params['final.bias']) @ output


def _fast_block(weights, x, heads):
    batch, time, d = x.shape
    width = d // heads
    normalized = mx.fast.layer_norm(x, weights['norm1.gain'], weights['norm1.bias'], 1e-5)
    q, k, v = ((normalized @ weights[name]).reshape(batch, time, heads, width).transpose(0, 2, 1, 3) for name in 'qkv')
    q = mx.fast.rope(q, width, traditional=False, base=10000.0, scale=1.0, offset=0)
    k = mx.fast.rope(k, width, traditional=False, base=10000.0, scale=1.0, offset=0)
    mixed = mx.fast.scaled_dot_product_attention(q, k, v, scale=width ** -0.5, mask='causal')
    x = x + mixed.transpose(0, 2, 1, 3).reshape(batch, time, d) @ weights['projection']
    hidden = mx.fast.layer_norm(x, weights['norm2.gain'], weights['norm2.bias'], 1e-5) @ weights['up']
    return x + 0.5 * hidden * (1 + mx.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * hidden ** 3))) @ weights['down']


def forward_fast(params, ids, config, cos=None, sin=None, checkpoint_blocks=False):
    """forward() with MLX's fused GPU kernels for layer norm, rotary positions and causal attention.

    The same math: layer norm with biased variance and eps 1e-5, half-split rotation with base 10000
    over the whole head width, softmax(q k^T / sqrt(width)) with a causal mask. Learned positions use forward().
    """
    if config.positions != 'rope':
        return forward(params, ids, config, cos, sin)
    batch, time = ids.shape
    d, heads = config.width, config.heads
    width = d // heads
    x = params['token'][ids]
    block_fn = mx.checkpoint(lambda weights, value: _fast_block(weights, value, heads)) if checkpoint_blocks else None
    for layer in range(config.layers):
        prefix = f'block{layer}.'
        if block_fn is not None:
            weights = {name[len(prefix):]: value for name, value in params.items() if name.startswith(prefix)}
            x = block_fn(weights, x)
            continue
        normalized = mx.fast.layer_norm(x, params[prefix + 'norm1.gain'], params[prefix + 'norm1.bias'], 1e-5)
        q, k, v = ((normalized @ params[prefix + name]).reshape(batch, time, heads, width).transpose(0, 2, 1, 3) for name in 'qkv')
        q = mx.fast.rope(q, width, traditional=False, base=10000.0, scale=1.0, offset=0)
        k = mx.fast.rope(k, width, traditional=False, base=10000.0, scale=1.0, offset=0)
        mixed = mx.fast.scaled_dot_product_attention(q, k, v, scale=width ** -0.5, mask='causal')
        x = x + mixed.transpose(0, 2, 1, 3).reshape(batch, time, d) @ params[prefix + 'projection']
        hidden = mx.fast.layer_norm(x, params[prefix + 'norm2.gain'], params[prefix + 'norm2.bias'], 1e-5) @ params[prefix + 'up']
        x = x + 0.5 * hidden * (1 + mx.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * hidden ** 3))) @ params[prefix + 'down']
    output = params['token'].T if config.tie_embeddings else params['output']
    return mx.fast.layer_norm(x, params['final.gain'], params['final.bias'], 1e-5) @ output


def loss(params, x, y, config, cos=None, sin=None, fast=False, checkpoint_blocks=False):
    """Mean next-token cross-entropy; targets of -1 (prompt, padding) are ignored."""
    logits = forward_fast(params, x, config, cos, sin, checkpoint_blocks) if fast else forward(params, x, config, cos, sin)
    flat, labels = logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
    valid = (labels >= 0).astype(mx.float32)
    picked = mx.take_along_axis(flat, mx.maximum(labels, 0)[:, None], axis=-1)[:, 0]
    return mx.sum((mx.logsumexp(flat, axis=-1) - picked) * valid) / mx.maximum(mx.sum(valid), 1)


class AdamW:
    """kelus.optim.AdamW on GPU arrays: global-norm clipping, bias correction, decay on matrices only."""

    def __init__(self, params, weight_decay=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, max_norm=1.0):
        self.weight_decay, self.beta1, self.beta2, self.epsilon, self.max_norm = weight_decay, beta1, beta2, epsilon, max_norm
        self.m = {name: mx.zeros_like(value) for name, value in params.items()}
        self.v = {name: mx.zeros_like(value) for name, value in params.items()}
        self.step_count = 0

    def update(self, params, grads, learning_rate):
        norm = mx.sqrt(sum(mx.sum(g * g) for g in grads.values()))
        factor = mx.minimum(1.0, self.max_norm / mx.maximum(norm, 1e-12))
        self.step_count += 1
        correct1, correct2 = 1 - self.beta1 ** self.step_count, 1 - self.beta2 ** self.step_count
        updated = {}
        for name, value in params.items():
            gradient = grads[name] * factor
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * gradient
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * gradient * gradient
            if value.ndim >= 2:
                value = value * (1 - learning_rate * self.weight_decay)
            updated[name] = value - learning_rate * (self.m[name] / correct1) / (mx.sqrt(self.v[name] / correct2) + self.epsilon)
        return updated, norm


    def metadata(self, learning_rate=0.0):
        return {'learning_rate': learning_rate, 'weight_decay': self.weight_decay, 'beta1': self.beta1, 'beta2': self.beta2,
                'epsilon': self.epsilon, 'step_count': self.step_count}

    def arrays(self):
        return {**{'adam.m.' + n: np.array(a) for n, a in self.m.items()}, **{'adam.v.' + n: np.array(a) for n, a in self.v.items()}}


def save(model, params, optimizer, path, metadata, learning_rate, *, weights_only=False, compression_level=6):
    """Write GPU weights as an ordinary Kani checkpoint (kelus.model.Model.save)."""
    for name, value in params.items():
        model.parameters[name].data = np.array(value, dtype=np.float32)

    class State:
        def metadata(self):
            return optimizer.metadata(learning_rate)

        def arrays(self):
            return optimizer.arrays()
    model.save(path, metadata, None if weights_only else State(), compression_level=compression_level)


def check():
    """GPU and NumPy agree on loss, gradients and one AdamW step for the same weights and batch."""
    from .optim import AdamW as NumpyAdamW
    from .tensor import cross_entropy
    from .tokenizer import BPETokenizer
    tokenizer = BPETokenizer.train(['the quick brown fox jumps over the lazy dog'] * 20, 300)
    model = Model(Config(context=64, width=64, heads=4, layers=2, seed=3, vocab_size=tokenizer.vocab_size, positions='rope', tie_embeddings=True), tokenizer)
    rng = np.random.default_rng(0)
    x = rng.integers(0, tokenizer.vocab_size, size=(3, 40))
    y = rng.integers(0, tokenizer.vocab_size, size=(3, 40))
    y[:, :5] = -1
    params, (cos, sin) = parameters(model), rope_tables(model)
    gpu_loss, grads = mx.value_and_grad(loss)(params, mx.array(x), mx.array(y), model.config, cos, sin)
    fast_loss, fast_grads = mx.value_and_grad(loss)(params, mx.array(x), mx.array(y), model.config, cos, sin, True)
    fast_worst = max(float(mx.max(mx.abs(fast_grads[n] - grads[n]))) / (float(mx.max(mx.abs(grads[n]))) + 1e-12) for n in grads)
    numpy_loss = cross_entropy(model(x), y)
    numpy_loss.backward()
    worst = max(float(np.max(np.abs(np.array(grads[n]) - p.grad))) / (float(np.max(np.abs(p.grad))) + 1e-12) for n, p in model.parameters.items())
    gpu_opt, numpy_opt = AdamW(params), NumpyAdamW(model.parameters, learning_rate=1e-3)
    updated, _ = gpu_opt.update(params, grads, 1e-3)
    numpy_opt.step()
    step_error = max(float(np.max(np.abs(np.array(updated[n]) - p.data))) for n, p in model.parameters.items())
    result = {'loss_gpu': round(float(gpu_loss), 6), 'loss_numpy': round(float(numpy_loss.data), 6),
              'worst_relative_gradient_error': worst, 'worst_weight_error_after_step': step_error,
              'fused_loss': round(float(fast_loss), 6), 'fused_worst_relative_gradient_error': fast_worst}


    result['match'] = (abs(result['loss_gpu'] - result['loss_numpy']) < 1e-4 and worst < 1e-3 and step_error < 5e-5
                       and abs(result['fused_loss'] - result['loss_numpy']) < 1e-4 and fast_worst < 1e-3)
    return result


def bench(width=320, layers=6, heads=8, context=1024, batch=4, steps=6):
    """Training steps per second for a random-weight model of this size, with Kani's 4,096-token vocabulary."""
    from pathlib import Path
    from .tokenizer import from_dict
    tokenizer = from_dict(json.loads((Path(__file__).resolve().parent.parent / 'data/foundation/tokenizer.json').read_text()))
    vocab = tokenizer.vocab_size
    config = Config(context=context, width=width, heads=heads, layers=layers, seed=1, vocab_size=vocab, positions='rope', tie_embeddings=True)
    model = Model(config, tokenizer)
    params, (cos, sin) = parameters(model), rope_tables(model)
    optimizer = AdamW(params)
    step = mx.compile(lambda p, x, y: mx.value_and_grad(loss)(p, x, y, config, cos, sin, True))
    x = mx.array(np.random.default_rng(0).integers(0, vocab, size=(batch, context)))
    timings = []
    for i in range(steps):
        started = time.perf_counter()
        value, grads = step(params, x, x)
        params, _ = optimizer.update(params, grads, 1e-4)
        mx.eval(params, optimizer.m, optimizer.v, value)
        timings.append(time.perf_counter() - started)
    seconds = float(np.median(timings[2:]))
    count = sum(int(np.prod(v.shape)) for v in params.values())
    return {'parameters': count, 'tokens_per_step': batch * context, 'seconds_per_step': round(seconds, 3),
            'tokens_per_second': round(batch * context / seconds), 'loss': round(float(value), 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('check')
    command = commands.add_parser('bench')
    for name, default in (('width', 320), ('layers', 6), ('heads', 8), ('context', 1024), ('batch', 4)):
        command.add_argument(f'--{name}', type=int, default=default)
    args = parser.parse_args()
    if args.command == 'check':
        print(json.dumps(check()))
    else:
        print(json.dumps(bench(args.width, args.layers, args.heads, args.context, args.batch)))


if __name__ == '__main__':
    main()
