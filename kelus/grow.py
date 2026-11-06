"""Deepen our own Kani checkpoint with initially identity residual blocks.

    python3 -m kelus.grow --source parent.npz --out grown.npz --layers 24 --context 2048

Preserves existing next-token predictions at initialization. Added capacity is
untrained; this does not establish stronger capabilities or GPT-4 equivalence.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from .model import Model
from .tensor import no_grad


def expand(model, layers, context, seed=2041):
    old = model.config
    if layers <= old.layers or layers % old.layers:
        raise ValueError('New layer count must be a larger integer multiple of the old count')
    if context < old.context or (context != old.context and old.positions != 'rope'):
        raise ValueError('Context expansion requires RoPE and must not shorten the context')
    grown = Model(replace(old, layers=layers, context=context, seed=seed), model.tokenizer)
    stride = layers // old.layers
    for name, parameter in model.parameters.items():
        if name.startswith('block'):
            block, suffix = name.split('.', 1)
            destination = f'block{int(block[5:]) * stride}.{suffix}'
        else:
            destination = name
        grown.parameters[destination].data[:] = parameter.data
    for layer in range(layers):
        if layer % stride:


            grown.parameters[f'block{layer}.projection'].data.fill(0)
            grown.parameters[f'block{layer}.down'].data.fill(0)
    return grown


def grow_checkpoint(source, out, layers=24, context=2048):
    source, out = Path(source), Path(out)
    if source.resolve() == out.resolve() or out.exists():
        raise ValueError('Output must be a new checkpoint, separate from the source')
    model, header, states = Model.load(source)
    del states
    grown = expand(model, layers, context)
    ids = np.random.default_rng(405).integers(0, model.config.vocab_size, (1, 16))
    with no_grad():
        before = model(ids).data
        after = grown(ids).data
    np.testing.assert_allclose(before, after, atol=2e-5, rtol=2e-5)
    with source.open('rb') as stream:
        source_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    metadata = {'stage': 'growth-initialization', 'steps': 0,
                'parent_steps': header.get('metadata', {}).get('steps', 0),
                'parent_sha256': source_hash, 'parent_parameters': model.parameter_count,
                'parameters': grown.parameter_count, 'parent_layers': model.config.layers,
                'initialization': 'Our trained Kani weights plus identity residual blocks; fresh optimizer required',
                'initial_logits_max_error': float(np.max(np.abs(before - after))),
                'release_status': 'Untrained added layers; capability evaluation required'}
    grown.save(out, metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--layers', type=int, default=24)
    parser.add_argument('--context', type=int, default=2048)
    args = parser.parse_args()
    print(json.dumps(grow_checkpoint(args.source, args.out, args.layers, args.context), indent=2))


if __name__ == '__main__':
    main()
