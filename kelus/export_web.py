"""Export a B1 checkpoint for the browser engine in site/b1.js.

Writes <out>/b1.json (config, tokenizer, tensor layout, training summary) and
<out>/b1.bin (little-endian float32 weights in that layout). With --int8, every matrix is stored as int8 with a
float32 scale per output channel (per row for the token table), after the float32 tensors: about 4x smaller. Optimizer moments
are not exported. The browser runs inference only; nothing is sent anywhere.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .model import Model


def quantize(matrix, axis):
    """int8 values and float32 scales, one scale per slice along `axis` (absolute maximum maps to 127)."""
    scale = np.abs(matrix).max(axis=axis) / 127.0
    scale[scale == 0] = 1.0
    values = np.clip(np.rint(matrix / np.expand_dims(scale, axis)), -127, 127).astype(np.int8)
    return values, scale.astype("<f4")


def export(checkpoint, out, int8=False):
    model, header, _ = Model.load(checkpoint)
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    tensors, chunks, offset, quantized = [], [], 0, []
    for name, parameter in model.parameters.items():
        data = np.ascontiguousarray(parameter.data, dtype="<f4")
        if int8 and data.ndim == 2:


            axis = 1 if name in ("token", "position") else 0
            values, scale = quantize(data, axis)
            quantized.append((name, list(data.shape), values, axis, offset))
            data = scale
            name = name + ".scale"
        tensors.append({"name": name, "shape": list(data.shape), "offset": offset})
        chunks.append(data.tobytes())
        offset += data.size

    position = offset * 4
    for name, shape, values, axis, scale_offset in quantized:
        tensors.append({"name": name, "shape": shape, "dtype": "int8", "byte_offset": position, "scale": scale_offset, "scale_axis": axis})
        chunks.append(values.tobytes())
        position += values.size
    chunks.append(b"\0" * (-position % 4))
    weights = b"".join(chunks)
    metadata = header.get("metadata", {})
    manifest = {
        "format": "kelus-b1-web", "version": 1, "name": metadata.get('display_name', model.name), "codename": metadata.get("codename", model.codename),
        "config": header["config"] | {"positions": model.config.positions, "tie_embeddings": model.config.tie_embeddings},
        "tokenizer": model.tokenizer.to_dict(), "tensors": tensors, "parameters": model.parameter_count,
        "training": {key: metadata.get(key) for key in ("steps", "validation_loss", "training_examples", "dataset_tokens")},
        "language_training": metadata.get("language_training"),
        "weights_sha256": hashlib.sha256(weights).hexdigest(),
        "source_checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
    }
    if metadata.get('specialist'):
        manifest['specialist'] = metadata['specialist']
    if int8:
        manifest['quantization'] = 'int8-per-channel'
        manifest['bytes'] = len(weights)
    (destination / "b1.bin").write_bytes(weights)
    (destination / "b1.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="site/model")
    parser.add_argument("--int8", action="store_true", help="8-bit matrices with per-channel scales (about 4x smaller)")
    args = parser.parse_args()
    manifest = export(args.checkpoint, args.out, int8=args.int8)
    print(json.dumps({k: manifest[k] for k in ("name", "parameters", "config", "training", "weights_sha256")}))


if __name__ == "__main__":
    main()
