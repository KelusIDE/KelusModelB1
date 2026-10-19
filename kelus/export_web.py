"""Export a B1 checkpoint for the browser engine in site/b1.js.

Writes <out>/b1.json (config, tokenizer, tensor layout, training summary) and
<out>/b1.bin (little-endian float32 weights in that layout). Optimizer moments
are not exported. The browser runs inference only; nothing is sent anywhere.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .model import Model


def export(checkpoint, out):
    model, header, _ = Model.load(checkpoint)
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    tensors, chunks, offset = [], [], 0
    for name, parameter in model.parameters.items():
        data = np.ascontiguousarray(parameter.data, dtype="<f4")
        tensors.append({"name": name, "shape": list(data.shape), "offset": offset})
        chunks.append(data.tobytes())
        offset += data.size
    weights = b"".join(chunks)
    metadata = header.get("metadata", {})
    manifest = {
        "format": "kelus-b1-web", "version": 1, "name": model.name,
        "config": header["config"] | {"positions": model.config.positions, "tie_embeddings": model.config.tie_embeddings},
        "tokenizer": model.tokenizer.to_dict(), "tensors": tensors, "parameters": model.parameter_count,
        "training": {key: metadata.get(key) for key in ("steps", "validation_loss", "training_examples", "dataset_tokens")},
        "language_training": metadata.get("language_training"),
        "weights_sha256": hashlib.sha256(weights).hexdigest(),
        "source_checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
    }
    (destination / "b1.bin").write_bytes(weights)
    (destination / "b1.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="site/model")
    args = parser.parse_args()
    manifest = export(args.checkpoint, args.out)
    print(json.dumps({k: manifest[k] for k in ("name", "parameters", "config", "training", "weights_sha256")}))


if __name__ == "__main__":
    main()
