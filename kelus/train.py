"""Train or resume B1 from randomly initialized weights using only local data."""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from .data import Dataset
from .model import Config, Model
from .optim import AdamW
from .tensor import cross_entropy, no_grad
from .tokenizer import BPETokenizer, ByteTokenizer


def evaluate(model, dataset, batch_size, seed=None):
    total, count = 0.0, 0
    with no_grad():
        for x, y in dataset.validation_batches(batch_size, model.config.context):
            targets = int((y >= 0).sum())
            total += float(cross_entropy(model(x), y).data) * targets
            count += targets
    return total / count


def training_texts(filename, seed):
    """Vocabulary learning sees exactly the training records, never validation."""
    records, seen = [], set()
    for line in Path(filename).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = json.dumps(record["messages"], sort_keys=True)
        if key not in seen:
            records.append(record); seen.add(key)
    if any(record.get("split") for record in records):
        selected = [record for record in records if record.get("split") == "train"]
    else:
        order = np.random.default_rng(seed).permutation(len(records))
        selected = [records[i] for i in order[max(1, int(len(records) * 0.15)):]]
    return [message["content"] for record in selected for message in record["messages"]]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train Kelus Model B1 from scratch. No pretrained weights or AI frameworks.")
    parser.add_argument("--data", default="data/conversations-v3.jsonl")
    parser.add_argument("--out", default="checkpoints/b1-v3.npz")
    parser.add_argument("--resume", help="Resume a B1 checkpoint, including Adam state")
    parser.add_argument("--init-from", help="Start a separate curriculum from our own B1 weights, with fresh Adam state")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing output checkpoint")
    parser.add_argument("--steps", type=int, default=12000, help="Additional optimizer steps")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW decay for matrices and embeddings")
    parser.add_argument("--dropout", type=float, default=0.1, help="Residual/embedding dropout for a new model")
    parser.add_argument("--positions", choices=("learned", "rope"), default="rope", help="Position encoding for a new model")
    parser.add_argument("--tie-embeddings", action=argparse.BooleanOptionalAction, default=True, help="Share token embeddings with the output layer in a new model")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--bpe-vocab", type=int, default=1024, help="Learn a local BPE vocabulary on the training split; 0 keeps bytes")
    parser.add_argument("--patience", type=int, default=10, help="Stop after this many validation checks without improvement; 0 disables")
    parser.add_argument("--last-reply-only", action="store_true", help="Experimental: supervise only the final reply in multi-turn training examples")
    parser.add_argument('--balanced-languages', action=argparse.BooleanOptionalAction, default=None,
                        help='Sample populated language cells equally; requires reviewed tagged data')
    args = parser.parse_args(argv)
    if args.resume and args.init_from:
        parser.error("Choose resume or init-from, not both")
    if args.init_from and Path(args.init_from).resolve() in (Path(args.out).resolve(), Path(args.out).with_name(Path(args.out).stem + '.best.npz').resolve()):
        parser.error("A new curriculum must not overwrite its source checkpoint")
    if not (args.steps > 0 and 1 <= args.batch <= 256 and math.isfinite(args.lr) and args.lr > 0 and args.log_every > 0 and args.patience >= 0):
        parser.error("Steps, batch size, learning rate, and log interval must be positive")
    if not (0 <= args.weight_decay < 1 and 0 <= args.dropout < 1):
        parser.error("Weight decay and dropout must be at least 0 and below 1")
    if Path(args.out).exists() and not args.overwrite and (not args.resume or Path(args.out).resolve() != Path(args.resume).resolve()):
        parser.error("Output checkpoint already exists. Resume it or pass --overwrite explicitly.")
    best_path = Path(args.out).with_name(Path(args.out).stem + ".best.npz")
    if best_path.exists() and not args.overwrite and not args.resume:
        parser.error("Best checkpoint already exists. Choose another output or pass --overwrite explicitly.")
    metadata = {}
    if args.resume:
        model, header, states = Model.load(args.resume)
        saved_balance = header.get('metadata', {}).get('balanced_languages', False)
        if args.balanced_languages is not None and args.balanced_languages != saved_balance:
            parser.error('Resume must preserve language sampling; start a new curriculum to change it')
        args.balanced_languages = saved_balance
        checkpoint_seed = header.get("metadata", {}).get("seed", 42)
        if args.seed is not None and args.seed != checkpoint_seed:
            parser.error("Resume seed must match the checkpoint to preserve its validation split.")
        args.seed = checkpoint_seed
        args.last_reply_only = args.last_reply_only or header.get("metadata", {}).get("training_targets") == "last assistant reply, earlier turns provide context"
        dataset = Dataset(args.data, seed=args.seed, tokenizer=model.tokenizer, last_reply_only=args.last_reply_only, balance_languages=bool(args.balanced_languages))
        if header.get("metadata", {}).get("data_sha256") != dataset.fingerprint:
            parser.error("Resume data differs from the checkpoint. Use the original dataset to preserve its validation split.")
        optimizer = AdamW(model.parameters)
        optimizer.restore(header.get("optimizer", {}), states)
        metadata = dict(header.get("metadata", {}))
    elif args.init_from:
        import hashlib
        model, header, _ = Model.load(args.init_from)
        args.seed = 42 if args.seed is None else args.seed
        dataset = Dataset(args.data, seed=args.seed, tokenizer=model.tokenizer, last_reply_only=args.last_reply_only, balance_languages=bool(args.balanced_languages))
        optimizer = AdamW(model.parameters, learning_rate=args.lr, weight_decay=args.weight_decay)
        metadata = {"parent_steps": header.get("metadata", {}).get("steps", 0),
                    "parent_sha256": hashlib.sha256(Path(args.init_from).read_bytes()).hexdigest()}
    else:
        args.seed = 42 if args.seed is None else args.seed
        tokenizer = BPETokenizer.train(training_texts(args.data, args.seed), args.bpe_vocab) if args.bpe_vocab else ByteTokenizer()
        dataset = Dataset(args.data, seed=args.seed, tokenizer=tokenizer, last_reply_only=args.last_reply_only, balance_languages=bool(args.balanced_languages))
        model = Model(Config(args.context, args.width, args.heads, args.layers, args.seed, tokenizer.vocab_size,
                             args.positions, args.tie_embeddings, args.dropout), tokenizer)
        optimizer = AdamW(model.parameters, learning_rate=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed + optimizer.step_count)
    if metadata.get("rng_state"):
        rng.bit_generator.state = metadata["rng_state"]
    started = time.monotonic()
    base_lr = float(metadata.get("base_learning_rate", optimizer.learning_rate))
    initial_validation = evaluate(model, dataset, args.batch, args.seed + 100)
    print(json.dumps({"name": model.name, "parameters": model.parameter_count, "training_examples": len(dataset.training), "validation_examples": len(dataset.validation), "dataset_tokens": dataset.token_count, "initial_validation_loss": initial_validation}), flush=True)
    first_step = optimizer.step_count
    best_validation = initial_validation
    if args.resume and best_path.exists():
        _, best_header, _ = Model.load(best_path)
        if best_header.get("metadata", {}).get("data_sha256") == dataset.fingerprint:
            best_validation = min(best_validation, best_header["metadata"]["validation_loss"])
    checks_without_improvement = 0
    metadata.update({"data_sha256": dataset.fingerprint, "seed": args.seed, "base_learning_rate": base_lr,
                     "training_examples": len(dataset.training), "validation_examples": len(dataset.validation),
                     "dataset_tokens": dataset.token_count, "educational_demo": True,
                     "validation_method": "all assistant targets, overlap scored once", "tokenizer": model.tokenizer.kind,
                     "training_targets": "last assistant reply, earlier turns provide context" if args.last_reply_only else "all assistant replies"})
    metadata['balanced_languages'] = bool(args.balanced_languages)
    if dataset.language_coverage is not None:
        metadata['language_training'] = dataset.language_coverage
    log_path = Path(args.out).with_suffix(".training.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not best_path.exists() or args.overwrite:
        model.save(best_path, {**metadata, "steps": optimizer.step_count,
                              "validation_loss": initial_validation,
                              "selection": "lowest full-validation loss"}, optimizer)
    interrupted = False
    try:
        with log_path.open("a", encoding="utf-8") as log:
            for local_step in range(args.steps):
                optimizer.zero_grad()
                x, y = dataset.batch("train", args.batch, model.config.context, rng, trim=True)
                loss = cross_entropy(model(x, training=True, rng=rng), y)
                value = float(loss.data)
                if not math.isfinite(value):
                    raise FloatingPointError("Nonfinite loss; the last valid checkpoint is preserved")
                loss.backward()
                # Warmup then cosine decay for this invocation's training budget.
                warmup = min(50, max(1, args.steps // 10))
                progress = max(0, local_step - warmup) / max(1, args.steps - warmup)
                optimizer.learning_rate = base_lr * min(1, (local_step + 1) / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
                gradient_norm = optimizer.step()
                step = optimizer.step_count
                if local_step == 0 or step % args.log_every == 0 or local_step == args.steps - 1:
                    validation = evaluate(model, dataset, args.batch, args.seed + 100)
                    metadata.update({"steps": step, "train_loss": value, "validation_loss": validation,
                                     "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                     "context_tokens": model.config.context, "rng_state": rng.bit_generator.state})
                    report = {"step": step, "train_loss": round(value, 4), "validation_loss": round(validation, 4), "gradient_norm": round(gradient_norm, 4), "elapsed_seconds": round(time.monotonic() - started, 1)}
                    print(json.dumps(report), flush=True)
                    log.write(json.dumps(report) + "\n"); log.flush()
                    model.save(args.out, metadata, optimizer)
                    if validation < best_validation:
                        best_validation = validation
                        checks_without_improvement = 0
                        model.save(best_path, {**metadata, "selection": "lowest full-validation loss"}, optimizer)
                    else:
                        checks_without_improvement += 1
                    if args.patience and checks_without_improvement >= args.patience:
                        print(f"Early stop: validation has not improved for {args.patience} checks. Best: {best_path}", flush=True)
                        break
    except KeyboardInterrupt:
        interrupted = True
        # Do not save an update interrupted inside an optimizer step. Periodic
        # atomic checkpoints remain resumable and mathematically consistent.
        print("Interrupted. The most recent completed checkpoint is preserved.", flush=True)
    if not interrupted:
        print(f"Saved {args.out} · {optimizer.step_count - first_step} new training steps", flush=True)


if __name__ == "__main__":
    main()
