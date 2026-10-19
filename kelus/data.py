"""Load local JSONL conversations and sample assistant-only training targets."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .tokenizer import PAD, ASSISTANT, ByteTokenizer, conversation


class Dataset:
    def __init__(self, filename, seed=42, validation_fraction=0.15, tokenizer=None, last_reply_only=False, balance_languages=False):
        raw = Path(filename).read_bytes()
        self.fingerprint = hashlib.sha256(raw).hexdigest()
        records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        from .curriculum import normalize, coverage
        tagged = any(any(key in row for key in ('code_language', 'reply_language')) for row in records)
        self.language_coverage = None
        self.language_buckets = []
        if tagged:
            records = normalize(records)
            if any(row['split'] == 'test' for row in records):
                raise ValueError('Held-out test records must never enter the training dataset')
            self.language_coverage = coverage(records)
        elif balance_languages:
            raise ValueError('Language balancing requires tagged, reviewed instruction data')
        examples, seen, splits, labels = [], set(), [], []
        self.tokenizer = tokenizer or ByteTokenizer()
        self.last_reply_only = last_reply_only
        for number, record in enumerate(records, 1):
            try:
                tokens, supervised = conversation(record["messages"], tokenizer=self.tokenizer)
                key = tuple(tokens)
                if key not in seen:
                    examples.append((np.array(tokens, dtype=np.int64), np.array(supervised, dtype=bool)))
                    splits.append(record.get("split"))
                    labels.append((record.get('code_language') or 'chat', record.get('reply_language')))
                    seen.add(key)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid training example on line {number}: {error}") from error
        if len(examples) < 4:
            raise ValueError("Provide at least four different conversations for a train/validation split")
        if any(split is not None for split in splits):
            if any(split not in ("train", "validation") for split in splits):
                raise ValueError("Every example must specify train or validation when explicit splits are used")
            self.training = [example for example, split in zip(examples, splits) if split == "train"]
            self.validation = [example for example, split in zip(examples, splits) if split == "validation"]
            if not self.training or not self.validation:
                raise ValueError("Both training and validation examples are required")
        else:
            order = np.random.default_rng(seed).permutation(len(examples))
            validation_count = max(1, int(len(examples) * validation_fraction))
            self.validation = [examples[index] for index in order[:validation_count]]
            self.training = [examples[index] for index in order[validation_count:]]
        if balance_languages:
            buckets = {}
            for example, split, label in zip(examples, splits, labels):
                if split == 'train':
                    buckets.setdefault(label, []).append(example)
            self.language_buckets = [buckets[key] for key in sorted(buckets)]
        self.token_count = sum(len(tokens) for tokens, _ in examples)

    def batch(self, split, batch_size, context, rng, trim=False):
        examples = self.training if split == "train" else self.validation
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        x = np.full((batch_size, context), PAD, dtype=np.int64)
        y = np.full((batch_size, context), -1, dtype=np.int64)
        longest = 1
        for row in range(batch_size):
            # Give each populated code-language/reply-language cell equal chance.
            # A small cell is sampled more often, not mistaken for more diverse data.
            pool = self.language_buckets[int(rng.integers(len(self.language_buckets)))] if split == 'train' and self.language_buckets else examples
            tokens, supervised = pool[int(rng.integers(len(pool)))]
            if split == "train" and self.last_reply_only:
                # Earlier replies supply history. Train the reply we would
                # actually be generating now, instead of spending most of a
                # multi-turn batch relearning already-answered older turns.
                supervised = supervised.copy()
                last_reply = int(np.flatnonzero(tokens == ASSISTANT)[-1])
                supervised[:last_reply + 1] = False
            # Every window retains at least one supervised assistant token.
            candidates = np.flatnonzero(supervised[1:])
            anchor = int(rng.choice(candidates))
            low, high = max(0, anchor - context + 1), min(anchor, max(0, len(tokens) - context - 1))
            start = int(rng.integers(low, high + 1)) if high >= low else low
            length = min(context, len(tokens) - start - 1)
            longest = max(longest, length)
            x[row, :length] = tokens[start:start + length]
            targets = tokens[start + 1:start + length + 1].copy()
            targets[~supervised[start + 1:start + length + 1]] = -1
            y[row, :length] = targets
        return (x[:, :longest], y[:, :longest]) if trim else (x, y)

    def validation_batches(self, batch_size, context):
        """Score every validation target exactly once, with preceding context.

        A half-context overlap carries earlier text into long conversations;
        overlapping labels are ignored instead of counted twice.
        """
        windows = []
        for tokens, supervised in self.validation:
            if self.last_reply_only:
                supervised = supervised.copy()
                supervised[:int(np.flatnonzero(tokens == ASSISTANT)[-1]) + 1] = False
            scored_until = 1
            while scored_until < len(tokens):
                start = max(0, scored_until - context // 2)
                end = min(len(tokens) - 1, start + context)
                targets = tokens[start + 1:end + 1].copy()
                valid = supervised[start + 1:end + 1].copy()
                valid[:max(0, scored_until - start - 1)] = False
                targets[~valid] = -1
                if valid.any():
                    windows.append((tokens[start:end], targets))
                scored_until = end + 1
        for offset in range(0, len(windows), batch_size):
            selected = windows[offset:offset + batch_size]
            length = max(len(x) for x, _ in selected)
            x = np.full((len(selected), length), PAD, dtype=np.int64)
            y = np.full_like(x, -1)
            for row, (inputs, targets) in enumerate(selected):
                x[row, :len(inputs)] = inputs
                y[row, :len(targets)] = targets
            yield x, y
