"""Kelus Model B1: a trainable causal transformer, initialized from scratch."""
from dataclasses import asdict, dataclass
import codecs
import json
import math
import os
from pathlib import Path
import tempfile
import numpy as np
from .tensor import Tensor, embedding
from .tokenizer import VOCAB_SIZE, PAD, BOS, USER, ASSISTANT, EOS, ByteTokenizer, from_dict, conversation


@dataclass(frozen=True)
class Config:
    context: int = 192
    width: int = 48
    heads: int = 4
    layers: int = 2
    seed: int = 42
    vocab_size: int = VOCAB_SIZE
    # The defaults reproduce the original B1 architecture, so older
    # checkpoints (whose headers lack these fields) load unchanged.
    positions: str = "learned"
    tie_embeddings: bool = False
    dropout: float = 0.0

    def validate(self):
        if not (16 <= self.context <= 2048 and 8 <= self.width <= 1024 and 1 <= self.layers <= 24
                and 1 <= self.heads <= self.width and self.width % self.heads == 0
                and VOCAB_SIZE <= self.vocab_size <= 8192):
            raise ValueError("Invalid model size: width must divide evenly into attention heads")
        if self.positions not in ("learned", "rope") or (self.positions == "rope" and (self.width // self.heads) % 2):
            raise ValueError("Positions must be learned, or rope with an even attention head width")
        if not 0 <= self.dropout < 1:
            raise ValueError("Dropout must be at least 0 and below 1")


class Model:
    name = "Kelus Model B1"

    def __init__(self, config=Config(), tokenizer=None):
        config.validate()
        self.config = config
        self.tokenizer = tokenizer or ByteTokenizer()
        if self.tokenizer.vocab_size != config.vocab_size:
            raise ValueError("Model and tokenizer vocabulary sizes differ")
        self.parameters = {}
        rng = np.random.default_rng(config.seed)

        def weight(name, shape, mode="random"):
            value = (rng.normal(0, 0.02, shape) if mode == "random" else
                     np.ones(shape) if mode == "ones" else np.zeros(shape)).astype(np.float32)
            self.parameters[name] = Tensor(value, requires_grad=True)

        d = config.width
        weight("token", (config.vocab_size, d))
        if config.positions == "learned":
            weight("position", (config.context, d))
        else:
            # Rotary positions: each query/key pair is rotated by its absolute
            # position, so attention scores depend on relative distance.
            half = d // config.heads // 2
            angles = np.arange(config.context)[:, None] * 10000.0 ** (-np.arange(half) / half)
            self.rope_cos = np.concatenate([np.cos(angles)] * 2, axis=1).astype(np.float32)
            self.rope_sin = np.concatenate([np.sin(angles)] * 2, axis=1).astype(np.float32)
            # x @ rotate_half == concat(-second_half, first_half)
            self.rotate_half = np.zeros((2 * half, 2 * half), dtype=np.float32)
            self.rotate_half[np.arange(half) + half, np.arange(half)] = -1
            self.rotate_half[np.arange(half), np.arange(half) + half] = 1
        for layer in range(config.layers):
            prefix = f"block{layer}."
            for norm in ("norm1", "norm2"):
                weight(prefix + norm + ".gain", (d,), "ones")
                weight(prefix + norm + ".bias", (d,), "zeros")
            for matrix, shape in (("q", (d, d)), ("k", (d, d)), ("v", (d, d)), ("projection", (d, d)), ("up", (d, 4*d)), ("down", (4*d, d))):
                weight(prefix + matrix, shape)
        weight("final.gain", (d,), "ones")
        weight("final.bias", (d,), "zeros")
        if not config.tie_embeddings:
            weight("output", (d, config.vocab_size))

    @property
    def parameter_count(self):
        return sum(parameter.data.size for parameter in self.parameters.values())

    def norm(self, x, name):
        centered = x - x.mean(axis=-1, keepdims=True)
        normalized = centered * ((centered * centered).mean(axis=-1, keepdims=True) + 1e-5) ** -0.5
        return normalized * self.parameters[name + ".gain"] + self.parameters[name + ".bias"]

    def rotate(self, x, start, time):
        """Apply rotary positions to [..., time, head_width] queries or keys."""
        cos, sin = self.rope_cos[start:start + time], self.rope_sin[start:start + time]
        return x * cos + (x @ self.rotate_half) * sin

    def output_matrix(self):
        return self.parameters["token"].transpose(1, 0) if self.config.tie_embeddings else self.parameters["output"]

    def __call__(self, token_ids, training=False, rng=None):
        """Full forward pass. `training` enables dropout, drawn from `rng`."""
        ids = np.asarray(token_ids, dtype=np.int64)
        if ids.ndim != 2 or not (1 <= ids.shape[1] <= self.config.context):
            raise ValueError("Expected [batch, time] token IDs within the context limit")
        if np.any(ids < 0) or np.any(ids >= self.config.vocab_size):
            raise ValueError("Invalid token ID")
        batch, time = ids.shape
        d, heads = self.config.width, self.config.heads
        head_width = d // heads
        rate = self.config.dropout if training else 0.0
        rng = rng if rng is not None else np.random.default_rng()

        def drop(value):
            if not rate:
                return value
            return value * ((rng.random(value.data.shape) >= rate) / (1 - rate)).astype(np.float32)

        x = embedding(self.parameters["token"], ids)
        if self.config.positions == "learned":
            x = x + embedding(self.parameters["position"], np.arange(time))
        x = drop(x)
        mask = np.where(np.triu(np.ones((time, time), dtype=bool), 1), -1e9, 0).astype(np.float32)
        for layer in range(self.config.layers):
            prefix = f"block{layer}."
            normalized = self.norm(x, prefix + "norm1")

            def project(name):
                return (normalized @ self.parameters[prefix + name]).reshape(batch, time, heads, head_width).transpose(0, 2, 1, 3)

            q, k, v = project("q"), project("k"), project("v")
            if self.config.positions == "rope":
                q, k = self.rotate(q, 0, time), self.rotate(k, 0, time)
            attention = ((q @ k.transpose(0, 1, 3, 2)) * (head_width ** -0.5) + mask).softmax()
            mixed = (attention @ v).transpose(0, 2, 1, 3).reshape(batch, time, d)
            x = x + drop(mixed @ self.parameters[prefix + "projection"])
            hidden = self.norm(x, prefix + "norm2") @ self.parameters[prefix + "up"]
            activated = 0.5 * hidden * (1 + (math.sqrt(2 / math.pi) * (hidden + 0.044715 * hidden ** 3)).tanh())
            x = x + drop(activated @ self.parameters[prefix + "down"])
        return self.norm(x, "final") @ self.output_matrix()

    def next_logits(self, tokens, cache):
        """Next-token logits after `tokens`, reusing cached keys and values.

        Inference only, in plain NumPy. `cache` is an empty dict owned by one
        generation; each call processes only tokens it has not seen, so a reply
        costs one position per step instead of a full forward pass.
        """
        config, values = self.config, {name: parameter.data for name, parameter in self.parameters.items()}
        start = cache.get("length", 0)
        new = np.asarray(tokens[start:], dtype=np.int64)
        count, end = len(new), len(tokens)
        if not (1 <= count and end <= config.context) or np.any(new < 0) or np.any(new >= config.vocab_size):
            raise ValueError("Expected new token IDs within the context limit")
        heads, head_width = config.heads, config.width // config.heads
        if "keys" not in cache:
            cache["keys"] = np.zeros((config.layers, heads, config.context, head_width), dtype=np.float32)
            cache["values"] = np.zeros_like(cache["keys"])

        def norm(x, name):
            centered = x - x.mean(axis=-1, keepdims=True)
            normalized = centered * ((centered * centered).mean(axis=-1, keepdims=True) + 1e-5) ** -0.5
            return normalized * values[name + ".gain"] + values[name + ".bias"]

        x = values["token"][new]
        if config.positions == "learned":
            x = x + values["position"][start:end]
        mask = np.where(np.arange(end)[None, :] > np.arange(start, end)[:, None], -1e9, 0).astype(np.float32)
        for layer in range(config.layers):
            prefix = f"block{layer}."
            normalized = norm(x, prefix + "norm1")
            q, k, v = ((normalized @ values[prefix + name]).reshape(count, heads, head_width).transpose(1, 0, 2) for name in ("q", "k", "v"))
            if config.positions == "rope":
                q, k = self.rotate(q, start, count), self.rotate(k, start, count)
            cache["keys"][layer, :, start:end], cache["values"][layer, :, start:end] = k, v
            scores = q @ cache["keys"][layer, :, :end].transpose(0, 2, 1) * (head_width ** -0.5) + mask
            scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
            scores /= scores.sum(axis=-1, keepdims=True)
            mixed = (scores @ cache["values"][layer, :, :end]).transpose(1, 0, 2).reshape(count, config.width)
            x = x + mixed @ values[prefix + "projection"]
            hidden = norm(x, prefix + "norm2") @ values[prefix + "up"]
            x = x + (0.5 * hidden * (1 + np.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * hidden ** 3)))) @ values[prefix + "down"]
        cache["length"] = end
        output = values["token"].T if config.tie_embeddings else values["output"]
        return (norm(x[-1:], "final") @ output)[0]

    def prepare_prompt(self, messages, reserve=64):
        """Keep complete recent turns rather than cut through a word or role.

        Generated tokens never slide the learned absolute positions. Leave
        room for a reply and tell callers when older turns were omitted.
        """
        retained = list(messages)
        tokens, _ = conversation(retained, generation=True, tokenizer=self.tokenizer)
        original_count = len(retained)
        reserve = min(reserve, max(1, self.config.context // 3))
        while len(tokens) + reserve > self.config.context and len(retained) > 1:
            retained = retained[2:]
            tokens, _ = conversation(retained, generation=True, tokenizer=self.tokenizer)
        if len(tokens) > self.config.context - max(2, min(16, self.config.context // 4)):
            raise ValueError("This message exceeds B1's context. Shorten it or start with a smaller code example.")
        return tokens, original_count - len(retained)

    def generate(self, messages, max_tokens=180, temperature=0.6, top_k=30, seed=42):
        if not (1 <= max_tokens <= 2048 and 0 <= temperature <= 2 and 1 <= top_k <= self.config.vocab_size):
            raise ValueError("Invalid generation settings")
        tokens, _ = self.prepare_prompt(messages, reserve=max_tokens)
        rng = np.random.default_rng(seed)
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        cache = {}
        for _ in range(min(max_tokens, self.config.context - len(tokens))):
            logits = np.array(self.next_logits(tokens, cache), dtype=np.float32)
            # Only text bytes and the assistant's end marker may be generated.
            logits[[PAD, BOS, USER, ASSISTANT]] = -np.inf
            if temperature == 0:
                token = int(np.argmax(logits))
            else:
                indices = np.argpartition(logits, -top_k)[-top_k:]
                values = logits[indices] / temperature
                probability = np.exp(values - values.max())
                probability /= probability.sum()
                token = int(rng.choice(indices, p=probability))
            if token == EOS:
                break
            tokens.append(token)
            text = decoder.decode(self.tokenizer.token_bytes(token), final=False)
            if text:
                yield text
        tail = decoder.decode(b'', final=True)
        if tail:
            yield tail

    def save(self, filename, metadata=None, optimizer=None):
        destination = Path(filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        header = {"format": "kelus-b1", "version": 1, "name": self.name, "config": asdict(self.config), "tokenizer": self.tokenizer.to_dict(), "metadata": metadata or {}}
        values = {"parameter." + name: parameter.data for name, parameter in self.parameters.items()}
        if optimizer:
            header["optimizer"] = optimizer.metadata()
            values.update(optimizer.arrays())
        values["header"] = np.array(json.dumps(header))
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".npz", delete=False) as file:
                temporary = file.name
                np.savez_compressed(file, **values)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, filename):
        with np.load(filename, allow_pickle=False) as archive:
            header = json.loads(str(archive["header"]))
            if header.get("format") != "kelus-b1" or header.get("version") != 1:
                raise ValueError("Unsupported checkpoint format")
            model = cls(Config(**header["config"]), from_dict(header.get("tokenizer", {"kind": "bytes"})))
            for name, parameter in model.parameters.items():
                value = archive["parameter." + name]
                if value.shape != parameter.data.shape or value.dtype.kind != "f" or not np.isfinite(value).all():
                    raise ValueError(f"Invalid checkpoint parameter: {name}")
                parameter.data[:] = value
            states = {name: archive[name].copy() for name in archive.files if name.startswith("adam.")}
        return model, header, states
