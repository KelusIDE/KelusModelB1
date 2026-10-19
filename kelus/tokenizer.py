"""Local byte and BPE tokenizers, learned from scratch without tokenizer libraries."""
from collections import Counter
from functools import lru_cache
import re

PAD, BOS, EOS, USER, ASSISTANT = range(256, 261)
VOCAB_SIZE = 261
SPECIAL_IDS = (PAD, BOS, EOS, USER, ASSISTANT)


def encode(text):
    if not isinstance(text, str):
        raise ValueError("Message content must be text")
    return list(text.encode("utf-8"))


def decode(tokens):
    return bytes(int(token) for token in tokens if 0 <= int(token) < 256).decode("utf-8", errors="replace")


class ByteTokenizer:
    kind = "bytes"
    vocab_size = VOCAB_SIZE

    encode = staticmethod(encode)
    decode = staticmethod(decode)

    def token_bytes(self, token):
        return bytes([token]) if 0 <= token < 256 else b""

    def to_dict(self):
        return {"kind": self.kind}


class BPETokenizer(ByteTokenizer):
    """Byte-pair merges learned exclusively from the local training split.

    Regex pieces preserve whitespace and punctuation, and prevent merges across
    words. Unknown text always falls back to UTF-8 bytes, with no unknown ID.
    """
    kind = "bpe"
    patterns = {
        1: re.compile(r" ?[A-Za-z]+| ?[0-9]+|\s+|[^A-Za-z0-9\s]"),
        # Version 2 keeps a code fence and its language tag in one piece, so an
        # opening ```python differs from a closing ``` in the learned vocabulary.
        2: re.compile(r"```[A-Za-z]*| ?[A-Za-z]+| ?[0-9]+|\s+|[^A-Za-z0-9\s]"),
    }

    def __init__(self, merges, version=1):
        if version not in self.patterns:
            raise ValueError("Unknown BPE pattern version")
        self.version = version
        self.pattern = self.patterns[version]
        self.merges = [tuple(pair) for pair in merges]
        self.pieces = [bytes([i]) for i in range(256)] + [b""] * 5
        self.ranks = {}
        for rank, pair in enumerate(self.merges):
            if (len(pair) != 2 or pair in self.ranks or
                    any(type(i) is not int or not 0 <= i < len(self.pieces) or i in SPECIAL_IDS for i in pair)):
                raise ValueError("Invalid BPE merge vocabulary")
            self.ranks[pair] = (rank, len(self.pieces))
            self.pieces.append(self.pieces[pair[0]] + self.pieces[pair[1]])
        self.vocab_size = len(self.pieces)

    @staticmethod
    def merge(sequence, pair, replacement):
        result, i = [], 0
        while i < len(sequence):
            if i + 1 < len(sequence) and (sequence[i], sequence[i + 1]) == pair:
                result.append(replacement); i += 2
            else:
                result.append(sequence[i]); i += 1
        return tuple(result)

    @classmethod
    def train(cls, texts, vocab_size=1024, version=2):
        if not VOCAB_SIZE <= vocab_size <= 8192:
            raise ValueError("BPE vocabulary must contain 261–8192 tokens")
        words = Counter(tuple(piece.encode("utf-8")) for text in texts for piece in cls.patterns[version].findall(text))
        merges = []
        for token in range(VOCAB_SIZE, vocab_size):
            counts = Counter()
            for word, count in words.items():
                for pair in zip(word, word[1:]):
                    counts[pair] += count
            if not counts:
                break
            pair, count = max(counts.items(), key=lambda item: (item[1], tuple(-i for i in item[0])))
            if count < 2:
                break
            merges.append(pair)
            updated = Counter()
            for word, count in words.items():
                updated[cls.merge(word, pair, token)] += count
            words = updated
        return cls(merges, version)

    @lru_cache(maxsize=16384)
    def encode_piece(self, piece):
        sequence = tuple(piece.encode("utf-8"))
        while len(sequence) > 1:
            candidates = [(self.ranks[pair], pair) for pair in zip(sequence, sequence[1:]) if pair in self.ranks]
            if not candidates:
                break
            (_, token), pair = min(candidates)
            sequence = self.merge(sequence, pair, token)
        return sequence

    def encode(self, text):
        if not isinstance(text, str):
            raise ValueError("Message content must be text")
        return [token for piece in self.pattern.findall(text) for token in self.encode_piece(piece)]

    def token_bytes(self, token):
        if not 0 <= token < self.vocab_size:
            raise ValueError("Invalid BPE token")
        return self.pieces[token]

    def decode(self, tokens):
        return b"".join(self.token_bytes(int(token)) for token in tokens).decode("utf-8", errors="replace")

    def to_dict(self):
        value = {"kind": self.kind, "merges": self.merges}
        if self.version != 1:
            value["version"] = self.version
        return value


def from_dict(value):
    if value.get("kind") == "bpe":
        return BPETokenizer(value["merges"], value.get("version", 1))
    if value.get("kind", "bytes") == "bytes":
        return ByteTokenizer()
    raise ValueError("Unknown tokenizer")


def conversation(messages, generation=False, tokenizer=None):
    tokenizer = tokenizer or ByteTokenizer()
    tokens, supervised = [BOS], [False]
    if not isinstance(messages, list) or not messages:
        raise ValueError("Expected a nonempty messages list")
    expected = "user"
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != expected:
            raise ValueError("Messages must alternate user and assistant, beginning with user")
        role = message["role"]
        content = tokenizer.encode(message.get("content"))
        # "supervise": false keeps an assistant turn as context only, e.g. a
        # deliberately wrong earlier reply that the user then corrects.
        target = role == "assistant" and message.get("supervise", True) is not False
        tokens.extend([USER if role == "user" else ASSISTANT] + content)
        supervised.extend([False] + [target] * len(content))
        if role == "assistant":
            tokens.append(EOS)
            supervised.append(target)
        expected = "assistant" if role == "user" else "user"
    if generation:
        if expected != "assistant":
            raise ValueError("A generation prompt must end with a user message")
        tokens.append(ASSISTANT)
        supervised.append(False)
    elif expected != "user":
        raise ValueError("Training conversations must end with an assistant reply")
    return tokens, supervised
