"""Kani Advanced: Kani's own transformer, pretrained from scratch on Kaggle's free GPUs.

The architecture and tokenizer are exactly the local Kani's (kelus/model.py, data/big-v1/tokenizer.json):
pre-LayerNorm blocks, rotary positions, a 4x GELU feed-forward and tied embeddings, so the exported
kani-advanced.npz loads straight into `kelus`. No pretrained weights: it learns from public text,
FineWeb-Edu and Cosmopedia v2, a new slice of it every Kaggle session.

    python kani_advanced.py prepare --settings settings.json    # download and tokenize this session's text
    torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29500 kani_advanced.py train --settings settings.json
    python kani_advanced.py sample --settings settings.json --prompt "The moon is"
    python kani_advanced.py export --settings settings.json     # kani-advanced.npz from checkpoint.pt
    python kani_advanced.py retokenize --settings settings.json --checkpoint old/checkpoint.pt --old-tokenizer old.json
    python kani_advanced.py grams --tests DIR --out contamination.npy   # the test sets' 13-word runs, never to train on
    python kani_advanced.py grow --settings settings.json --base kani-advanced.npz --progress progress.json   # a deeper run

Multilingual runs (BPE pattern version 3) give each source a `share` of the session's tokens, a `filter` that keeps
documents in their language's script, and a held-out `validation_file`; `retokenize` moves an English checkpoint
to the multilingual vocabulary, keeping everything it learned. Shares also mix English with code. A source's
`revision` reads another branch of its dataset, such as Hugging Face's parquet copy of a JSON dataset
('refs/convert/parquet'), and its `column` names the field holding the text ('text' unless set; 'code' for code).
"""
import functools
import argparse
import hashlib
from contextlib import nullcontext
import glob
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, BOS, EOS, USER, ASSISTANT = range(256, 261)

LATIN, HANGUL = "A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u024f\u1e00-\u1eff", "\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f"
KANA, HAN = "\u3040-\u309f\u30a0-\u30ff\u31f0-\u31ff", "\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
PATTERNS = {2: re.compile(r"```[A-Za-z]*| ?[A-Za-z]+| ?[0-9]+|\s+|[^A-Za-z0-9\s]"),
            3: re.compile(f"```[A-Za-z]*| ?[{LATIN}]+| ?[{HANGUL}]+|[{KANA}]+|[{HAN}]+| ?[0-9]+|\\s+|[^\\s]")}

DEFAULTS = {
    'run_dir': '/kaggle/working/kani-advanced',
    'input_root': '/kaggle/input',
    'data_dir': '/tmp/kani-advanced-data',
    'tokenizer': 'kani_tokenizer.json',
    'width': 1024, 'layers': 16, 'heads': 16, 'context': 1024,
    'target_tokens': 4_000_000_000,
    'session_tokens': 1_200_000_000,
    'batch_tokens': 262_144,
    'micro_batch': 4,
    'lr': 4e-4, 'min_lr_ratio': 0.1, 'warmup_tokens': 100_000_000, 'decay_fraction': 0.2,
    'weight_decay': 0.1,
    'hours': 10.0,
    'checkpoint_minutes': 30,
    'eval_every': 250, 'eval_batches': 16,
    'validation_documents': 4000,
    'sources': [
        {'repo': 'HuggingFaceFW/fineweb-edu', 'prefix': 'sample/10BT/', 'per_session': 1},
        {'repo': 'HuggingFaceTB/smollm-corpus', 'prefix': 'cosmopedia-v2/', 'per_session': 1},
    ],
    'gradient_checkpointing': False,
    'compile': False,
    'seed': 2041,
    'continue_from_previous': False,
    'workers': 0,
    'rewarm_tokens': 50_000_000,
    'decontaminate': False,
}


def load_settings(path):
    settings = {**DEFAULTS, **json.loads(Path(path).read_text())}
    w, h = settings['width'], settings['heads']

    if not (16 <= settings['context'] <= 2048 and 8 <= w <= 1024 and 1 <= settings['layers'] <= 24
            and w % h == 0 and (w // h) % 2 == 0):
        raise SystemExit('Kani needs context <= 2048, width <= 1024, layers <= 24, and an even head width')
    return settings




class Tokenizer:
    """kelus.tokenizer.BPETokenizer (versions 2 and 3), with an unbounded per-piece cache for speed."""

    def __init__(self, merges, version=2):
        self.version, self.pattern = version, PATTERNS[version]
        self.merges = [tuple(pair) for pair in merges]
        self.pieces = [bytes([i]) for i in range(256)] + [b''] * 5
        self.ranks = {}
        for rank, (a, b) in enumerate(self.merges):
            self.ranks[(a, b)] = (rank, len(self.pieces))
            self.pieces.append(self.pieces[a] + self.pieces[b])
        self.vocab_size = len(self.pieces)
        self.cache = {}

    @classmethod
    def load(cls, path):
        value = json.loads(Path(path).read_text())
        if value.get('kind') != 'bpe' or value.get('version') not in PATTERNS:
            raise SystemExit(f'{path} is not a Kani BPE tokenizer (version 2 or 3)')
        return cls(value['merges'], value['version'])

    def encode_piece(self, piece):
        cached = self.cache.get(piece)
        if cached is not None:
            return cached
        sequence = list(piece.encode('utf-8'))
        while len(sequence) > 1:
            best = None
            for pair in zip(sequence, sequence[1:]):
                rank = self.ranks.get(pair)
                if rank is not None and (best is None or rank < best[0]):
                    best = (rank, pair)
            if best is None:
                break
            (_, token), (a, b) = best
            merged, i = [], 0
            while i < len(sequence):
                if i + 1 < len(sequence) and sequence[i] == a and sequence[i + 1] == b:
                    merged.append(token)
                    i += 2
                else:
                    merged.append(sequence[i])
                    i += 1
            sequence = merged
        if len(self.cache) > 4_000_000:
            self.cache.clear()
        self.cache[piece] = sequence
        return sequence

    def encode(self, text):
        return [token for piece in self.pattern.findall(text) for token in self.encode_piece(piece)]

    def decode(self, tokens):
        return b''.join(self.pieces[t] for t in tokens if 0 <= t < self.vocab_size).decode('utf-8', errors='replace')

    def to_dict(self):
        return {'kind': 'bpe', 'merges': [list(pair) for pair in self.merges], 'version': self.version}


_worker = None


def fast_tokenizer(path):
    """The Rust tokenizer (kaggle/kani_tokenize: the same tokens, ~4.5x faster), or None when it is not installed."""
    try:
        from kani_tokenize import Tokenizer as Rust  # kaggle/kani_tokenize's source folder alone imports, but has no Tokenizer
    except ImportError:
        return None
    value = json.loads(Path(path).read_text())
    return Rust([tuple(pair) for pair in value['merges']], value['version'])


def _start_worker(path, banned=None):
    global _worker, _banned
    _worker = fast_tokenizer(path) or Tokenizer.load(path)
    _banned = np.load(banned) if banned else np.zeros(0, dtype=np.uint64)


# The test sets Kani is measured on (kelus/standard_eval.py: HellaSwag, ARC, PIQA, Winogrande, LAMBADA) are never
# trained on: a document sharing a run of 13 words with one of them is dropped before tokenizing, as GPT-3 did.
# contamination.npy (sorted hashes of those runs) comes from the `grams` command; prepare looks for it in run_dir, then
# beside an attached checkpoint, and stops when `decontaminate` is set and it is missing.
GRAM, _WORDS = 13, re.compile(r'[a-z0-9]+')
_banned = np.zeros(0, dtype=np.uint64)


@functools.lru_cache(maxsize=1 << 20)
def _word_hash(word):
    return int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), 'little')


def gram_hashes(text):
    """A 64-bit hash of every run of 13 lower-cased words (letters and digits only) in `text`."""
    words = _WORDS.findall(text.lower())
    count = len(words) - GRAM + 1
    if count < 1:
        return np.zeros(0, dtype=np.uint64)
    ids = np.fromiter(map(_word_hash, words), dtype=np.uint64, count=len(words))
    out = np.zeros(count, dtype=np.uint64)
    for k in range(GRAM):  # wraps at 2**64, the same on every machine
        out = out * np.uint64(1099511628211) + ids[k:k + count]
    return out


def contaminated(text, banned):
    grams = gram_hashes(text)
    if not len(grams) or not len(banned):
        return False
    at = np.minimum(np.searchsorted(banned, grams), len(banned) - 1)
    return bool((banned[at] == grams).any())


def benchmark_texts(folder):
    """Every question with each of its answers, from kelus.standard_eval's test files in `folder`."""
    folder = Path(folder)
    lines = lambda name: [json.loads(line) for line in (folder / name).read_text(encoding='utf-8').splitlines() if line.strip()]
    for row in lines('hellaswag_val.jsonl'):
        yield from (f"{row['activity_label']}: {row['ctx']} {ending}" for ending in row['endings'])
    for name in ('arc_easy_test.jsonl', 'arc_challenge_test.jsonl'):
        for row in lines(name):
            yield from (f"{row['question']} {choice}" for choice in row['choices']['text'])
    for row in lines('piqa_valid.jsonl'):
        yield from (f"{row['goal']} {row['sol1']}", f"{row['goal']} {row['sol2']}")
    for row in lines('winogrande_val.jsonl'):
        yield from (row['sentence'].replace('_', row['option1']), row['sentence'].replace('_', row['option2']))
    for row in lines('lambada_test.jsonl'):
        yield row['text']


def write_grams(folder, out):
    grams = np.unique(np.concatenate([gram_hashes(text) for text in benchmark_texts(folder)]))
    np.save(out, grams)
    print(f'{len(grams):,} distinct 13-word runs from the test sets in {folder} -> {out}')



PAIRS = ('们們 这這 说說 时時 国國 会會 来來 对對 发發 后後 个個 为為 学學 经經 动動 现現 机機 开開 关關 长長 问問 见見 进進 实實 两兩 车車 东東 门門 '
         '应應 电電 气氣 头頭 从從 书書 难難 爱愛 让讓 认認 识識 读讀 写寫 听聽 买買 卖賣 钱錢 银銀 医醫 药藥 话話 语語 请請 谢謝 体體 题題 华華 业業 '
         '产產 网網 线線 结結 给給 红紅 绿綠 鸟鳥 鱼魚 马馬 龙龍 风風 飞飛 云雲 万萬 与與 区區 级級 历歷 战戰 农農 观觀 欢歡 乐樂 节節 质質 资資 '
         '务務 处處 员員 种種 总總 统統 系係 组組 织織 设設 计計 论論 议議 记記 讲講 该該 报報 场場 还還 运運 远遠 选選 边邊 导導 '
         '将將 岁歲 热熱 图圖 团團 园園 圆圓 广廣 庆慶 张張 强強 归歸 当當 录錄 彻徹 忆憶 态態 怀懷 恶惡 悬懸 惊驚 愿願 戏戲 户戶').split()
SIMPLIFIED, TRADITIONAL = {p[0] for p in PAIRS}, {p[1] for p in PAIRS}


def in_script(text, kind):
    """True when most of the document's letters are in the script `kind` (FineWeb-2's language labels are not perfect)."""
    letters = [c for c in text[:5000] if c.isalpha()]
    if not letters:
        return False
    if kind == 'hangul':
        return sum('\uac00' <= c <= '\ud7a3' for c in letters) >= 0.6 * len(letters)
    kana = sum('\u3040' <= c <= '\u30ff' for c in letters)
    han = sum('\u4e00' <= c <= '\u9fff' for c in letters)
    if kind == 'kana':
        return kana >= 20 and kana + han >= 0.6 * len(letters)
    simplified, traditional = sum(c in SIMPLIFIED for c in text[:5000]), sum(c in TRADITIONAL for c in text[:5000])
    wanted = simplified if kind == 'simplified' else traditional
    return (kana == 0 and han >= 0.6 * len(letters) and simplified + traditional >= 3
            and wanted >= 4 * (simplified + traditional - wanted))


def encode_documents(texts, kind=None):
    """[BOS] text [EOS] for every document (in the script `kind`, when given), as kelus's raw corpora store them."""
    if len(_banned):
        texts = [text for text in texts if text and not contaminated(text, _banned)]
    if not isinstance(_worker, Tokenizer):
        if kind is not None:
            texts = [text for text in texts if text and in_script(text, kind)]
        return np.frombuffer(_worker.encode_documents(texts), dtype='<u2')
    out = []
    for text in texts:
        if text and (kind is None or in_script(text, kind)):
            out.append(BOS)
            out.extend(_worker.encode(text))
            out.append(EOS)
    return np.asarray(out, dtype='<u2')




def find_file(settings, name):
    """run_dir/<name>, else the same file in an attached earlier output of this notebook (or a dataset). With several
    attached, the one beside the progress.json that has seen the most tokens: the latest session's."""
    run = Path(settings['run_dir'])
    if (run / name).exists():
        return run / name
    root = Path(settings['input_root'])
    # In a folder named like run_dir (a notebook's output), or loose at a dataset's top level (files uploaded as they are).
    patterns = [f'{"*/" * depth}{run.name}/{name}' for depth in (1, 2, 3)] + [f'{"*/" * depth}{name}' for depth in (1, 2, 3)]
    found = list(dict.fromkeys(path for pattern in patterns for path in (sorted(root.glob(pattern)) if root.is_dir() else [])))

    def tokens(path):
        try:
            return json.loads((path.parent / 'progress.json').read_text())['tokens_seen']
        except (OSError, ValueError, KeyError):
            return -1
    return max(found, key=tokens) if found else None


def read_progress(settings):
    path = find_file(settings, 'progress.json')
    return json.loads(path.read_text()) if path else {'session': 0, 'tokens_seen': 0, 'step': 0}


def source_files(source):
    """Sorted data files of one source: Hugging Face parquet files, or local ones ('local' glob) for testing."""
    if 'local' in source:
        return sorted(glob.glob(source['local']))
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(source['repo'], repo_type='dataset', revision=source.get('revision'))
    return sorted(f for f in files if f.startswith(source['prefix']) and f.endswith('.parquet'))


def fetch(source, name, folder):
    if 'local' in source:
        return Path(name)
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(source['repo'], name, repo_type='dataset', revision=source.get('revision'), local_dir=folder))


def parquet_texts(path, documents, column='text'):
    """Lists of up to `documents` texts from a parquet file's text column, read lazily."""
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(path).iter_batches(columns=[column], batch_size=documents):
        yield batch.column(column).to_pylist()


def multilingual(settings):
    return any('share' in source for source in settings['sources'])


def source_key(source):
    return source.get('local') or f"{source['repo']}/{source.get('prefix', '')}"


def start_file(source, session, cursors):
    """The number of the first file of `source` that a session reads: the one after the last file read before (progress
    'next_file', which both this script and kani_tpu.py keep), else session x per_session + offset. Numbers wrap around
    the source's files."""
    key = source_key(source)
    return cursors[key] if key in cursors else session * source['per_session'] + source.get('offset', 0)


def data_files(settings):
    """Each source's training files, and the held-out validation file(s).

    English runs hold out the first source's last file. Multilingual runs hold out a file of every source (its
    `validation_file`, else its last file) and can start a source `offset` files in, so two sources reading the
    same files (Simplified and Traditional Chinese) read different ones.
    """
    lists, validation = [], [] if multilingual(settings) else None
    for index, source in enumerate(settings['sources']):
        files = source_files(source)
        if not multilingual(settings):
            if index == 0:
                validation, files = files[-1], files[:-1]
        else:
            held = source.get('validation_file') or files[-1]
            validation.append((index, held))
            files = [name for name in files if name != held]
        lists.append(files)
    return lists, validation


def session_plan(settings, session, cursors=None):
    """(source index, file) pairs to tokenize this session, the held-out validation file(s), and where the next
    session starts in each source."""
    lists, validation = data_files(settings)
    plan, after = [], {}
    for index, (source, files) in enumerate(zip(settings['sources'], lists)):
        start = start_file(source, session, cursors or {})
        plan += [(index, files[(start + j) % len(files)]) for j in range(source['per_session'])]
        after[source_key(source)] = start + source['per_session']
    return plan, validation, after


def validation_tokens(settings, validation, pool, folder):
    """The held-out documents' tokens: validation_documents in all, shared between the held-out files."""
    held_out = validation if multilingual(settings) else [(0, validation)]
    parts = []
    for index, name in held_out:
        source = settings['sources'][index]
        path = fetch(source, name, folder)
        encode = functools.partial(encode_documents, kind=source.get('filter'))
        wanted, texts = settings['validation_documents'] // len(held_out), []
        for batch in parquet_texts(path, 2000, source.get('column', 'text')):
            texts += [text for text in batch if source.get('filter') is None or in_script(text, source['filter'])]
            if len(texts) >= wanted:
                break
        texts = texts[:wanted]
        parts += pool.map(encode, [texts[i:i + 250] for i in range(0, len(texts), 250)])
        if 'local' not in source:
            path.unlink(missing_ok=True)
    return np.concatenate(parts)


def tokenize_file(source, name, out, limit, pool, workers, folder):
    """Append one parquet file's documents to `out` as tokens; returns the tokens written (about `limit` at most)."""
    path = fetch(source, name, folder)
    reader, written = parquet_texts(path, 500, source.get('column', 'text')), 0
    encode = functools.partial(encode_documents, kind=source.get('filter'))
    with open(out, 'ab') as stream:

        while written < limit and (group := list(itertools.islice(reader, 4 * workers))):
            for tokens in pool.map(encode, group):
                tokens.tofile(stream)
                written += len(tokens)
    if 'local' not in source:
        path.unlink(missing_ok=True)
    return written


def attached(settings):
    """The folders under input_root, two levels deep, to show what Kaggle mounted."""
    root = Path(settings['input_root'])
    if not root.is_dir():
        return [f'{root} does not exist']
    return sorted(str(p.relative_to(root)) for p in root.glob('*/*')) or sorted(p.name for p in root.iterdir()) or ['(nothing attached)']


def prepare(settings):
    from multiprocessing import Pool
    checkpoint = find_file(settings, 'checkpoint.pt')
    print('attached inputs:', ', '.join(attached(settings)[:20]), flush=True)
    Tokenizer.load(settings['tokenizer'])
    print('tokenizing with', 'Rust (kani_tokenize)' if fast_tokenizer(settings['tokenizer']) else 'Python', flush=True)
    if checkpoint:
        print(f'continuing from {checkpoint}', flush=True)
    elif settings['continue_from_previous']:
        raise SystemExit('No earlier checkpoint.pt found, so this session would start over from zero. '
                         'Attach this notebook\'s latest output (Add Input → Your Work → Notebooks), '
                         'or set continue_from_previous to False to begin a new run.')
    else:
        print('no earlier checkpoint: starting a NEW run from zero', flush=True)
    banned = find_file(settings, 'contamination.npy')
    if banned:
        print(f'dropping documents that share 13 words with the test sets ({len(np.load(banned)):,} runs, {banned})', flush=True)
        run = Path(settings['run_dir'])
        if banned.parent != run:  # it travels on with this session's output
            run.mkdir(parents=True, exist_ok=True)
            banned = Path(shutil.copy(banned, run / 'contamination.npy'))
    elif settings['decontaminate']:
        raise SystemExit('decontaminate is on but no contamination.npy was found: attach the dataset that holds it '
                         '(made by "kani_advanced.py grams") beside the checkpoint.')
    progress = read_progress(settings)
    session, out = progress['session'], Path(settings['data_dir'])
    plan, validation, after = session_plan(settings, session, progress.get('next_file'))
    key = {'session': session, 'plan': plan, 'validation': validation, 'tokenizer': settings['tokenizer'],
           'session_tokens': settings['session_tokens'], 'validation_documents': settings['validation_documents'],
           'decontaminated': bool(banned)}
    key = json.loads(json.dumps(key))
    if (out / 'manifest.json').exists() and json.loads((out / 'manifest.json').read_text()).get('key') == key:
        print(f'session {session}: {out} is already prepared')
        return
    out.mkdir(parents=True, exist_ok=True)
    for name in ('manifest.json', 'train.bin', 'validation.bin'):
        (out / name).unlink(missing_ok=True)
    folder, started, workers = str(out / 'downloads'), time.time(), settings['workers'] or os.cpu_count()
    print(f'session {session}: {", ".join(name for _, name in plan)}', flush=True)
    with Pool(workers, initializer=_start_worker, initargs=(settings['tokenizer'], banned and str(banned))) as pool:
        validation_tokens(settings, validation, pool, folder).tofile(out / 'validation.bin')
        total = 0
        for index, name in plan:
            source = settings['sources'][index]

            share = (int(settings['session_tokens'] * source['share'] / source['per_session']) if multilingual(settings)
                     else settings['session_tokens'] // len(plan))
            written = tokenize_file(source, name, out / 'train.bin', share, pool, workers, folder)
            total += written
            print(f'  {name}: {written:,} tokens ({time.time() - started:.0f}s)', flush=True)
    validation_count = (out / 'validation.bin').stat().st_size // 2
    (out / 'manifest.json').write_text(json.dumps({'key': key, 'train_tokens': total, 'validation_tokens': validation_count,
                                                   'next_file': after}, indent=1))
    print(f'{total:,} training tokens and {validation_count:,} validation tokens in {time.time() - started:.0f}s')




def rope_tables(context, head_width):
    """kelus.model's rotary tables: angle = position * 10000^(-i/half), repeated for both halves."""
    half = head_width // 2
    angles = np.arange(context)[:, None] * 10000.0 ** (-np.arange(half) / half)
    cos = np.concatenate([np.cos(angles)] * 2, axis=1).astype(np.float32)
    sin = np.concatenate([np.sin(angles)] * 2, axis=1).astype(np.float32)
    return torch.from_numpy(cos), torch.from_numpy(sin)


def rotate(x, cos, sin):
    half = x.shape[-1] // 2
    return x * cos + torch.cat([-x[..., half:], x[..., :half]], dim=-1) * sin


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width, eps=1e-5), nn.LayerNorm(width, eps=1e-5)
        self.q, self.k, self.v, self.projection = (nn.Linear(width, width, bias=False) for _ in range(4))
        self.up, self.down = nn.Linear(width, 4 * width, bias=False), nn.Linear(4 * width, width, bias=False)

    def forward(self, x, cos, sin):
        b, t, d = x.shape
        h = self.norm1(x)
        q, k, v = (layer(h).view(b, t, self.heads, d // self.heads).transpose(1, 2) for layer in (self.q, self.k, self.v))
        q, k = rotate(q, cos[:t], sin[:t]).to(v.dtype), rotate(k, cos[:t], sin[:t]).to(v.dtype)
        mixed = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(b, t, d)
        x = x + self.projection(mixed)
        return x + self.down(F.gelu(self.up(self.norm2(x)), approximate='tanh'))


class Kani(nn.Module):
    def __init__(self, vocab, width, layers, heads, context, checkpointing=False):
        super().__init__()
        self.context, self.checkpointing = context, checkpointing
        self.token = nn.Embedding(vocab, width)
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(layers))
        self.final = nn.LayerNorm(width, eps=1e-5)
        cos, sin = rope_tables(context, width // heads)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)
        for name, p in self.named_parameters():
            if p.dim() == 2:
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * layers) if name.endswith(('projection.weight', 'down.weight')) else 0.02)

    def forward(self, ids):
        x = self.token(ids)
        for block in self.blocks:
            if self.checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, self.cos, self.sin, use_reentrant=False)
            else:
                x = block(x, self.cos, self.sin)
        return F.linear(self.final(x), self.token.weight)


def build(settings, vocab):
    return Kani(vocab, settings['width'], settings['layers'], settings['heads'], settings['context'], settings['gradient_checkpointing'])


def kani_parameters(net):
    """The model's weights under kelus.model's names and layouts (matrices are input x output)."""
    state = {k: v.detach().float().cpu().numpy() for k, v in net.state_dict().items()}
    values = {'token': state['token.weight'], 'final.gain': state['final.weight'], 'final.bias': state['final.bias']}
    for i in range(len(net.blocks)):
        p = f'blocks.{i}.'
        for norm in ('norm1', 'norm2'):
            values[f'block{i}.{norm}.gain'] = state[p + norm + '.weight']
            values[f'block{i}.{norm}.bias'] = state[p + norm + '.bias']
        for matrix in ('q', 'k', 'v', 'projection', 'up', 'down'):
            values[f'block{i}.{matrix}'] = np.ascontiguousarray(state[p + matrix + '.weight'].T)
    return values


def export(net, settings, tokenizer, progress, path):
    """A checkpoint in kelus's format ("kelus-b1" version 1), loadable by kelus.model.Model.load."""
    config = {'context': settings['context'], 'width': settings['width'], 'heads': settings['heads'],
              'layers': settings['layers'], 'seed': settings['seed'], 'vocab_size': tokenizer.vocab_size,
              'positions': 'rope', 'tie_embeddings': True, 'dropout': 0.0}
    metadata = {'display_name': 'Kani Advanced', 'steps': progress['step'], 'tokens_seen': progress['tokens_seen'],
                'sessions': progress['session'], 'stage': 'pretraining (raw text); needs instruction tuning before chat',
                'data': [s.get('repo', s.get('local')) for s in settings['sources']], 'trained_on': 'Kaggle (kaggle/kani_advanced.py)'}
    header = {'format': 'kelus-b1', 'version': 1, 'name': 'Kani Seal 3', 'config': config,
              'tokenizer': tokenizer.to_dict(), 'metadata': metadata}
    values = {'parameter.' + k: v for k, v in kani_parameters(net).items()}
    temp = Path(str(path) + '.tmp.npz')
    np.savez(temp, header=np.array(json.dumps(header)), **values)
    temp.replace(path)
    print(f'exported {path} ({sum(p.numel() for p in net.parameters()):,} parameters, '
          f'{progress["tokens_seen"] / 1e9:.2f}B tokens seen)', flush=True)


@torch.no_grad()
def generate(net, tokenizer, prompt, tokens=60, temperature=0.8, top_k=40, seed=0):
    device = next(net.parameters()).device
    ids = [BOS] + tokenizer.encode(prompt)
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(tokens):
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            logits = net(torch.tensor([ids[-net.context:]], device=device))[0, -1].float()
        logits[[PAD, BOS, USER, ASSISTANT]] = -float('inf')
        values, indices = logits.topk(top_k)
        token = int(indices[torch.multinomial((values / temperature).softmax(-1), 1, generator=generator)])
        if token == EOS:
            break
        ids.append(token)
    return tokenizer.decode(ids[1:])


PROMPTS = ['The capital of France is', 'Photosynthesis is the process by which', 'In 1969, astronauts',
           'To sort a list in Python, you can', 'The main causes of the First World War were']

PROMPTS_MULTILINGUAL = PROMPTS[:3] + ['Der Himmel ist blau, weil', 'La capitale de la France est', 'El agua hierve a',
                                      'O Brasil é o maior país', '한국의 수도는', '日本で一番高い山は', '中国的首都是', '台灣最高的山是']




def learning_rate(settings, tokens, rewarm_from=None):
    """Warmup, then constant, then a linear decay over the last `decay_fraction` of target_tokens. After a retokenize
    (a fresh optimizer from `rewarm_from` tokens) the rate ramps up again over `rewarm_tokens`."""
    rate = schedule(settings, tokens)
    if rewarm_from is not None and tokens - rewarm_from < settings['rewarm_tokens']:
        rate *= (tokens - rewarm_from + settings['batch_tokens']) / settings['rewarm_tokens']
    return rate


def schedule(settings, tokens):
    peak, target = settings['lr'], settings['target_tokens']
    if tokens < settings['warmup_tokens']:
        return peak * (tokens + settings['batch_tokens']) / settings['warmup_tokens']
    decay_start = target * (1 - settings['decay_fraction'])
    if tokens < decay_start:
        return peak
    done = min(1.0, (tokens - decay_start) / max(target - decay_start, 1))
    return peak * (1 - done * (1 - settings['min_lr_ratio']))


def windows(stream, count, context, rng):
    starts = rng.integers(0, len(stream) - context - 1, size=count)
    blocks = torch.from_numpy(np.stack([stream[s:s + context + 1] for s in starts]).astype(np.int64))
    return blocks[:, :-1], blocks[:, 1:]


class Blocks:
    """Training text in a shuffled order without repeats: every context-long block of the session's text is read once
    before any block comes back (random windows re-read about a third of the text and never reach another third).
    The position follows the steps taken this session, so a restarted run continues where it stopped."""

    def __init__(self, stream, context, micro_batch, seed, session, rank, world, done_micro_steps):
        self.stream, self.context, self.micro_batch = stream, context, micro_batch
        self.count = (len(stream) - 1) // context
        self.seed, self.session, self.rank, self.world = seed, session, rank, world
        self.position = done_micro_steps * micro_batch * world
        self.order, self.epoch = None, -1

    def block(self, index):
        epoch, within = divmod(index, self.count)
        if epoch != self.epoch:
            self.order, self.epoch = np.random.default_rng([self.seed, self.session, epoch]).permutation(self.count), epoch
        start = int(self.order[within]) * self.context
        return self.stream[start:start + self.context + 1]

    def next(self):
        first = self.position + self.rank * self.micro_batch
        self.position += self.world * self.micro_batch
        blocks = torch.from_numpy(np.stack([self.block(first + j) for j in range(self.micro_batch)]).astype(np.int64))
        return blocks[:, :-1], blocks[:, 1:]


@torch.no_grad()
def validate(net, stream, settings, device):
    net.eval()
    rng, total = np.random.default_rng(0), 0.0
    for _ in range(settings['eval_batches']):
        x, y = windows(stream, settings['micro_batch'], settings['context'], rng)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            logits = net(x.to(device))
        total += F.cross_entropy(logits.float().flatten(0, 1), y.to(device).flatten()).item()
    net.train()
    return total / settings['eval_batches']


def save_checkpoint(settings, net, opt, scaler, progress):
    run = Path(settings['run_dir'])
    temp = run / 'checkpoint.tmp'
    torch.save({'net': net.state_dict(), 'opt': opt and opt.state_dict(), 'scaler': scaler.state_dict(), 'progress': progress,
                'settings': settings}, temp)
    temp.replace(run / 'checkpoint.pt')
    (run / 'progress.json').write_text(json.dumps(progress, indent=1))


def train(settings):
    distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    cuda = torch.cuda.is_available()
    if distributed:
        torch.distributed.init_process_group('nccl' if cuda else 'gloo')
    rank = torch.distributed.get_rank() if distributed else 0
    world = torch.distributed.get_world_size() if distributed else 1
    local = int(os.environ.get('LOCAL_RANK', '0'))
    device = torch.device('cuda', local) if cuda else torch.device('cpu')
    if cuda:
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    main = rank == 0
    per_step = world * settings['micro_batch'] * settings['context']
    if settings['batch_tokens'] % per_step:
        raise SystemExit(f"batch_tokens must be a multiple of GPUs x micro_batch x context = {per_step}")
    accumulate = settings['batch_tokens'] // per_step

    tokenizer = Tokenizer.load(settings['tokenizer'])
    data = Path(settings['data_dir'])
    if not (data / 'manifest.json').exists():
        raise SystemExit(f'No prepared text in {data}: run "prepare" first (it may have been interrupted)')
    manifest = json.loads((data / 'manifest.json').read_text())
    prepared_session = manifest['key']['session']
    stream = np.memmap(data / 'train.bin', dtype='<u2', mode='r')
    held_out = np.memmap(data / 'validation.bin', dtype='<u2', mode='r')
    run = Path(settings['run_dir'])
    run.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(settings['seed'])
    net = build(settings, tokenizer.vocab_size).to(device)
    decay = [p for n, p in net.named_parameters() if p.dim() == 2]
    no_decay = [p for n, p in net.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': settings['weight_decay']}, {'params': no_decay, 'weight_decay': 0.0}],
                            lr=settings['lr'], betas=(0.9, 0.95), eps=1e-8, fused=cuda)
    scaler = torch.amp.GradScaler('cuda', enabled=cuda)
    progress = {'session': 0, 'tokens_seen': 0, 'step': 0, 'history': []}
    checkpoint = find_file(settings, 'checkpoint.pt')
    if checkpoint:
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        for key in ('width', 'layers', 'heads', 'context'):
            if state['settings'][key] != settings[key]:
                raise SystemExit(f'{checkpoint} has {key}={state["settings"][key]}; keep the model settings or use a new run_dir')
        if state['net']['token.weight'].shape[0] != tokenizer.vocab_size:
            raise SystemExit(f'{checkpoint} has a {state["net"]["token.weight"].shape[0]:,}-token vocabulary, the tokenizer '
                             f'{tokenizer.vocab_size:,}: run retokenize first')
        net.load_state_dict(state['net'])
        if state['opt'] is not None:
            opt.load_state_dict(state['opt'])
            scaler.load_state_dict(state['scaler'])
        progress = state['progress']
        if main:
            print(f'resumed from {checkpoint}: session {progress["session"]}, step {progress["step"]:,}, '
                  f'{progress["tokens_seen"] / 1e9:.2f}B tokens', flush=True)
    model = nn.parallel.DistributedDataParallel(net, device_ids=[local] if cuda else None) if distributed else net
    forward = torch.compile(model) if settings['compile'] else model
    if main and prepared_session != progress['session']:
        print(f'! the prepared text is for session {prepared_session}, not {progress["session"]}: run prepare first '
              'for fresh text (training continues on the old text)', flush=True)
    if main:
        print(json.dumps({'parameters': sum(p.numel() for p in net.parameters()), 'gpus': world, 'device': str(device),
                          'session': progress['session'], 'train_tokens_available': len(stream),
                          'tokens_per_step': settings['batch_tokens'], 'accumulate': accumulate}), flush=True)


    session_start = progress['history'][-1]['step'] if progress.get('history') else 0
    blocks = Blocks(stream, settings['context'], settings['micro_batch'], settings['seed'], progress['session'], rank, world,
                    max(0, progress['step'] - session_start) * accumulate)
    started = last_checkpoint = time.time()
    budget, stop, stopping = settings['hours'] * 3600, torch.zeros(1, device=device), False
    total, count, window = torch.zeros((), device=device), 0, time.time()
    validation_loss = None
    while not stopping:
        lr = learning_rate(settings, progress['tokens_seen'], progress.get('rewarm_from'))
        for group in opt.param_groups:
            group['lr'] = lr
        for micro in range(accumulate):
            x, y = blocks.next()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            sync = nullcontext() if not distributed or micro == accumulate - 1 else model.no_sync()
            with sync:
                with torch.autocast(device.type, dtype=torch.float16, enabled=cuda):
                    logits = forward(x)
                loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten()) / accumulate
                scaler.scale(loss).backward()
            total += loss.detach()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        progress['step'] += 1
        progress['tokens_seen'] += settings['batch_tokens']
        count += 1

        if progress['step'] % 10 == 0:
            done = time.time() - started > budget or progress['tokens_seen'] >= settings['target_tokens']
            stop.fill_(float(done))
            if distributed:
                torch.distributed.broadcast(stop, 0)
            stopping = bool(stop.item())
        if main and (progress['step'] % settings['eval_every'] == 0 or stopping):
            validation_loss = validate(net, held_out, settings, device)
        if main and (progress['step'] % 20 == 0 or stopping):
            rate = count * settings['batch_tokens'] / (time.time() - window)
            status = {'step': progress['step'], 'tokens': f"{progress['tokens_seen'] / 1e9:.3f}B", 'loss': round(total.item() / count, 4),
                      'validation_loss': validation_loss and round(validation_loss, 4), 'lr': f'{lr:.2e}', 'tokens_per_second': round(rate),
                      'hours_left': round(max(0.0, budget - (time.time() - started)) / 3600, 2)}
            if progress['step'] == 20 and cuda:
                status['gpu_memory_gb'] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)
            print(json.dumps(status), flush=True)
            (run / 'status.json').write_text(json.dumps({**status, 'state': 'running', 'updated': time.time()}))
            total, count, window = torch.zeros((), device=device), 0, time.time()
        if main and time.time() - last_checkpoint > settings['checkpoint_minutes'] * 60:
            save_checkpoint(settings, net, opt, scaler, progress)
            last_checkpoint = time.time()
            print(f'checkpoint at step {progress["step"]:,}', flush=True)

    if main:
        progress['history'].append({'session': progress['session'], 'step': progress['step'], 'tokens_seen': progress['tokens_seen'],
                                    'validation_loss': validation_loss, 'hours': round((time.time() - started) / 3600, 2)})
        if prepared_session == progress['session'] and 'next_file' in manifest:
            progress['next_file'] = manifest['next_file']
        progress['session'] += 1
        save_checkpoint(settings, net, opt, scaler, progress)
        export(net, settings, tokenizer, progress, run / 'kani-advanced.npz')
        complete = progress['tokens_seen'] >= settings['target_tokens']
        (run / 'status.json').write_text(json.dumps({'step': progress['step'], 'tokens_seen': progress['tokens_seen'],
                                                     'validation_loss': validation_loss, 'state': 'complete' if complete else 'stopped'}))
        net.eval()
        prompts = PROMPTS_MULTILINGUAL if tokenizer.version == 3 else PROMPTS
        samples = {prompt: generate(net, tokenizer, prompt) for prompt in prompts}
        (run / 'samples.json').write_text(json.dumps(samples, indent=1, ensure_ascii=False))
        for prompt, text in samples.items():
            print(f'--- {text}', flush=True)
        print(('target reached: ' if complete else 'session done: ') + f'{progress["tokens_seen"] / 1e9:.2f}B of '
              f'{settings["target_tokens"] / 1e9:.1f}B tokens, validation loss {validation_loss:.4f}', flush=True)
    if distributed:
        torch.distributed.destroy_process_group()


def retokenize(settings, checkpoint, old_path):
    """Move `checkpoint` (trained with the tokenizer at `old_path`) to settings['tokenizer'], as this run's start.

    The blocks keep everything they learned. Each new token's embedding (also the output layer, which is tied)
    starts as the mean of the old embeddings of the old tokens that spell it, so known words keep their meaning.
    The optimizer starts afresh, so training warms the learning rate up again (rewarm_tokens).
    """
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    old, new = Tokenizer.load(old_path), Tokenizer.load(settings['tokenizer'])
    table = state['net']['token.weight']
    if table.shape[0] != old.vocab_size:
        raise SystemExit(f'{checkpoint} does not use {old_path}')
    rows = []
    for token in range(new.vocab_size):
        if token < 261:
            rows.append(table[token])
            continue
        piece = new.pieces[token]
        try:
            ids = old.encode(piece.decode('utf-8'))
        except UnicodeDecodeError:
            ids = list(piece)
        rows.append(table[ids].mean(0))
    state['net']['token.weight'] = torch.stack(rows)
    progress = {**state['progress'], 'rewarm_from': state['progress']['tokens_seen'], 'retokenized_from': str(checkpoint)}
    run = Path(settings['run_dir'])
    run.mkdir(parents=True, exist_ok=True)
    torch.save({'net': state['net'], 'opt': None, 'scaler': None, 'progress': progress,
                'settings': {**state['settings'], 'tokenizer': settings['tokenizer']}}, run / 'checkpoint.tmp')
    (run / 'checkpoint.tmp').replace(run / 'checkpoint.pt')
    (run / 'progress.json').write_text(json.dumps(progress, indent=1))
    print(f'retokenized {checkpoint} ({old.vocab_size:,} tokens) to {new.vocab_size:,} tokens in {run}: step {progress["step"]:,}, '
          f'{progress["tokens_seen"] / 1e9:.2f}B tokens', flush=True)




def grow(settings, base, progress):
    """Start a deeper run from a trained Kani (`base`, kelus's npz format): its blocks in order, with a copy of every
    k-th block inserted after it until there are settings['layers'] blocks. A copy starts with its attention projection
    and feed-forward `down` at zero, so it adds nothing and the grown model is exactly the base (LLaMA Pro's block
    expansion); training then grows the copies into new layers. The optimizer starts fresh: train() ramps the
    learning rate up again over rewarm_tokens from `progress['rewarm_from']`."""
    with np.load(base) as z:
        header, values = json.loads(str(z['header'])), {k.removeprefix('parameter.'): z[k] for k in z.files if k != 'header'}
    config, layers = header['config'], settings['layers']
    for key in ('width', 'heads', 'context'):
        if config[key] != settings[key]:
            raise SystemExit(f'{base} has {key}={config[key]}, the settings {settings[key]}')
    extra, old = layers - config['layers'], config['layers']
    if not 0 < extra <= old or old % extra:
        raise SystemExit(f'grow adds 1 to {old} blocks, a whole fraction of {old}; asked for {extra}')
    every, order = old // extra, []
    for i in range(old):
        order.append((i, False))
        if i % every == every - 1:
            order.append((i, True))
    state = {'token.weight': values['token'], 'final.weight': values['final.gain'], 'final.bias': values['final.bias']}
    for j, (i, copy) in enumerate(order):
        for norm in ('norm1', 'norm2'):
            state[f'blocks.{j}.{norm}.weight'] = values[f'block{i}.{norm}.gain']
            state[f'blocks.{j}.{norm}.bias'] = values[f'block{i}.{norm}.bias']
        for matrix in ('q', 'k', 'v', 'projection', 'up', 'down'):
            weight = np.ascontiguousarray(values[f'block{i}.{matrix}'].T)
            state[f'blocks.{j}.{matrix}.weight'] = np.zeros_like(weight) if copy and matrix in ('projection', 'down') else weight
    net = build(settings, values['token'].shape[0])
    net.load_state_dict({k: torch.from_numpy(np.array(v, dtype=np.float32)) for k, v in state.items()})
    progress = {**progress, 'rewarm_from': progress['tokens_seen'],
                'grown_from': {'checkpoint': str(base), 'layers': old, 'tokens_seen': progress['tokens_seen'], 'copied_blocks': [i for i, c in order if c]}}
    run = Path(settings['run_dir'])
    run.mkdir(parents=True, exist_ok=True)
    save_checkpoint(settings, net, None, torch.amp.GradScaler('cuda', enabled=False), progress)
    print(f'grew {base} from {old} to {layers} blocks ({sum(p.numel() for p in net.parameters()):,} parameters); '
          f'copies after blocks {[i for i, c in order if c]}', flush=True)
    return net


def load_trained(settings):
    checkpoint = find_file(settings, 'checkpoint.pt')
    if checkpoint is None:
        raise SystemExit('No checkpoint.pt yet; train first.')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    tokenizer = Tokenizer.load(settings['tokenizer'])
    net = build({**settings, **{k: state['settings'][k] for k in ('width', 'layers', 'heads', 'context')}}, tokenizer.vocab_size)
    net.load_state_dict(state['net'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return net.to(device).eval(), tokenizer, state


def main():
    import sys
    if 'ipykernel' in sys.modules:
        print('This is the training script, not the notebook. On Kaggle use File → Import Notebook with '
              'kaggle/kani-advanced.ipynb, which writes this file and runs it.')
        return
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['prepare', 'train', 'sample', 'export', 'retokenize', 'grams', 'grow'])
    parser.add_argument('--settings', default='settings.json')
    parser.add_argument('--prompt', action='append', help='sample: text to continue (repeatable)')
    parser.add_argument('--tokens', type=int, default=80)
    parser.add_argument('--checkpoint', help='retokenize: the checkpoint.pt to move to the new vocabulary')
    parser.add_argument('--old-tokenizer', help="retokenize: that checkpoint's tokenizer")
    parser.add_argument('--tests', help="grams: the folder of kelus.standard_eval's test files")
    parser.add_argument('--out', help='grams: where to write contamination.npy')
    parser.add_argument('--base', help='grow: the trained kani-advanced.npz to grow')
    parser.add_argument('--progress', help="grow: the base's progress.json (session, step, tokens_seen, next_file)")
    args = parser.parse_args()
    if args.command == 'grams':
        if not (args.tests and args.out):
            raise SystemExit('grams needs --tests and --out')
        write_grams(args.tests, args.out)
        return
    settings = load_settings(args.settings)
    if args.command == 'prepare':
        prepare(settings)
    elif args.command == 'train':
        train(settings)
    elif args.command == 'grow':
        if not (args.base and args.progress):
            raise SystemExit('grow needs --base and --progress')
        grow(settings, args.base, json.loads(Path(args.progress).read_text()))
    elif args.command == 'retokenize':
        if not (args.checkpoint and args.old_tokenizer):
            raise SystemExit('retokenize needs --checkpoint and --old-tokenizer')
        retokenize(settings, args.checkpoint, args.old_tokenizer)
    elif args.command == 'export':
        net, tokenizer, state = load_trained(settings)
        Path(settings['run_dir']).mkdir(parents=True, exist_ok=True)
        export(net, {**settings, **state['settings'], 'run_dir': settings['run_dir']}, tokenizer, state['progress'],
               Path(settings['run_dir']) / 'kani-advanced.npz')
    else:
        net, tokenizer, _ = load_trained(settings)
        for i, prompt in enumerate(args.prompt or (PROMPTS_MULTILINGUAL if tokenizer.version == 3 else PROMPTS)):
            print(f'--- {generate(net, tokenizer, prompt, tokens=args.tokens, seed=i)}')


if __name__ == '__main__':
    main()
