"""Kani's multilingual tokenizer: BPE pattern version 3 learned from nine languages in balance.

English (FineWeb-Edu, Cosmopedia v2 and Kani's own chat, code and reading data), Korean, German, French, Spanish,
Japanese, Portuguese, and Chinese in both scripts (FineWeb-2), about the same number of characters each. Only the
first part of chosen row groups is read from Hugging Face, so nothing is downloaded whole. Documents must be mostly in
their language's script (FineWeb-2's labels are not perfect); Mandarin is split into Simplified and Traditional by
characters that differ between the two scripts.

    python3 data/make_multilingual_tokenizer.py                 # data/multilingual-v1/: samples, tokenizer.json, report.json
    python3 data/make_multilingual_tokenizer.py --chars 1000000 --vocab 8192 --out /tmp/try   # a quick trial
"""
import argparse
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kelus.tokenizer import BPETokenizer  # noqa: E402

FINEWEB2 = 'datasets/HuggingFaceFW/fineweb-2/data/{}/train/000_00000.parquet'
SOURCES = {
    'en': ('datasets/HuggingFaceFW/fineweb-edu/sample/10BT/000_00000.parquet', None),
    'ko': (FINEWEB2.format('kor_Hang'), 'hangul'),
    'de': (FINEWEB2.format('deu_Latn'), None),
    'fr': (FINEWEB2.format('fra_Latn'), None),
    'es': (FINEWEB2.format('spa_Latn'), None),
    'pt': (FINEWEB2.format('por_Latn'), None),
    'ja': (FINEWEB2.format('jpn_Jpan'), 'kana'),
    'zh-Hans': (FINEWEB2.format('cmn_Hani'), 'simplified'),
    'zh-Hant': (FINEWEB2.format('cmn_Hani'), 'traditional'),
}
COSMOPEDIA = 'datasets/HuggingFaceTB/smollm-corpus/cosmopedia-v2/train-00000-of-00104.parquet'

PAIRS = ('们們 这這 说說 时時 国國 会會 来來 对對 发發 后後 个個 为為 学學 经經 动動 现現 机機 开開 关關 长長 问問 见見 进進 实實 两兩 车車 东東 门門 '
         '应應 电電 气氣 头頭 从從 书書 难難 爱愛 让讓 认認 识識 读讀 写寫 听聽 买買 卖賣 钱錢 银銀 医醫 药藥 话話 语語 请請 谢謝 体體 题題 华華 业業 '
         '产產 网網 线線 结結 给給 红紅 绿綠 鸟鳥 鱼魚 马馬 龙龍 风風 飞飛 云雲 万萬 与與 区區 级級 历歷 战戰 农農 观觀 欢歡 乐樂 节節 质質 资資 '
         '务務 处處 员員 种種 总總 统統 系係 组組 织織 设設 计計 论論 议議 记記 讲講 该該 报報 场場 进進 还還 这這 运運 远遠 选選 边邊 导導 '
         '将將 岁歲 热熱 现現 图圖 团團 园園 圆圓 广廣 庆慶 应應 张張 强強 归歸 当當 录錄 彻徹 忆憶 态態 怀懷 恶惡 悬懸 惊驚 愿願 戏戲 户戶').split()
SIMPLIFIED, TRADITIONAL = {p[0] for p in PAIRS}, {p[1] for p in PAIRS}


def script_share(text, kind):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    if kind == 'hangul':
        return sum('가' <= c <= '힣' for c in letters) / len(letters)
    if kind == 'kana':
        kana = sum('぀' <= c <= 'ヿ' for c in letters)
        return (kana + sum('一' <= c <= '鿿' for c in letters)) / len(letters) if kana >= 20 else 0.0
    han = sum('一' <= c <= '鿿' for c in letters) / len(letters)
    simplified, traditional = sum(c in SIMPLIFIED for c in text), sum(c in TRADITIONAL for c in text)
    if simplified + traditional < 3 or any('぀' <= c <= 'ヿ' for c in text[:2000]):
        return 0.0
    wanted = simplified if kind == 'simplified' else traditional
    return han if wanted >= 4 * (simplified + traditional - wanted) else 0.0


def read(path, chars, kind, rng, skip=0, fs=None):
    """Documents from spread-out row groups of `path` until `chars` characters, mostly in the script `kind`."""
    import pyarrow.parquet as pq
    texts, total = [], 0
    with fs.open(path, 'rb', block_size=8 * 2 ** 20) as handle:
        parquet = pq.ParquetFile(handle)
        groups = list(range(parquet.num_row_groups))
        rng.shuffle(groups)
        for group in groups[skip:]:
            for text in parquet.read_row_group(group, columns=['text']).column('text').to_pylist():
                if kind and script_share(text, kind) < 0.6:
                    continue
                texts.append(text[:20000])
                total += len(texts[-1])
                if total >= chars:
                    return texts
    return texts


def kani_texts(limit):
    """Kani's own conversations (chat, code, reading with citations, places), so its formats get good tokens."""
    texts, total = [], 0
    for name in ('instruct-v9.jsonl', 'reading-v1.jsonl'):
        for line in (ROOT / 'data' / name).open(encoding='utf-8'):
            for message in json.loads(line)['messages']:
                texts.append(message['content'])
                total += len(message['content'])
            if total >= limit:
                return texts
    return texts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--chars', type=int, default=8_000_000, help='characters of web text per language')
    parser.add_argument('--vocab', type=int, default=32768)
    parser.add_argument('--out', type=Path, default=ROOT / 'data/multilingual-v1')
    args = parser.parse_args()
    from huggingface_hub import HfFileSystem
    fs, rng = HfFileSystem(), random.Random(2408)
    args.out.mkdir(parents=True, exist_ok=True)
    samples, held_out = {}, {}
    for language, (path, kind) in SOURCES.items():
        started = time.time()
        texts = read(path, args.chars + 200_000, kind, random.Random(f'{language}-{rng.random()}'), fs=fs)
        held_out[language], samples[language] = texts[:40], texts[40:]
        if language == 'en':
            samples[language] += read(COSMOPEDIA, args.chars // 2, None, random.Random(7), fs=fs) + kani_texts(args.chars // 2)
        size = sum(map(len, samples[language]))
        (args.out / f'{language}.txt').write_text('\n\n'.join(samples[language]), encoding='utf-8')
        print(f'{language}: {len(samples[language]):,} documents, {size / 1e6:.1f}M characters ({time.time() - started:.0f}s)', flush=True)
    started = time.time()
    tokenizer = BPETokenizer.train([text for texts in samples.values() for text in texts], args.vocab, version=3)
    print(f'{tokenizer.vocab_size:,} tokens learned in {(time.time() - started) / 60:.1f} min', flush=True)
    (args.out / 'tokenizer.json').write_text(json.dumps(tokenizer.to_dict()), encoding='utf-8')

    from kelus.tokenizer import from_dict
    english = from_dict(json.loads((ROOT / 'data/big-v1/tokenizer.json').read_text()))
    report = {}
    for language, texts in held_out.items():
        chars = sum(map(len, texts))
        new, old = sum(len(tokenizer.encode(t)) for t in texts), sum(len(english.encode(t)) for t in texts)
        report[language] = {'characters_per_token': round(chars / new, 2), 'english_tokenizer': round(chars / old, 2)}
        print(f'{language:8} {chars / new:5.2f} characters per token (the English tokenizer: {chars / old:4.2f})')
    (args.out / 'report.json').write_text(json.dumps({'vocab': tokenizer.vocab_size, 'held_out': report}, indent=1) + '\n')


if __name__ == '__main__':
    main()
