"""The BPE trainer and the multilingual pattern (version 3)."""
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from kelus.tokenizer import VOCAB_SIZE, BPETokenizer

ROOT = Path(__file__).resolve().parent.parent
TEXTS = ['Das Wetter ist heute schön, und wir gehen später in den Park.', 'Il fait beau aujourd’hui à Paris, allons au café.',
         'Hoy hace buen tiempo en Madrid; vamos a la playa.', 'O tempo está bom hoje em Lisboa, vamos à praia.',
         '오늘 날씨가 좋네요. 공원에 갈까요?', '今天天气很好，我们去公园吧。', '今天天氣很好，我們去公園吧。', '今日はいい天気ですね。公園に行きましょう。',
         'The weather is nice today, so we are going to the park.', 'def add(a, b):\n    return a + b\n', '```python\nprint("hi")\n```']


def reference(texts, vocab_size, version):
    """The original trainer: recount every pair after each merge (slow, obviously correct)."""
    words = Counter(tuple(piece.encode('utf-8')) for text in texts for piece in BPETokenizer.patterns[version].findall(text))
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
            updated[BPETokenizer.merge(word, pair, token)] += count
        words = updated
    return merges


class TokenizerTests(unittest.TestCase):
    def test_fast_trainer_learns_exactly_the_reference_merges(self):
        for version in (2, 3):
            for size in (300, 600, 1200):
                texts = TEXTS * 3 + ['aaaa aaaaa', 'banana bandana']
                self.assertEqual(BPETokenizer.train(texts, size, version).merges, reference(texts, size, version), (version, size))

    def test_version_3_keeps_words_in_every_language_and_round_trips(self):
        pattern = BPETokenizer.patterns[3]
        self.assertEqual(pattern.findall('Das Wetter ist schön.'), ['Das', ' Wetter', ' ist', ' schön', '.'])
        self.assertEqual(pattern.findall('오늘 날씨가 좋네요.'), ['오늘', ' 날씨가', ' 좋네요', '.'])
        self.assertEqual(pattern.findall('今天天气很好，我们去公园吧。'), ['今天天气很好', '，', '我们去公园吧', '。'])
        self.assertEqual(pattern.findall('今日はいい天気ですね。'), ['今日', 'はいい', '天気', 'ですね', '。'])
        tokenizer = BPETokenizer.train(TEXTS * 5, 900, version=3)
        for text in TEXTS + ['x² 😀 €5 Ω ñ', 'mixed 한국어 and 中文 and español']:
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        self.assertLess(len(tokenizer.encode('今天天气很好')), len('今天天气很好'.encode('utf-8')) / 2)
        self.assertEqual(BPETokenizer(tokenizer.merges, 3).encode('오늘 날씨'), tokenizer.encode('오늘 날씨'))

    @unittest.skipUnless(shutil.which('node'), 'Node.js runs the browser engine')
    def test_version_3_splits_and_encodes_the_same_in_the_browser_engine(self):
        tokenizer = BPETokenizer.train(TEXTS * 5, 900, version=3)
        texts = TEXTS + ['x² 😀 €5 Ω ñ　全角 スペース', 'Ça va? Ünïcödé — “quotes” ½ 𠀋', '﻿bom\xa0nbsp']
        script = ("import { Tokenizer } from './site/b1.js'; const [spec, texts] = JSON.parse(process.argv[1]);"
                  "const t = new Tokenizer(spec); console.log(JSON.stringify(texts.map(x => t.encode(x))));")
        out = subprocess.run(['node', '--input-type=module', '-e', script, json.dumps([{'kind': 'bpe', 'version': 3, 'merges': tokenizer.merges}, texts])],
                             cwd=ROOT, check=True, capture_output=True, text=True).stdout
        self.assertEqual(json.loads(out), [tokenizer.encode(text) for text in texts])


if __name__ == '__main__':
    unittest.main()
