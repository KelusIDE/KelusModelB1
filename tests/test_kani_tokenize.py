"""kaggle/kani_tokenize (Rust) gives exactly kelus.tokenizer.BPETokenizer's tokens, for both pattern versions.

Needs the module built (pip install ./kaggle/kani_tokenize, with Rust installed); otherwise this skips.
"""
import importlib.util
import json
from pathlib import Path
import unittest
import tempfile

import numpy as np

from kelus.tokenizer import BOS, EOS, from_dict

ROOT = Path(__file__).resolve().parent.parent
HAVE = importlib.util.find_spec('kani_tokenize') is not None
TEXTS = ['The moon goes around the earth once a month.', '  two  spaces\n\n\nthen\ttabs', '```python\nprint(1)\n```',
         'Ça été ß über ñandú', '오늘 날씨가 좋아서 공원에 갔어요.', '今天天氣很好，我們去公園吧。', '日本で一番高い山は富士山です。',
         'separators\x1c\x1d\x1e\x1f and spaces\x85\xa0 　end', 'x = [i ** 2 for i in range(10)]  # 0, 1, 4', '',

         ''.join(chr(c) for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF)]


@unittest.skipUnless(HAVE, 'needs kani_tokenize built')
class KaniTokenizeTests(unittest.TestCase):
    def tokenizers(self, name):
        value = json.loads((ROOT / 'kaggle' / name).read_text())
        import kani_tokenize
        return from_dict(value), kani_tokenize.Tokenizer([tuple(pair) for pair in value['merges']], value['version'])

    def test_same_tokens_as_python(self):
        for name in ('kani_tokenizer.json', 'kani_tokenizer_multilingual.json'):
            python, rust = self.tokenizers(name)
            self.assertEqual(rust.vocab_size, python.vocab_size)
            for text in TEXTS:
                self.assertEqual(rust.encode(text), python.encode(text), f'{name}: {text[:40]!r}')

    def test_documents_are_wrapped_and_empty_ones_skipped(self):
        python, rust = self.tokenizers('kani_tokenizer.json')
        got = np.frombuffer(rust.encode_documents(['Hello there.', '', None, 'Bye.']), dtype='<u2').tolist()
        self.assertEqual(got, [BOS, *python.encode('Hello there.'), EOS, BOS, *python.encode('Bye.'), EOS])

    def test_bad_vocabularies_are_refused(self):
        import kani_tokenize
        for merges, version in (([(1, 2)], 9), ([(256, 1)], 2), ([(1, 999)], 2), ([(1, 2), (1, 2)], 2)):
            with self.assertRaises(ValueError):
                kani_tokenize.Tokenizer(merges, version)

    def test_rust_dataset_preserves_tokens_masks_and_samples(self):
        from kelus.data import Dataset
        python, _ = self.tokenizers('kani_tokenizer.json')
        records = [dict(split=split, messages=[dict(role='user', content=q), dict(role='assistant', content=a)])
                   for split, q, a in [('train', 'Write a paragraph about rain.', 'Rain brings water to plants and fills rivers.'),
                                       ('train', 'I like cats.', 'What do you enjoy about cats?'),
                                       ('train', '```python\nprint(1)\n```', 'This prints the number one.'),
                                       ('validation', '오늘은?', '오늘 날씨가 좋아요.')]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.jsonl'
            path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
            ordinary = Dataset(path, tokenizer=python)
            fast = Dataset(path, tokenizer=python, rust_tokenizer=True)
            self.assertEqual(fast.fingerprint, ordinary.fingerprint)
            for name in ('training', 'validation'):
                for (x, y), (a, b) in zip(getattr(ordinary, name), getattr(fast, name)):
                    np.testing.assert_array_equal(x, a)
                    np.testing.assert_array_equal(y, b)
            for a, b in zip(ordinary.batch('train', 3, 128, np.random.default_rng(42)),
                            fast.batch('train', 3, 128, np.random.default_rng(42))):
                np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
