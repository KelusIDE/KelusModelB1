"""kaggle/kani_advanced.py end to end with a tiny model: an English run, then retokenized to the multilingual
vocabulary and trained on a weighted mix of languages whose documents are filtered by script.

Needs PyTorch and pyarrow (the cloud machines have them; the project's own environment does not, so this skips):
    python -m unittest tests.test_kani_advanced        # with an environment that has torch and pyarrow
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from kelus.tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / 'kaggle' / 'kani_advanced.py'
HAVE = all(importlib.util.find_spec(name) for name in ('torch', 'pyarrow'))
ENGLISH = ['The moon goes around the earth once a month.', 'Plants make food from sunlight, water and air.',
           'A river flows from the mountains to the sea.', 'Python lists keep items in order.'] * 40
KOREAN = ['오늘 날씨가 좋아서 공원에 갔어요.', '한국의 수도는 서울입니다.', '저는 매일 아침 커피를 마셔요.'] * 40
TRADITIONAL = ['今天天氣很好，我們去公園吧。', '這個問題很難，我們一起學習。', '台灣的電車很方便，時間很準。'] * 40
SIMPLIFIED = ['今天天气很好，我们去公园吧。', '这个问题很难，我们一起学习。'] * 40
MARKER = 'Zebracorn'


def write_parquet(path, texts):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({'text': texts}), path, row_group_size=50)


@unittest.skipUnless(HAVE, 'needs torch and pyarrow')
class KaniAdvancedTests(unittest.TestCase):
    def run_script(self, *args):
        try:
            result = subprocess.run([sys.executable, str(SCRIPT), *args], cwd=self.root, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired as error:
            self.fail(f'{args[0]} did not finish: {str(error.stdout)[-2000:]} {str(error.stderr)[-3000:]}')
        self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-3000:])
        return result.stdout

    def settings(self, name, **extra):
        value = {'run_dir': str(self.root / 'runs' / name), 'input_root': str(self.root / 'input'), 'data_dir': str(self.root / 'data' / name),
                 'width': 32, 'layers': 2, 'heads': 2, 'context': 32, 'micro_batch': 2, 'batch_tokens': 256,
                 'session_tokens': 12000, 'validation_documents': 60, 'hours': 0.0015, 'checkpoint_minutes': 0,
                 'eval_every': 4, 'eval_batches': 2, 'warmup_tokens': 2048, 'target_tokens': 10 ** 9, 'workers': 2, **extra}
        path = self.root / f'{name}-settings.json'
        path.write_text(json.dumps(value))
        return str(path)

    def test_english_run_moves_to_the_multilingual_vocabulary_and_keeps_training(self):
        with tempfile.TemporaryDirectory() as root:
            self.root = Path(root)
            files = self.root / 'files'
            for i in range(3):
                write_parquet(files / f'en/{i}.parquet', ENGLISH[i::3] * 3)
                write_parquet(files / f'ko/{i}.parquet', [text for pair in zip(KOREAN, [f'{MARKER} {t}' for t in ENGLISH]) for text in pair])
                write_parquet(files / f'zh/{i}.parquet', TRADITIONAL + SIMPLIFIED)
            english = BPETokenizer.train(ENGLISH, 320, version=2)
            mixed = BPETokenizer.train(ENGLISH + KOREAN + TRADITIONAL + SIMPLIFIED, 700, version=3)
            (self.root / 'english.json').write_text(json.dumps(english.to_dict()))
            (self.root / 'mixed.json').write_text(json.dumps(mixed.to_dict()))


            plain = self.settings('english', tokenizer=str(self.root / 'english.json'),
                                  sources=[{'local': str(files / 'en/*.parquet'), 'per_session': 1}])
            self.run_script('prepare', '--settings', plain)
            self.assertIn('session done', self.run_script('train', '--settings', plain))
            checkpoint = self.root / 'runs/english/checkpoint.pt'


            sources = [{'local': str(files / 'en/*.parquet'), 'per_session': 1, 'share': 0.5},
                       {'local': str(files / 'ko/*.parquet'), 'per_session': 1, 'share': 0.25, 'filter': 'hangul'},
                       {'local': str(files / 'zh/*.parquet'), 'per_session': 1, 'share': 0.25, 'filter': 'traditional', 'offset': 1}]
            mixed_settings = self.settings('mixed', tokenizer=str(self.root / 'mixed.json'), sources=sources)
            out = self.run_script('retokenize', '--settings', mixed_settings, '--checkpoint', str(checkpoint),
                                  '--old-tokenizer', str(self.root / 'english.json'))
            self.assertIn(f'to {mixed.vocab_size} tokens', out)
            import torch
            before = torch.load(checkpoint, weights_only=True)
            after = torch.load(self.root / 'runs/mixed/checkpoint.pt', weights_only=True)
            old, new = before['net']['token.weight'], after['net']['token.weight']
            self.assertEqual(tuple(new.shape), (mixed.vocab_size, 32))
            self.assertTrue(torch.equal(new[:261], old[:261]))
            word = next(t for t in range(261, mixed.vocab_size) if mixed.pieces[t] == b' moon' or mixed.pieces[t].startswith(b' mo'))
            ids = english.encode(mixed.pieces[word].decode('utf-8'))
            self.assertTrue(torch.allclose(new[word], old[ids].mean(0)))
            self.assertIsNone(after['opt'])
            self.assertEqual(after['progress']['rewarm_from'], before['progress']['tokens_seen'])
            for key in before['net']:
                if key != 'token.weight':
                    self.assertTrue(torch.equal(before['net'][key], after['net'][key]), key)


            self.run_script('prepare', '--settings', mixed_settings)
            data = self.root / 'data/mixed'
            manifest = json.loads((data / 'manifest.json').read_text())
            self.assertEqual(len(manifest['key']['validation']), 3)
            text = mixed.decode(np.fromfile(data / 'train.bin', dtype='<u2').tolist())
            self.assertIn('서울', text)
            self.assertIn('我們', text)
            self.assertNotIn(MARKER, text)
            self.assertNotIn('我们', text)
            out = self.run_script('train', '--settings', mixed_settings)
            self.assertIn('resumed from', out)
            first = json.loads(next(line for line in out.splitlines() if line.startswith('{"step"')))
            self.assertLess(float(first['lr']), 1e-4)
            state = torch.load(self.root / 'runs/mixed/checkpoint.pt', weights_only=True)
            self.assertIsNotNone(state['opt'])


            from kelus.model import Model
            model, header, _ = Model.load(self.root / 'runs/mixed/kani-advanced.npz')
            self.assertEqual(header['tokenizer']['version'], 3)
            self.assertEqual(model.config.vocab_size, mixed.vocab_size)
            self.assertIsInstance(''.join(model.generate([{'role': 'user', 'content': '오늘 날씨'}], max_tokens=4, temperature=0)), str)

    def test_documents_sharing_13_words_with_a_test_set_are_dropped(self):
        with tempfile.TemporaryDirectory() as root:
            self.root = Path(root)
            tests = self.root / 'tests'
            tests.mkdir()
            leaked = 'The old lighthouse keeper climbed the winding stairs every night to light the lamp for the ships'
            rows = {'hellaswag_val.jsonl': {'activity_label': 'Cooking', 'ctx': 'A man cracks an egg.', 'endings': ['He fries it.', 'He sings.']},
                    'arc_easy_test.jsonl': {'question': 'What do plants need?', 'choices': {'text': ['Light', 'Rocks']}},
                    'arc_challenge_test.jsonl': {'question': 'Why is the sky blue?', 'choices': {'text': ['Scattering', 'Paint']}},
                    'piqa_valid.jsonl': {'goal': 'Open a jar', 'sol1': 'Twist the lid', 'sol2': 'Eat the lid'},
                    'winogrande_val.jsonl': {'sentence': 'Sam beat Ana so _ won.', 'option1': 'Sam', 'option2': 'Ana'},
                    'lambada_test.jsonl': {'text': leaked}}
            for name, row in rows.items():
                (tests / name).write_text(json.dumps(row) + '\n')
            run = self.root / 'runs/clean'
            run.mkdir(parents=True)
            self.assertIn('distinct 13-word runs', self.run_script('grams', '--tests', str(tests), '--out', str(run / 'contamination.npy')))
            files = self.root / 'files'
            texts = ENGLISH + [f'{MARKER} said: {leaked.lower()}, and then went home.']
            for i in range(2):
                write_parquet(files / f'en/{i}.parquet', texts)
            tokenizer = BPETokenizer.train(ENGLISH + [MARKER], 320, version=2)
            (self.root / 'tokenizer.json').write_text(json.dumps(tokenizer.to_dict()))
            settings = self.settings('clean', tokenizer=str(self.root / 'tokenizer.json'), decontaminate=True,
                                     sources=[{'local': str(files / 'en/*.parquet'), 'per_session': 1}])
            self.assertIn('dropping documents that share 13 words', self.run_script('prepare', '--settings', settings))
            text = tokenizer.decode(np.fromfile(self.root / 'data/clean/train.bin', dtype='<u2').tolist())
            self.assertIn('The moon goes around', text)
            self.assertNotIn(MARKER, text)
            # Files uploaded loose to a dataset (no kani-advanced folder) are found too.
            loose = self.root / 'input/datasets/someone/kani-start'
            loose.mkdir(parents=True)
            (run / 'contamination.npy').rename(loose / 'contamination.npy')
            (self.root / 'data/clean/manifest.json').unlink()
            self.assertIn('kani-start/contamination.npy', self.run_script('prepare', '--settings', settings))
            (loose / 'contamination.npy').unlink()
            (run / 'contamination.npy').unlink()
            result = subprocess.run([sys.executable, str(SCRIPT), 'prepare', '--settings', settings], cwd=self.root, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('no contamination.npy was found', result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
