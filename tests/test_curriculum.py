import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from kelus.curriculum import coverage, family_split, normalize, prepare, read_records
from kelus.data import Dataset
from kelus.export_web import export
from kelus.languages import canonical, scope, verification_problem, verification_template
from kelus.model import Model
from kelus.tokenizer import BPETokenizer, ByteTokenizer
from kelus.train import main as train


def record(name, code='python', language='en', split='train', answer=None):
    return {'family': name, 'code_language': code, 'reply_language': language, 'split': split,
            'reviewed': True, 'source': {'id':'unit-fixture', 'license':'original test fixture', 'reference':'tests/test_curriculum.py'},
            'messages': [{'role':'user','content':f'{name}: {code} / {language}'},
                         {'role':'assistant','content':answer or 'X ' + name}]}


def save(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


class CurriculumTests(unittest.TestCase):
    def test_aliases_keep_distinct_languages_and_scripts(self):
        self.assertEqual(len(scope()['programming']), 23)
        self.assertEqual(len(scope()['chat']), 7)
        self.assertEqual(canonical('ps1', 'programming'), 'powershell')
        self.assertEqual(canonical('C++', 'programming'), 'cpp')
        self.assertEqual(canonical('C#', 'programming'), 'csharp')
        self.assertEqual(canonical('france', 'chat'), 'fr')
        self.assertNotEqual(canonical('简体中文','chat'), canonical('繁體中文','chat'))
        with self.assertRaises(ValueError): canonical('shell', 'programming')
        with self.assertRaises(ValueError): canonical('Chinese', 'chat')

    def test_translation_families_do_not_leak_across_splits(self):
        rows = [record('same-task', language=language) for language in scope()['chat']]
        for row in rows: row.pop('split')
        result = normalize(rows, assign_splits=True)
        self.assertEqual({r['split'] for r in result}, {family_split('same-task')})
        result[0]['split'] = 'test' if result[1]['split'] != 'test' else 'train'
        with self.assertRaisesRegex(ValueError, 'crosses splits'): normalize(result)
        duplicate = record('different-task', split='test')
        duplicate['messages'] = rows[0]['messages']
        with self.assertRaisesRegex(ValueError, 'conflicting'): normalize([record('same-task'), duplicate])

    def test_preparation_keeps_test_separate_reports_gaps_and_protects_output(self):
        rows = [record('task-' + str(i), split=s) for i,s in enumerate(['train','train','validation','test'])]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / 'records.jsonl'; out = root / 'ready'
            save(source, rows)
            report = prepare([source], out)
            self.assertEqual(len(read_records(out / 'instruction.jsonl')), 3)
            self.assertEqual(len(read_records(out / 'heldout.jsonl')), 1)
            self.assertIn('cobol', report['splits']['train']['missing_code'])
            self.assertIn('ko', report['splits']['train']['missing_chat'])
            with self.assertRaisesRegex(ValueError, 'already exists'): prepare([source], out)
            with self.assertRaisesRegex(ValueError, 'missing'): prepare([source], root / 'strict', True)
            self.assertFalse((root / 'strict').exists())
            rows[0]['reviewed'] = False; save(source, rows)
            with self.assertRaisesRegex(ValueError, 'Unreviewed'): prepare([source], root / 'unchecked')

    def test_balanced_sampling_and_heldout_rejection(self):
        rows = [record('many-' + str(i), answer='X') for i in range(90)]
        rows += [record('rare', code='rust', language='ko', answer='Y'), record('validation', split='validation')]
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / 'data.jsonl'; save(file, rows)
            balanced = Dataset(file, balance_languages=True)
            plain = Dataset(file)
            def sampled_y(data):
                _, labels = data.batch('train', 1000, 128, np.random.default_rng(12))
                return int(np.count_nonzero(np.any(labels == ord('Y'), axis=1)))
            self.assertTrue(420 < sampled_y(balanced) < 580)
            self.assertLess(sampled_y(plain), 40)
            # Restoring RNG state reproduces both bucket choice and window choice.
            first = np.random.default_rng(9); second = np.random.default_rng()
            second.bit_generator.state = first.bit_generator.state
            for a,b in zip(balanced.batch('train',4,32,first), balanced.batch('train',4,32,second)):
                np.testing.assert_array_equal(a,b)
            rows.append(record('heldout', split='test')); save(file, rows)
            with self.assertRaisesRegex(ValueError, 'Held-out'): Dataset(file, balance_languages=True)

    def test_seven_chat_targets_round_trip_in_tokenizer(self):
        texts = ['Explain this code.', '이 코드를 설명해 줘.', 'Explique ce code.', 'Explica este código.',
                 'Erkläre diesen Code.', '请用简体中文解释这段代码。', '請用繁體中文解釋這段程式碼。']
        tokenizer = BPETokenizer.train(['ordinary English training text'] * 5, vocab_size=280, version=2)
        for text in texts:
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        # Round-trip is a representation test, not a claim of model fluency.

    def test_balanced_training_resume_and_export_preserve_coverage(self):
        rows = [record('train-a'), record('train-b', code='ps1', language='ko', answer='안녕하세요'),
                record('validation-a', split='validation'), record('validation-b', code='C#', language='fr', split='validation')]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); data = root / 'data.jsonl'; checkpoint = root / 'b1.npz'; save(data, rows)
            args = ['--data',str(data),'--out',str(checkpoint),'--steps','1','--context','16','--width','8',
                    '--layers','1','--heads','2','--batch','1','--bpe-vocab','0','--patience','0','--dropout','0']
            with contextlib.redirect_stdout(io.StringIO()):
                train(args + ['--balanced-languages'])
                train(args + ['--resume',str(checkpoint)])
            _, header, _ = Model.load(checkpoint)
            self.assertEqual(header['metadata']['steps'], 2)
            self.assertIs(header['metadata']['balanced_languages'], True)
            report = header['metadata']['language_training']
            self.assertEqual(report['splits']['train']['counts']['code:powershell'], 1)
            self.assertEqual(export(checkpoint, root / 'web')['language_training'], report)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train(args + ['--resume',str(checkpoint),'--no-balanced-languages'])

    def test_blank_or_partial_verification_never_counts_as_support(self):
        template = verification_template('checkpoint')
        self.assertIsNotNone(verification_problem(template, 'checkpoint'))
        for category in ('programming','chat'):
            for entry in template[category].values():
                entry.update(reviewer='unit fixture', evidence='Simulated evidence, no real model claim')
                entry['checks'] = dict.fromkeys(entry['checks'], True)
        self.assertIsNone(verification_problem(template, 'checkpoint'))
        template['chat']['zh-Hant']['checks']['explain_code'] = False
        self.assertIn('zh-Hant', verification_problem(template, 'checkpoint'))
        self.assertIn('another checkpoint', verification_problem(template, 'changed'))


if __name__ == '__main__':
    unittest.main()
