import json
import tempfile
import unittest
from pathlib import Path

from kelus.standard_eval import GPT3, TASKS, documents, ladder


class StandardEvalTest(unittest.TestCase):
    def test_gpt2_small_sits_below_gpt3_small(self):
        # GPT-2 small (124M) as measured by kelus.standard_eval on 2026-10-31.
        results = {'hellaswag': {'acc_norm': 0.3119}, 'arc_easy': {'acc': 0.4381}, 'arc_challenge': {'acc_norm': 0.2244},
                   'piqa': {'acc': 0.6289}, 'winogrande': {'acc': 0.5162}, 'lambada': {'acc': 0.3256}}
        g = ladder(results)
        self.assertIsNone(g['reached'])
        self.assertEqual(g['next']['model'], 'GPT-3 Small (125M)')
        self.assertAlmostEqual(g['average'], 40.75, places=1)
        self.assertEqual([r['model'] for r in g['rungs']], list(GPT3))

    def test_rung_reached_by_average(self):
        results = {task: {'acc': 0.5, 'acc_norm': 0.5} for task in TASKS}
        self.assertEqual(ladder(results)['reached'], 'GPT-3 Medium (350M)')
        self.assertIsNone(ladder({'hellaswag': {'acc_norm': 0.9}}))

    def test_documents_follow_the_harness_prompts(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'winogrande_val.jsonl').write_text(json.dumps({'sentence': 'Sam beat Ana so _ won.', 'option1': 'Sam', 'option2': 'Ana', 'answer': '1'}) + '\n')
            Path(folder, 'lambada_test.jsonl').write_text(json.dumps({'text': 'the cat sat on the mat'}) + '\n')
            self.assertEqual(list(documents('winogrande', folder)), [([('Sam beat Ana so Sam', ' won.'), ('Sam beat Ana so Ana', ' won.')], 0)])
            self.assertEqual(list(documents('lambada', folder)), [([('the cat sat on the', ' mat')], 0)])


if __name__ == '__main__':
    unittest.main()
