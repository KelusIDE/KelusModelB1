"""The kelus command's editor protocol and one-shot answers."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from kelus.cli import checkpoint_path
from kelus.model import Config, Model

ROOT = Path(__file__).resolve().parent.parent


class CliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.checkpoint = str(Path(cls.temp.name) / 'tiny.npz')
        Model(Config(context=64, width=8, heads=2, layers=1, seed=7)).save(cls.checkpoint, {'steps': 3})

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def kelus(self, *args, stdin=''):
        result = subprocess.run([sys.executable, '-m', 'kelus.cli', *args], input=stdin, capture_output=True, text=True, cwd=ROOT, timeout=60)
        return result

    def test_stdio_streams_validates_and_cancels(self):
        requests = [{'type': 'chat', 'id': 'a', 'messages': [{'role': 'user', 'content': 'hello!'}], 'max_tokens': 6},
                    {'type': 'chat', 'id': 'b', 'messages': [{'role': 'assistant', 'content': 'hi'}]},
                    {'type': 'cancel', 'id': 'c'},
                    {'type': 'chat', 'id': 'c', 'messages': [{'role': 'user', 'content': 'hello!'}]},
                    {'type': 'ping', 'id': 'd'}]
        result = self.kelus('stdio', '--checkpoint', self.checkpoint, stdin=''.join(json.dumps(r) + '\n' for r in requests) + 'not json\n')
        events = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(events[0], {'type': 'ready', 'name': 'Kani Seal 3', 'checkpoint': 'tiny', 'parameters': events[0]['parameters'], 'steps': 3,
                                     'codename': 'Kani B2:10F', 'config': events[0]['config'], 'tokenizer_spec': {'kind':'bytes'}})
        self.assertEqual(events[0]['config']['context'], 64)
        by_id = lambda rid: [e for e in events if e.get('id') == rid]
        self.assertEqual([by_id('a')[0]['type'], by_id('a')[-1]['type']], ['start', 'done'])
        self.assertEqual(by_id('b')[0]['type'], 'error')
        self.assertEqual(by_id('c')[-1], {**by_id('c')[-1], 'type': 'done', 'chunks': 0, 'cancelled': True})
        self.assertIn('Unknown request', by_id('d')[0]['message'])
        self.assertIn({'type': 'error', 'message': 'Invalid JSON request'}, events)  # reported as soon as it is read

    def test_stdio_reports_a_missing_checkpoint(self):
        result = self.kelus('stdio', '--model', 'no-such-model')
        self.assertEqual(result.returncode, 1)
        self.assertIn('No checkpoint named "no-such-model"', json.loads(result.stdout)['message'])

    def test_ask_labels_local_helpers_and_reads_stdin(self):
        result = self.kelus('ask', '--json', '--checkpoint', self.checkpoint, stdin='what is 6 * 7?')
        self.assertEqual(json.loads(result.stdout), {'source': 'local_calculator', 'text': '6 * 7 = 42'})
        self.assertIn('Nothing to ask', self.kelus('ask', '--checkpoint', self.checkpoint).stderr)

    def test_model_names_resolve_inside_checkpoints(self):
        with self.assertRaises(FileNotFoundError):
            checkpoint_path('no-such-model')
        self.assertEqual(checkpoint_path(checkpoint='x.npz'), Path('x.npz').resolve())


if __name__ == '__main__':
    unittest.main()
