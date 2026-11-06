import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kelus.local_chat_train import main


class LocalChatTrainTests(unittest.TestCase):
    def run_fixture(self, folder):
        parent, data, cases, run = [folder / name for name in ('parent.npz', 'data.jsonl', 'cases.json', 'run')]
        parent.write_bytes(b'preserved-parent')
        data.write_bytes(b'training-data')
        cases.write_text('[{"id":"writing","kind":"manual","prompt":"Write a paragraph."}]\n')
        run.mkdir()
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        (run / 'baseline.json').write_text(json.dumps(dict(results=[{}],
            cases_sha256=sha(cases), checkpoint_sha256=sha(parent))))
        argv = ['local_chat_train', '--run', str(run), '--parent', str(parent), '--data', str(data),
                '--cases', str(cases), '--steps', '2', '--probe-max-tokens', '256']
        return argv, run, cases

    def test_external_probe_bytes_and_budget_preserve_cached_baseline(self):
        with tempfile.TemporaryDirectory() as folder:
            argv, run, cases = self.run_fixture(Path(folder))
            with patch('sys.argv', argv), patch('kelus.local_chat_train.signal.signal'), \
                 patch('kelus.local_chat_train.shutil.disk_usage', return_value=SimpleNamespace(free=8 * 2**30)), \
                 patch('kelus.local_chat_train.subprocess.Popen') as popen:
                popen.return_value.wait.return_value = 0
                main()
            commands = [call.args[0] for call in popen.call_args_list]
            self.assertEqual(len(commands), 3)
            self.assertEqual((run / 'probes.json').read_bytes(), cases.read_bytes())
            for command in commands[1:]:
                self.assertEqual(command[command.index('--max-tokens') + 1], '256')
            self.assertEqual(json.loads((run / 'status.json').read_text())['state'], 'complete')

    def test_insufficient_disk_stops_before_training(self):
        with tempfile.TemporaryDirectory() as folder:
            argv, run, _ = self.run_fixture(Path(folder))
            with patch('sys.argv', argv), patch('kelus.local_chat_train.signal.signal'), \
                 patch('kelus.local_chat_train.shutil.disk_usage', return_value=SimpleNamespace(free=2**30)), \
                 patch('kelus.local_chat_train.subprocess.Popen') as popen:
                with self.assertRaisesRegex(OSError, '6.5 GiB'):
                    main()
            popen.assert_not_called()
            self.assertEqual(json.loads((run / 'status.json').read_text())['state'], 'failed')


if __name__ == '__main__':
    unittest.main()
