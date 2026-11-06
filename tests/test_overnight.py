import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kelus.overnight import outcome, trainer_alive


class OvernightTests(unittest.TestCase):
    def test_pid_reuse_does_not_identify_an_unrelated_process_as_trainer(self):
        with patch('kelus.overnight.subprocess.run') as run:
            run.return_value.stdout='/bin/zsh\n'
            self.assertFalse(trainer_alive(123))
            run.return_value.stdout='python -m kelus.gpu_train instruct --out checkpoints/kani-big-v2-chat.npz\n'
            self.assertTrue(trainer_alive(123))

    def test_no_automatic_gain_is_reported_honestly(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            for name,score in [('baseline',2),('best',1),('final',2)]:
                (folder/f'{name}.json').write_text(json.dumps(dict(automatic_passed=score,automatic_total=9)))
            report=outcome(folder)
            self.assertEqual(report['automatic_gain'],0)
            self.assertTrue(report['manual_review_required'])
            self.assertFalse(report['deployed'])


if __name__=='__main__':
    unittest.main()
