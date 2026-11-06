import json
from pathlib import Path
import tempfile
import unittest

from kelus.progress import summary


class ProgressTests(unittest.TestCase):
    def log(self, folder, tail):
        p = Path(folder) / 'instruction.log'
        p.write_text('{"name": "Kani"}\n' + '\n'.join(json.dumps(r) for r in [
            dict(step=1,elapsed_seconds=1,validation_loss=1.8),
            dict(step=450,elapsed_seconds=600,validation_loss=1.6955)]) + '\n' + tail)
        return p

    def test_user_interruption_uses_real_target(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.log(d, 'Traceback (most recent call last):\nKeyboardInterrupt\n')
            (Path(d)/'status.json').write_text(json.dumps(dict(state='stopped',total_steps=1500)))
            result = summary(p, {}, True)
            self.assertIn('450/1,500', result)
            self.assertIn('30.0%', result)
            self.assertIn('stopped', result)
            self.assertNotIn('failed', result)

    def test_real_failure_without_budget_does_not_show_complete(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.log(d, 'Traceback (most recent call last):\nOSError: No space left on device\n')
            result = summary(p, {})
            self.assertIn('failed', result)
            self.assertIn('target unknown', result)
            self.assertNotIn('100.0%', result)

    def test_another_instruction_job_does_not_mark_stopped_run_active(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.log(d, 'KeyboardInterrupt\n')
            (Path(d)/'recipe.json').write_text(json.dumps(dict(total_steps=1500)))
            result = summary(p, {'instruction.log':7500})
            self.assertIn('450/1,500', result)
            self.assertIn('stopped', result)


if __name__ == '__main__':
    unittest.main()
