"""README generation must describe only the released checkpoint and stay idempotent."""
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ElementTree
from kelus.readme import refresh, snapshot_foundation

SHA = 'a' * 64
MARKERS = ''.join(f'<!-- readme:{name} -->\nold\n<!-- /readme:{name} -->\n\n' for name in ['hero', 'badges', 'foundation', 'results'])


class ReadmeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        score = lambda p, t: {'passed': p, 'total': t}
        self.write('site/model/b1.json', {'parameters': 431424, 'source_checkpoint_sha256': SHA})
        self.write('reports/benchmark/b1.json', {
            'checkpoint': 'checkpoints/b1-test.best.npz', 'checkpoint_sha256': SHA,
            'overall': score(9, 10), 'context_regression': score(4, 5),
            'groups': {'chat': score(3, 4), 'concept': score(2, 2), 'code': score(2, 2), 'multi_turn': score(2, 2)},
            'results': [{'group': 'chat', 'intent': 'greeting', 'prompt': 'hi <b>', 'reply': 'Hello & "welcome"', 'passed': True},
                        {'group': 'chat', 'intent': 'greeting', 'prompt': 'hey', 'reply': 'Second greeting', 'passed': True},
                        {'group': 'chat', 'intent': 'thanks', 'prompt': 'thanks', 'reply': 'wrong', 'passed': False}]})
        self.write('reports/benchmark/b1-original.json', {'checkpoint_sha256': SHA, 'scores': {'regression': score(5, 5), 'heldout': score(12, 15)}})
        self.write('reports/code-readiness-test.json', {'sha256': SHA, 'python_passed': 0, 'python_total': 4,
                                                        'website': {'structure_checks': {'heading': True, 'button': False}}})
        (self.root / 'README.md').write_text('# B1\n\n' + MARKERS)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_refresh_fills_sections_and_is_idempotent(self):
        changed = refresh(self.root)
        self.assertIn('README.md', changed)
        self.assertIn('assets/readme/hero-dark.svg', changed)
        readme = (self.root / 'README.md').read_text()
        self.assertNotIn('\nold\n', readme)
        self.assertIn('**9/10**', readme)
        self.assertIn('| New tasks | Website structure checks | 1/2 | 50% |', readme)
        self.assertIn('No foundation snapshot yet', readme)
        self.assertEqual(refresh(self.root), [])

    def test_figures_are_valid_xml_with_escaped_replies(self):
        refresh(self.root)
        for path in (self.root / 'assets/readme').glob('*.svg'):
            ElementTree.fromstring(path.read_text())
        hero = (self.root / 'assets/readme/hero-light.svg').read_text()
        self.assertIn('Hello &amp; &quot;welcome&quot;', hero)
        self.assertNotIn('Second greeting', hero)  # one sample per intent
        self.assertNotIn('wrong', hero)  # failed probes are never replayed

    def test_reports_for_another_checkpoint_are_refused(self):
        self.write('site/model/b1.json', {'parameters': 1, 'source_checkpoint_sha256': 'b' * 64})
        with self.assertRaisesRegex(ValueError, 'released checkpoint'):
            refresh(self.root)

    def test_missing_markers_are_refused(self):
        (self.root / 'README.md').write_text('# B1\n')
        with self.assertRaisesRegex(ValueError, 'markers'):
            refresh(self.root)

    def test_check_mode_writes_nothing(self):
        self.assertIn('README.md', refresh(self.root, write=False))
        self.assertIn('\nold\n', (self.root / 'README.md').read_text())
        self.assertFalse((self.root / 'assets').exists())

    def test_foundation_snapshot_drops_local_details_and_partial_lines(self):
        run = self.root / 'training-runs/b1-foundation'
        run.mkdir(parents=True)
        row = lambda step, loss, t: {'stage': 'pretraining', 'state': 'running', 'pid': 42, 'step': step, 'total_steps': 100,
                                     'parameters': 8691840, 'context': 1024, 'tokens_seen': step * 4096, 'validation_loss': loss,
                                     'eta_seconds': 60, 'updated_at': t, 'checkpoint': '/Users/someone/latest.npz'}
        (run / 'status.json').write_text(json.dumps(row(40, 3.0, 10)))
        (run / 'metrics.jsonl').write_text('\n'.join([json.dumps(row(0, 8.0, 1)), json.dumps(row(50, 2.5, 20)), '{"step": 6']))
        self.assertTrue(snapshot_foundation(self.root))
        snapshot = json.loads((self.root / 'reports/foundation-progress.json').read_text())
        self.assertEqual(snapshot['step'], 50)  # newest row wins, even over status.json
        self.assertEqual(snapshot['validation_history'], [[0, 8.0], [50, 2.5]])
        self.assertNotIn('pid', snapshot)
        self.assertNotIn('/Users/', json.dumps(snapshot))
        self.assertFalse(snapshot_foundation(self.root))
        refresh(self.root)
        readme = (self.root / 'README.md').read_text()
        self.assertIn('step 50 of 100 (50%)', readme)
        self.assertIn('F["Foundation pretraining<br/>● in progress · step 50 / 100"]:::now', readme)


if __name__ == '__main__':
    unittest.main()
