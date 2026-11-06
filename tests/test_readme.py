"""README generation must describe only the released checkpoint and stay idempotent."""
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ElementTree
from kelus.readme import refresh, snapshot_foundation, pipeline

SHA = 'a' * 64
MARKERS = ''.join(f'<!-- readme:{name} -->\nold\n<!-- /readme:{name} -->\n\n' for name in ['hero', 'badges', 'foundation', 'results'])


class CurrentReadmeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.write('kani-builds.json', [
            {'id': 'latest', 'export': 'site/models/chat'},
            {'id': 'cide', 'export': 'site/models/code'},
            {'id': 'seal-3', 'export': 'site/model'},
        ])
        entries = {}
        for name, directory, size, context in [('Otter test', 'models/chat', 200000, 1024),
                                              ('Cide test', 'models/code', 300000, 4096),
                                              ('Seal test', 'model', 10000, 256)]:
            self.write(f'site/{directory}/b1.json', {'name': name, 'parameters': size,
                       'config': {'context': context}, 'weights_sha256': SHA})
            entries[name] = {'directory': directory, 'sha256': SHA, 'bytes': size}
        self.write('model-host/catalog.json', {'repository': 'KelusIDE/KelusModelB1', 'tag': 'test-release', 'models': entries})
        for folder, version in [('mobile', '0.5.10'), ('cli', '0.18.0'), ('vscode-extension', '0.17.0')]:
            self.write(f'{folder}/package.json', {'version': version})
        self.write('api/models.json', [{'id': 'older-api-model'}])
        icon = self.root / 'brand/kani-otter-4/kani-otter-4.svg'
        icon.parent.mkdir(parents=True)
        icon.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><circle cx="256" cy="256" r="100"/></svg>')
        self.template = '# Kani\n\n' + ''.join(f'<!-- readme:{name} -->\nold\n<!-- /readme:{name} -->\n\n'
                                                 for name in ['hero', 'badges', 'releases'])
        (self.root / 'README.md').write_text(self.template)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_release_metadata_updates_without_weights_checkpoints_or_reports(self):
        refresh(self.root)
        readme = (self.root / 'README.md').read_text()
        self.assertIn('Otter test', readme)
        self.assertIn('200,000 | 1,024 tokens', readme)
        self.assertIn('300,000 | 4,096 tokens', readme)
        self.assertIn('0.5.10', readme)
        self.assertIn('older-api-model', readme)
        self.assertNotIn('158/199', readme)
        self.assertEqual(refresh(self.root), [])
        for path in (self.root / 'assets/readme').glob('*.svg'):
            ElementTree.fromstring(path.read_text())
        self.write('mobile/package.json', {'version': '0.6.0'})
        self.assertEqual(refresh(self.root), ['README.md'])
        self.assertIn('0.6.0', (self.root / 'README.md').read_text())

    def test_unpublished_manifest_is_rejected_before_writing(self):
        path = self.root / 'site/models/chat/b1.json'
        value = json.loads(path.read_text())
        value['weights_sha256'] = 'b' * 64
        self.write('site/models/chat/b1.json', value)
        with self.assertRaisesRegex(ValueError, 'checksums differ'):
            refresh(self.root)
        self.assertEqual((self.root / 'README.md').read_text(), self.template)
        self.assertFalse((self.root / 'assets').exists())

    def test_check_mode_does_not_write_release_content(self):
        self.assertIn('README.md', refresh(self.root, write=False))
        self.assertEqual((self.root / 'README.md').read_text(), self.template)
        self.assertFalse((self.root / 'assets').exists())


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
        self.assertNotIn('Second greeting', hero)
        self.assertNotIn('wrong', hero)

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

    def test_completed_pretraining_does_not_claim_instruction_training_is_running(self):
        chart = pipeline({'state': 'complete', 'step': 4000, 'total_steps': 4000,
                          'release_state': 'waiting_instruction_candidate'})
        self.assertIn('F["Foundation pretraining<br/>✓ done"]:::done', chart)
        self.assertIn('I["Instruction training<br/>next · awaiting completed candidate"]:::now', chart)
        self.assertNotIn('in progress', chart)

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
        self.assertEqual(snapshot['step'], 50)
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
