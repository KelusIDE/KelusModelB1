"""Release gates must hold bad/stale candidates and publish a valid release once."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from kelus.auto_deploy import digest, gate, read, tick, tree_digest, write
from kelus.languages import scope_sha256, verification_template


class AutoDeployTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.state = Path(self.temp.name)
        self.release = self.state / 'releases' / 'test'
        r = self.release
        (r / 'site/model').mkdir(parents=True)
        (r / 'candidate.npz').write_bytes(b'fixture-checkpoint')
        self.sha = digest(r / 'candidate.npz')
        (r / 'site/model/b1.bin').write_bytes(b'fixture-weights')
        write(r / 'firebase.json', {'hosting': {'site': 'kelus-b1', 'public': 'site'}})
        write(r / 'readiness.json', {'sha256': self.sha, 'python_total': 4, 'python_passed': 4,
              'website': {'reply': '<html>fixture</html>', 'structure_checks': dict.fromkeys(['html_document','heading','button','inline_css','inline_javascript'], True)}})
        score = lambda p, t: {'passed': p, 'total': t}
        write(r / 'regressions.json', {'checkpoint_sha256': self.sha, 'scores': {'regression': score(5,5), 'heldout': score(12,15)}, 'screenshot_sequence': score(5,5)})
        write(r / 'benchmark.json', {'checkpoint_sha256': self.sha, 'overall': score(158,199), 'context_regression': score(179,184)})
        write(r / 'release.json', {'checkpoint_sha256': self.sha, 'stage': 'instruction', 'training_complete': True,
              'bundle_sha256': tree_digest(r / 'site'), 'config_sha256': digest(r / 'firebase.json'),
              'unit_tests_passed': True, 'browser_parity_passed': True, 'language_scope_sha256': scope_sha256(),
              'reports': {name: digest(r / name) for name in ['readiness.json','regressions.json','benchmark.json']}})
        write(r / 'website-verification.json', {'checkpoint_sha256': self.sha, 'bundle_sha256': tree_digest(r / 'site'),
              'reply_sha256': hashlib.sha256(b'<html>fixture</html>').hexdigest(), 'tester': 'test fixture', 'notes': 'simulated gate input, not real browser evidence',
              'checks': dict.fromkeys(['counter_starts_at_zero','counter_increments_twice','heading_and_button_visible','b1_preview_loads_and_replies'], True)})
        languages = verification_template(self.sha)
        for category in ('programming', 'chat'):
            for entry in languages[category].values():
                entry.update(reviewer='test fixture', evidence='Simulated unit-test evidence only')
                entry['checks'] = dict.fromkeys(entry['checks'], True)
        write(r / 'language-verification.json', languages)
        write(self.state / 'candidate.json', {'release': str(r)})
        self.foundation = self.state / 'foundation.json'
        write(self.foundation, {'state':'complete', 'step':4000, 'total_steps':4000})
        (self.state / 'enabled').touch()
        self.published = []
        self.verified = []

    def tick(self, publisher=None):
        tick(self.state, self.foundation, publisher or (lambda r,l: self.published.append(r)), lambda r:self.verified.append(r))
        return read(self.state / 'status.json')['state']

    def alter(self, filename, change, refresh=False):
        path = self.release / filename
        value = read(path); change(value); write(path, value)
        if refresh:
            self.alter('release.json', lambda m:m['reports'].update({filename:digest(path)}))

    def test_publish_valid_candidate_once_and_disable(self):
        self.assertIsNone(gate(self.release))
        self.assertEqual(self.tick(), 'deployed')
        self.assertEqual(self.tick(), 'deployed')
        self.assertEqual(len(self.published), 1)
        self.assertEqual(len(self.verified), 1)
        (self.state / 'enabled').unlink()
        self.assertEqual(self.tick(), 'disabled')
        self.assertEqual(len(self.published), 1)

    def test_pretraining_and_missing_instruction_candidate_never_publish(self):
        write(self.foundation, {'state':'running', 'step':1000, 'total_steps':4000})
        self.assertEqual(self.tick(), 'waiting_pretraining')
        write(self.foundation, {'state':'complete', 'step':4000, 'total_steps':4000})
        (self.state / 'candidate.json').unlink()
        self.assertEqual(self.tick(), 'waiting_instruction_candidate')
        self.assertFalse(self.published)

    def test_base_or_unfinished_candidate_cannot_publish(self):
        self.alter('release.json', lambda r:r.update(stage='pretraining'))
        self.assertEqual(self.tick(), 'held')
        self.alter('release.json', lambda r:r.update(stage='instruction', training_complete=False))
        self.assertEqual(self.tick(), 'held')
        self.assertFalse(self.published)

    def test_missing_real_browser_evidence_blocks(self):
        (self.release / 'website-verification.json').unlink()
        self.assertEqual(self.tick(), 'held')
        self.assertFalse(self.published)

    def test_failed_coding_and_regressions_block(self):
        self.alter('readiness.json', lambda r:r.update(python_passed=3), refresh=True)
        self.assertIn('coding', gate(self.release))
        self.alter('readiness.json', lambda r:r.update(python_passed=4), refresh=True)
        self.alter('benchmark.json', lambda r:r['overall'].update(passed=157), refresh=True)
        self.assertIn('regressions', gate(self.release))
        self.assertFalse(self.published)

    def test_stale_checkpoint_reports_evidence_or_bundle_block(self):
        self.alter('website-verification.json', lambda r:r.update(checkpoint_sha256='stale'))
        self.assertIn('another model', gate(self.release))
        self.alter('regressions.json', lambda r:r.update(checkpoint_sha256='stale'), refresh=True)
        self.assertIn('another checkpoint', gate(self.release))
        (self.release / 'site/model/b1.bin').write_bytes(b'changed')
        self.assertIn('bundle', gate(self.release))
        self.assertEqual(self.tick(), 'held')
        self.assertFalse(self.published)

    def test_interrupted_publish_is_not_retried(self):
        def fail(r, log):
            self.published.append(r)
            raise RuntimeError('uncertain network failure')
        self.assertEqual(self.tick(fail), 'needs_attention')
        self.assertEqual(self.tick(fail), 'deployment_needs_attention')
        self.assertEqual(len(self.published), 1)
        self.assertFalse(self.verified)

    def test_new_language_gate_holds_missing_failed_or_stale_results(self):
        evidence = self.release / 'language-verification.json'
        passing = read(evidence)
        evidence.unlink()
        self.assertEqual(self.tick(), 'held')
        write(evidence, passing)
        self.alter('language-verification.json', lambda r:r['programming']['cobol']['checks'].update(edge_cases=False))
        self.assertIn('cobol', gate(self.release))
        write(evidence, passing)
        self.alter('language-verification.json', lambda r:r.update(checkpoint_sha256='old'))
        self.assertIn('another checkpoint', gate(self.release))
        self.assertFalse(self.published)

    def test_malformed_reports_fail_closed(self):
        (self.release / 'release.json').write_text('{}')
        self.assertEqual(self.tick(), 'needs_attention')
        self.assertFalse(self.published)


if __name__ == '__main__':
    unittest.main()
