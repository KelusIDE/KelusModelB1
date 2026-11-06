import ast
import unittest
from unittest.mock import patch

from data.make_cide_v8 import ENTITIES, question, specs
from kelus.crud_tasks import reference, signature, verify
from kelus.cide_candidate import gate, gpu_jobs


class CurriculumTests(unittest.TestCase):
    def test_every_operation_has_executable_references_across_splits(self):
        names = {}
        for split in ENTITIES:
            names[split] = set()
            for spec in specs(split, 3):
                code = reference(spec)
                ast.parse(code)
                self.assertTrue(verify(code, spec))
                names[split].add(spec['name'])
                for style in range(12):
                    prompt = question(spec, style)
                    self.assertIn(signature(spec), prompt)
                    for field in ('table', 'owner', 'column'):
                        self.assertIn(spec[field], prompt)
                    if spec['operation'] in ('create', 'update'):
                        self.assertIn(str(spec['limit']), prompt)
        self.assertFalse(names['train'] & names['validation'])
        self.assertFalse(names['train'] & names['test'])
        self.assertFalse(names['validation'] & names['test'])

    def test_wording_styles_are_distinct_and_repair_target_is_valid(self):
        spec = next(specs('train', 1))
        prompts = [question(spec, style) for style in range(12)]
        self.assertEqual(len(set(prompts)), 12)
        self.assertIn('def python ', prompts[6])
        self.assertNotIn('def python ', reference(spec))

    def test_release_review_requires_behavior_gains_and_no_regressions(self):
        reports = {'wording': {'passed': 10}, 'development': {'passed': 3},
                   'basic': {'passed': 8}, 'crud': {'passed': 23},
                   'websites': {'working_structure': '5/8'}, 'languages': {'right_language': '113/152'}}
        decision = gate(reports, {'passed': 0})
        self.assertEqual(decision['decision'], 'eligible_for_review')
        self.assertFalse(decision['automatically_deployed'])
        for suite, key, value in [('wording', 'passed', 9), ('development', 'passed', 2),
                                  ('basic', 'passed', 7), ('crud', 'passed', 22),
                                  ('websites', 'working_structure', '4/8'), ('languages', 'right_language', '112/152')]:
            changed = {**reports, suite: {key: value}}
            self.assertEqual(gate(changed, {'passed': 0})['decision'], 'withhold')
        self.assertEqual(gate(reports, {'passed': 11})['decision'], 'withhold')

    def test_queue_detects_training_without_matching_monitor_or_itself(self):
        listing = '12 python -m kelus.gpu_train instruct --steps 7000\n13 python -m kelus.progress file\n14 python -m kelus.cide_candidate\n'
        with patch('kelus.cide_candidate.subprocess.check_output', return_value=listing):
            self.assertEqual(gpu_jobs(), [12])


if __name__ == '__main__':
    unittest.main()
