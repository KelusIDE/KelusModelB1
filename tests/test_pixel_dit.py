import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

try:
    from kelus.pixel_dit import find_subject, resume_config, train, init, positions, forward, sample, integrate
except ImportError:
    find_subject = None

SUBJECTS = ['Apple', 'Car', 'Cat', 'Girl', 'Man', 'Palm tree', 'Tree', 'Woman']


@unittest.skipIf(find_subject is None, 'MLX is not installed')
class PixelSubjectTests(unittest.TestCase):
    def test_whole_words_and_plurals(self):
        self.assertEqual(find_subject('draw two apples', SUBJECTS), 'Apple')
        self.assertEqual(find_subject('Draw a CAR, please', SUBJECTS), 'Car')
        self.assertEqual(find_subject('draw a woman reading', SUBJECTS), 'Woman')

    def test_longest_subject_wins(self):
        self.assertEqual(find_subject('draw a palm tree on a beach', SUBJECTS), 'Palm tree')
        self.assertEqual(find_subject('draw an anime girl with cat ears', SUBJECTS), 'Girl')

    def test_no_subject_inside_other_words(self):
        self.assertIsNone(find_subject('draw a cartoon', SUBJECTS))
        self.assertIsNone(find_subject('draw a category chart', SUBJECTS))
        self.assertIsNone(find_subject('draw a sunset over the ocean', SUBJECTS))


@unittest.skipIf(find_subject is None, 'MLX is not installed')
class PixelResumeTests(unittest.TestCase):
    def test_rejects_overwrite_missing_and_mismatched_resume(self):
        config = {'width': 8, 'depth': 1, 'heads': 2, 'classes': 2,
                  'subjects': ['Cat', 'Dog'], 'size': 64, 'patch': 4}
        with TemporaryDirectory() as temp:
            run = Path(temp)
            with self.assertRaisesRegex(ValueError, 'requires both'):
                resume_config(run, config, True, 10)
            (run / 'latest.npz').touch()
            (run / 'config.json').write_text(json.dumps({**config, 'step': 2, 'adam_steps': 2}))
            with self.assertRaisesRegex(ValueError, 'already contains'):
                resume_config(run, config, False, 10)
            for changed in ({'width': 16}, {'subjects': ['Dog', 'Cat']}):
                with self.assertRaisesRegex(ValueError, 'Resume mismatch'):
                    resume_config(run, {**config, **changed}, True, 10)
            with self.assertRaisesRegex(ValueError, 'must exceed'):
                resume_config(run, config, True, 2)
            self.assertEqual(resume_config(run, config, True, 10)['step'], 2)

    def test_cpu_resume_matches_uninterrupted_weights_optimizer_and_rng(self):
        import mlx.core as mx
        import numpy as np
        previous_device = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            with TemporaryDirectory() as temp:
                root = Path(temp)
                data = root / 'data'
                data.mkdir()
                rng = np.random.default_rng(12)
                np.save(data / 'images.npy', rng.integers(0, 256, (4, 64, 64, 3), dtype=np.uint8))
                np.save(data / 'labels.npy', np.array([0, 1, 0, 1]))
                (data / 'manifest.json').write_text(json.dumps({'subjects': ['Cat', 'Dog']}))
                options = dict(data=str(data), width=8, depth=1, heads=2, seed=3, batch=1,
                               lr=0.001, ema=0.99, checkpoint_every=1, preview_every=100, preview_seed=0)
                full, resumed = root / 'full', root / 'resumed'
                train(SimpleNamespace(**options, run=str(full), resume=False, steps=3))
                train(SimpleNamespace(**options, run=str(resumed), resume=False, steps=2))
                before = json.loads((resumed / 'config.json').read_text())
                train(SimpleNamespace(**options, run=str(resumed), resume=True, steps=3))
                with np.load(full / 'latest.npz') as left, np.load(resumed / 'latest.npz') as right:
                    self.assertEqual(set(left.files), set(right.files))
                    for key in left.files:
                        np.testing.assert_allclose(left[key], right[key], atol=1e-6, rtol=1e-6, err_msg=key)
                after = json.loads((resumed / 'config.json').read_text())
                self.assertEqual(after['adam_steps'], 3)
                self.assertNotEqual(before['rng_state'], after['rng_state'])
                self.assertEqual(after['rng_state'], json.loads((full / 'config.json').read_text())['rng_state'])
                status = json.loads((resumed / 'status.json').read_text())
                self.assertEqual(status['state'], 'complete')
                self.assertIsInstance(status['train_loss'], float)
        finally:
            mx.set_default_device(previous_device)


@unittest.skipIf(find_subject is None, 'MLX is not installed')
class PixelSamplerTests(unittest.TestCase):
    def setUp(self):
        import mlx.core as mx
        self.previous_device = mx.default_device()
        mx.set_default_device(mx.cpu)

    def tearDown(self):
        import mlx.core as mx
        mx.set_default_device(self.previous_device)

    def test_heun_reduces_error_for_an_analytic_flow(self):
        import math
        import mlx.core as mx
        initial = mx.array([0.1])
        exact = 0.1 / math.e
        euler = float(integrate(lambda x, t: x, initial, 20, 'euler')[0])
        heun = float(integrate(lambda x, t: x, initial, 10, 'heun')[0])


        self.assertLess(abs(heun - exact), abs(euler - exact) / 10)

    def test_default_matches_the_original_sampler_and_is_repeatable(self):
        import mlx.core as mx
        import numpy as np
        p = init(2, width=8, depth=1, heads=2, seed=7)
        p['final.w'] = mx.random.normal(p['final.w'].shape, key=mx.random.key(8)) * 0.05
        p['final.ada.w'] = mx.random.normal(p['final.ada.w'].shape, key=mx.random.key(9)) * 0.05
        pos = positions(8)
        labels, steps, seed, guidance = [0, 1], 3, 17, 2.0
        x = mx.random.normal((2, 64, 64, 3), key=mx.random.key(seed))
        for i in range(steps, 0, -1):
            t = mx.full((2,), i / steps)
            both = forward(p, mx.concatenate([x, x]), mx.concatenate([t, t]), mx.array([0, 1, 2, 2]), 2, pos)
            x = x - (1 / steps) * (both[2:] + guidance * (both[:2] - both[2:]))
            mx.eval(x)
        expected = np.array(mx.clip(x, -1, 1))
        actual = np.array(sample(p, labels, 2, pos, steps, guidance, seed))
        np.testing.assert_array_equal(actual, expected)
        heun = np.array(sample(p, labels, 2, pos, steps, guidance, seed, solver='heun'))
        repeat = np.array(sample(p, labels, 2, pos, steps, guidance, seed, solver='heun'))
        np.testing.assert_array_equal(heun, repeat)
        self.assertTrue(np.isfinite(heun).all())

    def test_invalid_sampling_options_fail(self):
        import mlx.core as mx
        p = init(2, width=8, depth=1, heads=2)
        pos = positions(8)
        for options in ({'steps': 0}, {'steps': -3}, {'solver': 'unknown'}, {'guidance': float('nan')}, {'null': 0}):
            with self.assertRaises(ValueError):
                sample(p, [0], 2, pos, **options)
        with self.assertRaises(ValueError):
            sample(p, [2], 2, pos)


if __name__ == '__main__':
    unittest.main()
