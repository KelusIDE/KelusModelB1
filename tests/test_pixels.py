import tempfile
from pathlib import Path
import unittest
import numpy as np
from kelus.pixels import Config, PixelModel, captions
from kelus.optim import AdamW


class PixelTests(unittest.TestCase):
    def test_learning_and_prompt_conditioning(self):
        model = PixelModel(Config(size=8, width=32, buckets=64))
        images = np.stack([np.full((8, 8, 3), -0.8, np.float32), np.full((8, 8, 3), 0.8, np.float32)])
        prompts = ['dark', 'light']
        def evaluate():
            return float(model.loss(images, prompts, np.random.default_rng(9)).data)
        before = evaluate()
        optimizer = AdamW(model.parameters, learning_rate=0.003)
        rng = np.random.default_rng(7)
        for _ in range(200):
            optimizer.zero_grad()
            loss = model.loss(images, prompts, rng)
            loss.backward()
            optimizer.step()
        self.assertLess(evaluate(), before * 0.6)
        self.assertGreater(model.sample('light', seed=5).mean(), model.sample('dark', seed=5).mean())

    def test_checkpoint_and_deterministic_sampling(self):
        model = PixelModel(Config(size=8, width=16, buckets=32))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.npz'
            model.save(path, {'step': 0})
            loaded, metadata = PixelModel.load(path)
            self.assertEqual(metadata['step'], 0)
            np.testing.assert_array_equal(model.sample('cat', steps=3), loaded.sample('cat', steps=3))
        self.assertEqual(model.sample('cat', steps=3).shape, (8, 8, 3))
        self.assertFalse(np.array_equal(model.sample('cat', steps=3), model.sample('dog', steps=3)))

    def test_stable_tokenization(self):
        np.testing.assert_array_equal(captions(['Red cat'], 64), captions(['red CAT'], 64))
        self.assertTrue(np.isfinite(captions([''], 64)).all())
