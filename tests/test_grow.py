import tempfile
from pathlib import Path
import unittest

import numpy as np

from kelus.grow import expand, grow_checkpoint
from kelus.model import Config, Model
from kelus.tensor import no_grad, cross_entropy


class GrowthTests(unittest.TestCase):
    def test_predictions_preserved_and_new_layers_can_learn(self):
        model = Model(Config(context=16, width=8, heads=2, layers=2, positions='rope', tie_embeddings=True))
        grown = expand(model, 4, 32)
        ids = np.array([[2, 7, 11, 8]])
        with no_grad():
            np.testing.assert_array_equal(model(ids).data, grown(ids).data)
        np.testing.assert_allclose(model.next_logits(ids[0], {}), grown.next_logits(ids[0], {}), atol=1e-6)
        cross_entropy(grown(ids), np.array([[7, 11, 8, 9]])).backward()
        self.assertGreater(np.abs(grown.parameters['block1.down'].grad).sum(), 0)
        self.assertGreater(np.abs(grown.parameters['block1.projection'].grad).sum(), 0)
        self.assertGreater(grown.parameter_count, model.parameter_count)
        with self.assertRaises(ValueError):
            expand(model, 3, 32)

    def test_parent_remains_unchanged_and_export_is_loadable(self):
        with tempfile.TemporaryDirectory() as temp:
            source, out = Path(temp)/'source.npz', Path(temp)/'grown.npz'
            Model(Config(context=16, width=8, heads=2, layers=1, positions='rope', tie_embeddings=True)).save(source, {'steps': 123})
            before = source.read_bytes()
            report = grow_checkpoint(source, out, 2, 32)
            self.assertEqual(report['parent_steps'], 123)
            self.assertEqual(source.read_bytes(), before)
            loaded, _, states = Model.load(out)
            self.assertEqual(loaded.config.layers, 2)
            self.assertEqual(states, {})
            with self.assertRaises(ValueError):
                grow_checkpoint(source, out, 2, 32)


if __name__ == '__main__':
    unittest.main()
