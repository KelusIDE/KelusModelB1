import contextlib
import io
import json
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

try:
    import mlx.core as mx
    from kelus.gpu import AdamW, parameters, rope_tables, loss
    from kelus.gpu_train import pretrain
except ImportError:
    mx = None

from kelus.model import Config, Model
from tests.test_pretrain import PretrainingTests


@unittest.skipIf(mx is None, 'Requires MLX')
class CloudTrainingTests(unittest.TestCase):
    def setUp(self):
        self.device = mx.default_device()
        mx.set_default_device(mx.cpu)

    def tearDown(self):
        mx.set_default_device(self.device)

    def test_checkpointed_gradients_match_normal_gradients(self):
        model = Model(Config(context=16, width=8, heads=2, layers=2, positions='rope', tie_embeddings=True))
        params = parameters(model)
        cos, sin = rope_tables(model)
        x, y = mx.array([[1, 2, 3]]), mx.array([[2, 3, 4]])
        grad = mx.value_and_grad(loss)
        a, ga = grad(params, x, y, model.config, cos, sin, True, False)
        b, gb = grad(params, x, y, model.config, cos, sin, True, True)
        self.assertAlmostEqual(float(a), float(b), places=5)
        for name in ga:
            np.testing.assert_allclose(np.array(ga[name]), np.array(gb[name]), atol=2e-6, rtol=2e-5, err_msg=name)

    def test_interruption_resume_keeps_weights_rng_and_persistent_checkpoint(self):
        with TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            data = PretrainingTests().fixture(root)
            options = dict(data=str(data), steps=4, batch=1, accumulate=2, lr=0.001, seed=13,
                           weight_decay=0.01, context=16, width=8, layers=1, heads=2,
                           checkpoint_every=2, validation_batches=2, gradient_checkpointing=True,
                           max_seconds=0, init_from=None, overwrite=False)
            full, split, backup = root/'full', root/'split', root/'backup'
            pretrain(SimpleNamespace(**options, run=str(full), resume=False, backup_dir=None))
            update = AdamW.update
            def stop_after_two(opt, *args, **kwargs):
                result = update(opt, *args, **kwargs)
                if opt.step_count == 2:
                    signal.raise_signal(signal.SIGTERM)
                return result
            with patch.object(AdamW, 'update', stop_after_two):
                pretrain(SimpleNamespace(**options, run=str(split), resume=False, backup_dir=str(backup)))
            self.assertEqual(json.loads((split/'status.json').read_text())['state'], 'stopped')
            pretrain(SimpleNamespace(**options, run=str(split), resume=True, backup_dir=str(backup)))
            a, ha, oa = Model.load(full/'latest.npz')
            b, hb, ob = Model.load(split/'latest.npz')
            for name in a.parameters:
                np.testing.assert_allclose(a.parameters[name].data, b.parameters[name].data, atol=1e-7, rtol=1e-6)
            for name in oa:
                np.testing.assert_allclose(oa[name], ob[name], atol=1e-7, rtol=1e-6)
            self.assertEqual(ha['metadata']['rng_state'], hb['metadata']['rng_state'])
            self.assertEqual((backup/'latest.npz').read_bytes(), (split/'latest.npz').read_bytes())
            self.assertEqual(hb['metadata']['tokens_seen'], 128)
            self.assertEqual(json.loads((backup/'status.json').read_text())['state'], 'complete')
            with self.assertRaisesRegex(ValueError, 'recipe'):
                pretrain(SimpleNamespace(**{**options, 'steps': 5}, run=str(split), resume=True, backup_dir=None))


if __name__ == '__main__':
    unittest.main()
