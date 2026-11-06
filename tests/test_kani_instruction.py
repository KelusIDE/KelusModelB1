"""Checkpoint compatibility, assistant masking, leakage rejection and resumable pilot smoke test."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from kelus.model import Config, Model
from kelus.tokenizer import conversation


@unittest.skipUnless(importlib.util.find_spec('torch'), 'needs isolated instruction-tools environment')
class InstructionTests(unittest.TestCase):
    def test_import_export_preserves_numpy_weights_and_logits(self):
        import torch
        from kaggle.kani_instruction import load_base, export
        from kelus.tensor import no_grad
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / 'base.npz', Path(directory) / 'export.npz'
            original = Model(Config(width=16, layers=2, heads=2, context=128, positions='rope', tie_embeddings=True))
            original.save(source)
            net, _, header = load_base(source)
            ids = np.array([[257, 259, 65, 260, 66]], dtype=np.int64)
            with torch.no_grad(), no_grad():
                np.testing.assert_allclose(net(torch.tensor(ids)).numpy(), original(ids).data, atol=1e-5, rtol=1e-4)
            export(net, header, target, {'stage': 'test'})
            restored, _, _ = Model.load(target)
            for name in original.parameters:
                np.testing.assert_array_equal(original.parameters[name].data, restored.parameters[name].data)

    def test_mask_leakage_train_and_resume(self):
        from kaggle.kani_instruction import load_base, records
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, data, run = root / 'base.npz', root / 'data.jsonl', root / 'run'
            model = Model(Config(width=16, layers=1, heads=2, context=128, positions='rope', tie_embeddings=True))
            model.save(source)
            rows = [dict(split=split, messages=[dict(role='user', content=q), dict(role='assistant', content=a)])
                    for split, q, a in [('train', 'Repeat blue.', 'blue'), ('train', 'Repeat red.', 'red'), ('validation', 'Repeat green.', 'green')]]
            data.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            _, tokenizer, _ = load_base(source)
            encoded = records(data, tokenizer, 128)
            _, mask = conversation(rows[0]['messages'], tokenizer=tokenizer)
            self.assertEqual((encoded['train'][0][1] != -100).tolist(), mask[1:])
            command = [sys.executable, '-m', 'kaggle.kani_instruction', '--checkpoint', str(source), '--data', str(data),
                       '--run', str(run), '--device', 'cpu', '--steps', '2', '--batch', '1', '--accumulate', '2',
                       '--save-every', '1', '--validation-examples', '1', '--max-seconds', '60']
            subprocess.run(command, check=True, capture_output=True, timeout=90)
            self.assertEqual(json.loads((run / 'status.json').read_text())['step'], 2)
            trained, _, _ = Model.load(run / 'latest.npz')
            self.assertFalse(np.array_equal(trained.parameters['token'].data, model.parameters['token'].data))
            subprocess.run(command + ['--resume'], check=True, capture_output=True, timeout=90)
            resumed, _, _ = Model.load(run / 'latest.npz')
            np.testing.assert_array_equal(trained.parameters['token'].data, resumed.parameters['token'].data)
            rows[-1]['messages'][0]['content'] = rows[0]['messages'][0]['content']
            data.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            with self.assertRaisesRegex(ValueError, 'leakage'):
                records(data, tokenizer, 128)


if __name__ == '__main__':
    unittest.main()
