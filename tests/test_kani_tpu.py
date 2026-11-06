"""kaggle/kani_tpu.py: the JAX model and optimizer match kani_advanced.py's PyTorch ones, the feeder mixes, filters and
restarts its text exactly, and a run moves GPU -> TPU -> GPU through one checkpoint (JAX on the CPU, two fake chips).

Needs PyTorch, JAX and pyarrow (the project's own environment does not have them, so this skips):
    python -m unittest tests.test_kani_tpu        # with an environment that has torch, jax and pyarrow
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

from kelus.tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent.parent
KAGGLE = ROOT / 'kaggle'
HAVE = all(importlib.util.find_spec(name) for name in ('torch', 'jax', 'pyarrow'))
ENGLISH = ['The moon goes around the earth once a month.', 'Plants make food from sunlight, water and air.',
           'A river flows from the mountains to the sea.', 'Bees carry pollen from flower to flower.'] * 30
CODE = ['def add(a, b):\n    return a + b\n', 'for item in items:\n    print(item)\n'] * 30
LEAKED = 'The old lighthouse keeper climbed the winding stairs every night to light the lamp for the ships'
MARKER = 'Zebracorn'


def write_parquet(path, column, texts):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({column: texts}), path, row_group_size=50)


@unittest.skipUnless(HAVE, 'needs torch, jax and pyarrow')
class KaniTpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(KAGGLE))
        import kani_advanced
        import kani_tpu
        cls.ka, cls.kt = kani_advanced, kani_tpu

    def test_jax_model_and_optimizer_match_pytorch(self):
        """In float64 on both sides, so any difference is a real one, not rounding (Adam magnifies the rounding of tiny
        gradients to ~1e-5 in float32)."""
        import jax
        import torch
        import torch.nn.functional as F
        ka, kt = self.ka, self.kt
        jax.config.update('jax_enable_x64', True)
        self.addCleanup(jax.config.update, 'jax_enable_x64', False)
        settings = {**ka.DEFAULTS, **kt.TPU_DEFAULTS, 'width': 32, 'layers': 2, 'heads': 4, 'context': 16}
        torch.manual_seed(0)
        net = ka.build(settings, 300).double()
        opt = kt.optimizer(net, settings)
        state = kt.to_jax_state(net, opt)
        ids = np.random.default_rng(0).integers(0, 300, size=(2, 4, 17))  # two micro steps of four sequences
        forward, train_step, eval_step = kt.build_steps(settings, dtype='float64')
        with torch.no_grad():
            expected = net(torch.from_numpy(ids[0, :, :-1])).numpy()
        np.testing.assert_allclose(np.asarray(jax.jit(forward)(state['params'], ids[0, :, :-1])), expected, rtol=1e-10, atol=1e-12)
        step = jax.jit(train_step)
        for lr in (1e-2, 3e-3, 1e-2):  # kani_advanced.py's step: AdamW after clipping the gradient norm to 1
            for group in opt.param_groups:
                group['lr'] = lr
            for micro in range(2):
                logits = net(torch.from_numpy(ids[micro, :, :-1]))
                (F.cross_entropy(logits.flatten(0, 1), torch.from_numpy(ids[micro, :, 1:]).flatten()) / 2).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            state, loss, _ = step(state, ids[..., :-1], ids[..., 1:], np.float64(lr))
        params, m, v = (kt.unstacked(state[key]) for key in ('params', 'm', 'v'))
        for name, p in net.named_parameters():
            np.testing.assert_allclose(np.asarray(params[name]), p.detach().numpy(), rtol=1e-8, atol=1e-10, err_msg=name)
            np.testing.assert_allclose(np.asarray(m[name]), opt.state[p]['exp_avg'].numpy(), rtol=1e-8, atol=1e-12, err_msg=name)
            np.testing.assert_allclose(np.asarray(v[name]), opt.state[p]['exp_avg_sq'].numpy(), rtol=1e-8, atol=1e-14, err_msg=name)
        self.assertEqual(float(state['count']), float(opt.state[next(net.parameters())]['step']))
        # The validation loss is each sequence's mean loss, weighted (padding weighs nothing).
        with torch.no_grad():
            logits = net(torch.from_numpy(ids[1, :, :-1]))
            mean = F.cross_entropy(logits.flatten(0, 1), torch.from_numpy(ids[1, :, 1:]).flatten()).item()
        weighted = float(eval_step(state['params'], ids[1, :, :-1], ids[1, :, 1:], np.array([1, 1, 1, 0])))
        logits = net(torch.from_numpy(ids[1, :3, :-1]))
        self.assertAlmostEqual(weighted / 3, F.cross_entropy(logits.flatten(0, 1), torch.from_numpy(ids[1, :3, 1:]).flatten()).item(), places=9)
        # A JAX state written back into PyTorch (float32, as checkpoints are) is the same state.
        copy_net = ka.build(settings, 300)
        copy_opt = kt.optimizer(copy_net, settings)
        kt.from_jax_state(copy_net, copy_opt, jax.device_get(state))
        again = kt.to_jax_state(copy_net, copy_opt)
        for name in state['params']:
            np.testing.assert_array_equal(again['v'][name], np.asarray(state['v'][name], dtype=np.float32))
            np.testing.assert_array_equal(again['params'][name], np.asarray(state['params'][name], dtype=np.float32))
        self.assertEqual(float(again['count']), 3.0)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        files = self.root / 'files'
        for i in range(4):
            write_parquet(files / f'en/{i}.parquet', 'text', [f'File {i}. {text}' for text in ENGLISH] + [f'{MARKER} said: {LEAKED.lower()}.'])
            write_parquet(files / f'code/{i}.parquet', 'code', CODE)
        self.tokenizer = BPETokenizer.train(ENGLISH + CODE + [MARKER], 400, version=2)
        (self.root / 'tokenizer.json').write_text(json.dumps(self.tokenizer.to_dict()))
        tests = self.root / 'tests'
        tests.mkdir()
        rows = {'hellaswag_val.jsonl': {'activity_label': 'Cooking', 'ctx': 'A man cracks an egg.', 'endings': ['He fries it.']},
                'arc_easy_test.jsonl': {'question': 'What do plants need?', 'choices': {'text': ['Light']}},
                'arc_challenge_test.jsonl': {'question': 'Why is the sky blue?', 'choices': {'text': ['Scattering']}},
                'piqa_valid.jsonl': {'goal': 'Open a jar', 'sol1': 'Twist the lid', 'sol2': 'Eat the lid'},
                'winogrande_val.jsonl': {'sentence': 'Sam beat Ana so _ won.', 'option1': 'Sam', 'option2': 'Ana'},
                'lambada_test.jsonl': {'text': LEAKED}}
        for name, row in rows.items():
            (tests / name).write_text(json.dumps(row) + '\n')
        dataset = self.root / 'input/kani-start/kani-advanced'
        dataset.mkdir(parents=True)
        self.script('kani_advanced.py', 'grams', '--tests', str(tests), '--out', str(dataset / 'contamination.npy'))
        self.sources = [{'local': str(files / 'en/*.parquet'), 'per_session': 1, 'share': 0.75},
                        {'local': str(files / 'code/*.parquet'), 'column': 'code', 'per_session': 1, 'share': 0.25, 'offset': 1}]

    def tearDown(self):
        self.temp.cleanup()

    def settings(self, **extra):
        value = {'run_dir': str(self.root / 'run/kani-advanced'), 'input_root': str(self.root / 'input'), 'data_dir': str(self.root / 'data'),
                 'tokenizer': str(self.root / 'tokenizer.json'), 'sources': self.sources, 'decontaminate': True,
                 'width': 32, 'layers': 2, 'heads': 2, 'context': 32, 'micro_batch': 2, 'batch_tokens': 256,
                 'session_tokens': 6000, 'validation_documents': 40, 'hours': 0.0015, 'checkpoint_minutes': 0,
                 'eval_every': 4, 'eval_batches': 2, 'warmup_tokens': 2048, 'target_tokens': 10 ** 9, 'workers': 2,
                 'tpu_micro_batch': 2, 'shard_steps': 3, 'shards_ahead': 2, 'require_tpu': False, **extra}
        path = self.root / 'settings.json'
        path.write_text(json.dumps(value))
        return str(path)

    def script(self, name, *args):
        env = {**os.environ, 'JAX_PLATFORMS': 'cpu', 'XLA_FLAGS': '--xla_force_host_platform_device_count=2'}
        try:
            result = subprocess.run([sys.executable, str(KAGGLE / name), *args], cwd=self.root, env=env, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired as error:
            self.fail(f'{name} {args[0]} did not finish: {str(error.stdout)[-2000:]} {str(error.stderr)[-3000:]}')
        self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-3000:])
        return result.stdout

    def feed(self, first_shard, shards):
        """Run the feeder until it has written `shards` (it then waits for them to be read) and return their tokens."""
        data = self.root / 'feed'
        data.mkdir(exist_ok=True)
        for old in data.glob('shard-*'):
            old.unlink()
        (data / 'feed.json').write_text(json.dumps({'start': [2, 1], 'first_shard': first_shard,
                                                    'contamination': str(self.root / 'input/kani-start/kani-advanced/contamination.npy')}))
        settings = self.settings(data_dir=str(data), shards_ahead=len(shards))
        feeder = subprocess.Popen([sys.executable, str(KAGGLE / 'kani_tpu.py'), 'feed', '--settings', settings], cwd=self.root,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            deadline = time.time() + 120
            while not all((data / f'shard-{n:05d}.bin').exists() for n in shards):
                self.assertIsNone(feeder.poll(), feeder.stdout.read() if feeder.poll() is not None else '')
                self.assertLess(time.time(), deadline)
                time.sleep(0.2)
        finally:
            feeder.terminate()
            feeder.wait()
            feeder.stdout.close()
        return {n: np.fromfile(data / f'shard-{n:05d}.bin', dtype='<u2') for n in shards}, data

    def test_feeder_mixes_sources_drops_leaks_and_restarts_exactly(self):
        first, data = self.feed(0, [0, 1])
        size = 3 * 256 + 1
        self.assertEqual([len(tokens) for tokens in first.values()], [size, size])
        text = self.tokenizer.decode(first[0].tolist())
        self.assertIn('File 2. The moon', text)  # English from file number 2 on
        self.assertIn('def add(a, b)', text)     # and code (column 'code') in every shard
        self.assertNotIn(MARKER, text)
        english = int(size * 0.75 / 1.0) + (size - int(size * 0.75) - int(size * 0.25))
        self.assertIn('def add', self.tokenizer.decode(first[0][english:].tolist()))
        self.assertNotIn('def add', self.tokenizer.decode(first[0][:english - 30].tolist()))
        self.assertGreater(len(np.fromfile(data / 'validation.bin', dtype='<u2')), 100)
        again, _ = self.feed(1, [1])
        np.testing.assert_array_equal(again[1], first[1])
        self.assertFalse((data / 'shard-00000.bin').exists())

    def test_a_run_moves_from_the_gpu_to_the_tpu_and_back(self):
        import torch
        settings = self.settings()
        self.script('kani_advanced.py', 'prepare', '--settings', settings)
        self.assertIn('session done', self.script('kani_advanced.py', 'train', '--settings', settings))
        run = self.root / 'run/kani-advanced'
        gpu = json.loads((run / 'progress.json').read_text())
        self.assertEqual(gpu['next_file'], {self.sources[0]['local']: 1, self.sources[1]['local']: 2})

        # The TPU session: seven steps (it stops at the target), across three shards of three steps.
        settings = self.settings(target_tokens=gpu['tokens_seen'] + 7 * 256, hours=1.0)
        out = self.script('kani_tpu.py', 'train', '--settings', settings)
        self.assertIn('continuing from', out)
        self.assertIn('dropping documents that share 13 words', out)
        self.assertIn('target reached', out)
        tpu = json.loads((run / 'progress.json').read_text())
        self.assertEqual((tpu['session'], tpu['step']), (2, gpu['step'] + 7))
        self.assertIn('cpu', tpu['history'][-1]['machine'].lower())
        self.assertNotIn('feed', tpu)
        self.assertGreaterEqual(tpu['next_file'][self.sources[0]['local']], 2)
        state = torch.load(run / 'checkpoint.pt', weights_only=True)
        self.assertEqual(float(state['opt']['state'][0]['step']), tpu['step'])
        self.assertEqual(state['scaler']['scale'], 65536.0)
        self.assertEqual(sorted(json.loads((run / 'samples.json').read_text())), sorted(self.ka.PROMPTS))
        from kelus.model import Model
        model, header, _ = Model.load(run / 'kani-advanced.npz')
        self.assertEqual(header['metadata']['tokens_seen'], tpu['tokens_seen'])

        # Back on the GPU: it resumes from the TPU's checkpoint and reads the files after the TPU's.
        settings = self.settings(target_tokens=10 ** 9)
        self.script('kani_advanced.py', 'prepare', '--settings', settings)
        manifest = json.loads((self.root / 'data/manifest.json').read_text())
        start = tpu['next_file'][self.sources[0]['local']] % 3
        self.assertEqual(manifest['key']['plan'][0], [0, str(self.root / f'files/en/{start}.parquet')])
        out = self.script('kani_advanced.py', 'train', '--settings', settings)
        self.assertIn(f'resumed from {run}/checkpoint.pt: session 2, step {tpu["step"]:,}', out)


if __name__ == '__main__':
    unittest.main()
