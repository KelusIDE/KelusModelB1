import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from kelus.pixel_continue import gpu_jobs, prepare


class PixelContinuationTests(unittest.TestCase):
    def test_only_gpu_python_trainers_block_queue(self):
        listing = '''
100 /usr/bin/python3 -m kelus.gpu_train pretrain --steps 40000
101 caffeinate -i python3 -m kelus.gpu_train pretrain --steps 40000
102 /usr/bin/python3 -m kelus.train --steps 6000
103 /usr/bin/python3 -m kelus.pixel_continue
104 /usr/bin/python3 -m kelus.pixel_dit train --resume
105 /bin/zsh -c python3 -m kelus.pixel_dit train
106 /usr/bin/python3 -m kelus.sketch train
'''
        self.assertEqual([job['pid'] for job in gpu_jobs(listing)], [100, 104, 106])

    def test_prepare_preserves_source_and_refuses_different_plans(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source, run, data = [root / name for name in ('source', 'continued', 'data')]
            for folder in (source, run, data):
                folder.mkdir()
            python = root / 'python'
            python.touch()
            config = {'step': 40000, 'adam_steps': 40000, 'width': 320, 'depth': 10,
                      'heads': 5, 'subjects': ['Cat', 'Dog']}
            (source / 'config.json').write_text(json.dumps(config))
            (source / 'latest.npz').write_bytes(b'checkpoint-and-optimizer')
            (source / 'status.json').write_text(json.dumps({'state': 'complete'}))
            (data / 'manifest.json').write_text(json.dumps({'subjects': ['Cat', 'Dog']}))
            for name in ('images.npy', 'labels.npy'):
                (data / name).touch()
            plan = prepare(source, run, data, 80000, python)
            self.assertEqual(plan['additional_steps'], 40000)
            self.assertEqual((run / 'latest.npz').read_bytes(), (source / 'latest.npz').read_bytes())
            self.assertEqual(plan, prepare(source, run, data, 80000, python))
            (run / 'latest.npz').write_bytes(b'updated-weights')
            self.assertEqual((source / 'latest.npz').read_bytes(), b'checkpoint-and-optimizer')
            with self.assertRaisesRegex(ValueError, 'plan differs'):
                prepare(source, run, data, 90000, python)
            with self.assertRaisesRegex(ValueError, 'separate run'):
                prepare(source, source, data, 80000, python)
            (data / 'manifest.json').write_text(json.dumps({'subjects': ['Dog', 'Cat']}))
            with self.assertRaisesRegex(ValueError, 'subjects/order'):
                prepare(source, run, data, 80000, python)


if __name__ == '__main__':
    unittest.main()
