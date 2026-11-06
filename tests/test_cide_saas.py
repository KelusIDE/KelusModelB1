import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from data.make_cide_saas import accepted, saas_tasks
from kelus.cide_saas_eval import database, functions, verify
from kelus.corpus import digest
from kelus.model import Config, Model
from kelus.tokenizer import ByteTokenizer


class CideSaasTests(unittest.TestCase):
    def test_reference_crud_contracts_and_owner_isolation(self):
        mapping = {'saas_owned_read':('list_task0','list_entries'), 'saas_owned_create':('create_task0','create_entry'),
                   'saas_owned_update':('update_task0','edit_entry'), 'saas_owned_delete':('delete_task0','remove_entry'),
                   'saas_dashboard':('summary_task0','entry_count')}
        checked = set()
        for row in saas_tasks():
            family = row['family']
            if family not in mapping or family in checked: continue
            old, new = mapping[family]
            code = row['messages'][-1]['content'].replace(old,new).replace('tasks','entries')
            self.assertTrue(verify(functions(code,[new]),new)); checked.add(family)
        self.assertEqual(len(checked),5)

    def test_evaluator_blocks_filesystem_sql_and_python_escape(self):
        db = database()
        try:
            for sql in ("ATTACH DATABASE '/tmp/cide-denied' AS extra", "SELECT load_extension('bad')", "SELECT randomblob(100000000)"):
                with self.assertRaises(sqlite3.DatabaseError): db.execute(sql)
        finally: db.close()
        for code in ("def x():\n return __import__('os')", "def x():\n return ().__class__", "def x():\n while True: pass"):
            with self.assertRaises(ValueError): functions(code,['x'])

    def test_source_filters_exclude_secrets_and_generated_files(self):
        for path in ('.env', '.github/workflows/build.yml', 'frontend/src/client/generated.ts', 'dist/index.js', 'a.min.js'):
            self.assertFalse(accepted(path))
        for path in ('backend/app/api/routes/items.py', 'lib/auth/session.ts', 'app/dashboard/page.tsx'):
            self.assertTrue(accepted(path))

    def test_real_pipeline_preserves_parent_and_completes_both_phases(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); data = root/'data'; data.mkdir(); run = root/'run'
            tokenizer = ByteTokenizer(); parent = root/'parent.npz'
            Model(Config(context=1024,width=8,heads=2,layers=1),tokenizer).save(parent,{'steps':750})
            original = digest(parent)
            (data/'tokenizer.json').write_text(json.dumps(tokenizer.to_dict()))
            for split, start in [('train',32),('validation',64)]:
                (data/(split+'.bin')).write_bytes(np.tile(np.arange(start,start+32,dtype='<u2'),40).tobytes())
            manifest = dict(format='kelus-raw-corpus-v1', tokenizer_sha256=digest(data/'tokenizer.json'),
                            files={s:digest(data/(s+'.bin')) for s in ('train','validation')}, tokens={'train':1280,'validation':1280})
            (data/'manifest.json').write_text(json.dumps(manifest))
            records = [dict(split='validation' if i==3 else 'train',messages=[dict(role='user',content=f'Add {i} and 1'),dict(role='assistant',content=str(i+1))]) for i in range(4)]
            (data/'instructions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            command = [sys.executable,'-m','kelus.cide_saas_train','--run',str(run),'--data',str(data),'--parent',str(parent),
                       '--code-steps','2','--instruction-steps','2']
            completed = subprocess.run(command,capture_output=True,text=True,timeout=60)
            self.assertEqual(completed.returncode,0,completed.stderr+completed.stdout)
            self.assertEqual(digest(parent),original)
            self.assertEqual(json.loads((run/'pipeline.json').read_text())['completed_steps'],4)
            self.assertEqual(json.loads((run/'code/status.json').read_text())['step'],2)
            _,header,_ = Model.load(run/'instruction.npz')
            self.assertEqual(header['metadata']['steps'],2)
            self.assertTrue((run/'saas-eval.json').exists())
            self.assertTrue((run/'cide-saas-v2-candidate.npz').exists())
            duplicate = subprocess.run(command,capture_output=True,text=True,timeout=10)
            self.assertNotEqual(duplicate.returncode,0)
            self.assertIn('Run exists',duplicate.stderr)


if __name__ == '__main__': unittest.main()
