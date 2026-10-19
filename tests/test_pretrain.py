import contextlib
import io
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from kelus.corpus import digest, identity, split_for
from kelus.model import Model
from kelus.optim import AdamW
from kelus.pretrain import Corpus, main
from kelus.tokenizer import ByteTokenizer

class PretrainingTests(unittest.TestCase):
    def fixture(self,root):
        data=root/'data';data.mkdir()
        (data/'tokenizer.json').write_text(json.dumps(ByteTokenizer().to_dict()))
        streams={'train':np.tile(np.arange(40,90,dtype='<u2'),20),'validation':np.tile(np.arange(90,110,dtype='<u2'),50)}
        for split,tokens in streams.items():(data/(split+'.bin')).write_bytes(tokens.tobytes())
        (data/'manifest.json').write_text(json.dumps({'format':'kelus-raw-corpus-v1','tokenizer_sha256':digest(data/'tokenizer.json'),
            'tokens':{k:len(v) for k,v in streams.items()},'files':{s:digest(data/(s+'.bin')) for s in streams}}))
        return data
    def test_shifted_labels_and_exact_document_identity_split(self):
        with tempfile.TemporaryDirectory() as root:
            data=self.fixture(Path(root));c=Corpus(data);x,y=c.batch('train',2,16,np.random.default_rng(5))
            np.testing.assert_array_equal(x[:,1:],y[:,:-1]);self.assertFalse(np.any(y<0))
            self.assertEqual(split_for(identity('a  b\nc')),split_for(identity('a b c')))
            (data/'train.bin').write_bytes(b'\0\0')
            with self.assertRaisesRegex(ValueError,'changed'):Corpus(data)
    def test_rejects_unverified_source_archive_before_extraction(self):
        from kelus.corpus import collect
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'downloads').mkdir()
            (root/'downloads/cpython.tar.gz').write_bytes(b'not the pinned upstream archive')
            with self.assertRaisesRegex(ValueError,'fingerprint mismatch'):collect(root)
            self.assertFalse((root/'documents.jsonl').exists())
            self.assertFalse((root/'documents.partial').exists())

    def test_resume_matches_uninterrupted_weights_and_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=self.fixture(root)
            base=['--data',str(data),'--steps','4','--context','16','--width','8','--heads','2','--layers','1','--batch','1','--accumulate','2','--validation-batches','1','--checkpoint-every','2']
            with contextlib.redirect_stdout(io.StringIO()):
                main(base+['--run',str(root/'full')])
                real_step=AdamW.step
                def stop_after_two(opt,*args,**kwargs):
                    result=real_step(opt,*args,**kwargs)
                    if opt.step_count==2:signal.raise_signal(signal.SIGTERM)
                    return result
                with patch.object(AdamW,'step',stop_after_two):main(base+['--run',str(root/'split')])
                self.assertEqual(json.loads((root/'split/status.json').read_text())['state'],'stopped')
                main(base+['--run',str(root/'split'),'--resume'])
            first,h1,o1=Model.load(root/'full/latest.npz');second,h2,o2=Model.load(root/'split/latest.npz')
            for name in first.parameters:np.testing.assert_array_equal(first.parameters[name].data,second.parameters[name].data)
            for name in o1:np.testing.assert_array_equal(o1[name],o2[name])
            self.assertEqual(h1['metadata']['rng_state'],h2['metadata']['rng_state'])
            self.assertEqual(h2['metadata']['tokens_seen'],128)
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):main(base+['--run',str(root/'split')])
            changed=base.copy();changed[changed.index('--steps')+1]='5'
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):main(changed+['--run',str(root/'split'),'--resume'])

if __name__=='__main__':unittest.main()
