import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from kelus.vision import VisionConfig, VisionModel, synthetic, evaluate, annotated_dataset
from kelus.tensor import cross_entropy, no_grad
from kelus.optim import AdamW

class VisionTests(unittest.TestCase):
    def config(self): return VisionConfig(image_size=24,patch_size=6,width=8,heads=2,layers=1)

    def test_patch_attention_gradient_matches_finite_difference(self):
        model=VisionModel(self.config()); images,targets=synthetic(model.config,2,32)
        loss=cross_entropy(model(images),targets); loss.backward()
        for name,index in [('patch',(2,1)),('block0.q',(1,2)),('position',(3,2)),('regions',(2,3))]:
            weight=model.parameters[name]; analytic=float(weight.grad[index]); old=float(weight.data[index]); delta=.002
            with no_grad():
                weight.data[index]=old+delta; plus=float(cross_entropy(model(images),targets).data)
                weight.data[index]=old-delta; minus=float(cross_entropy(model(images),targets).data)
            weight.data[index]=old
            self.assertAlmostEqual(analytic,(plus-minus)/(2*delta),delta=.003,msg=name)

    def test_training_reduces_loss_and_checkpoint_roundtrips(self):
        model=VisionModel(self.config()); images,targets=synthetic(model.config,8,19)
        before=evaluate(model,images,targets)['loss']; optimizer=AdamW(model.parameters,learning_rate=.01)
        for _ in range(30):
            optimizer.zero_grad(); loss=cross_entropy(model(images),targets); loss.backward(); optimizer.step()
        self.assertLess(evaluate(model,images,targets)['loss'], before*.75)
        with tempfile.TemporaryDirectory() as root:
            dest=Path(root)/'vision.npz'; model.save(dest,{'steps':30}); loaded,meta=VisionModel.load(dest)
            with no_grad(): np.testing.assert_array_equal(model(images).data,loaded(images).data)
            self.assertEqual(meta['steps'],30)

    def test_split_generation_is_reproducible_and_distinct(self):
        a,labels=synthetic(self.config(),4,10); b,_=synthetic(self.config(),4,10); c,_=synthetic(self.config(),4,1_000_010)
        np.testing.assert_array_equal(a,b); self.assertFalse(np.array_equal(a,c))
        self.assertEqual(labels.shape,(4,16)); self.assertTrue(np.all((a>=0)&(a<=1)))

    @unittest.skipUnless(importlib.util.find_spec('PIL'), 'Pillow is optional for image loading')
    def test_annotated_masks_letterbox_and_duplicate_splits(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as root:
            root=Path(root); rows=[]
            for i in range(4):
                Image.new('RGB',(48,24),(30+i*20,90,130)).save(root/f'{i}.png')
                mask=Image.fromarray(np.full((24,48),3,dtype=np.uint8)).convert('P')
                mask.save(root/f'{i}-mask.png')
                rows.append({'image':f'{i}.png','mask':f'{i}-mask.png','split':'train' if i<2 else 'validation'})
            manifest=root/'manifest.jsonl'; manifest.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            groups=annotated_dataset(manifest,self.config())
            self.assertEqual(groups['train'][0].shape,(2,24,24,3))
            np.testing.assert_array_equal(groups['train'][1][0].reshape(4,4),[[0]*4,[3]*4,[3]*4,[0]*4])
            rows[-1]['image']='0.png'; manifest.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaisesRegex(ValueError,'leakage'): annotated_dataset(manifest,self.config())

    def test_bad_inputs_rejected(self):
        model=VisionModel(self.config())
        with self.assertRaises(ValueError): model(np.zeros((1,10,10,3)))
        with self.assertRaises(ValueError): model(np.full((1,24,24,3),np.nan))
        with self.assertRaises(ValueError): VisionModel(VisionConfig(width=13))

if __name__=='__main__': unittest.main()
