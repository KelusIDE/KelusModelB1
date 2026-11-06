"""Scratch UI vision research: a patch transformer trained with our own autograd.

Predicts coarse region labels, not text, captions, click targets or actions.
Pillow decodes images; all neural-network operations and training use local code.
"""
from dataclasses import asdict, dataclass
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
import numpy as np
from .tensor import Tensor, cross_entropy, no_grad
from .optim import AdamW

LABELS = ('background', 'toolbar', 'sidebar', 'editor', 'terminal', 'dialog')


@dataclass(frozen=True)
class VisionConfig:
    image_size: int = 96
    patch_size: int = 12
    width: int = 48
    heads: int = 4
    layers: int = 2
    seed: int = 2026

    def validate(self):
        if not (24 <= self.image_size <= 256 and 4 <= self.patch_size <= 32
                and self.image_size % self.patch_size == 0
                and (self.image_size // self.patch_size) ** 2 <= 256
                and 8 <= self.width <= 256 and 1 <= self.heads <= self.width
                and self.width % self.heads == 0 and 1 <= self.layers <= 6):
            raise ValueError('Invalid vision configuration')


class VisionModel:
    def __init__(self, config=VisionConfig()):
        config.validate()
        self.config, self.parameters = config, {}
        rng = np.random.default_rng(config.seed)
        def weight(name, shape, kind='random'):
            data = (np.ones(shape) if kind == 'ones' else np.zeros(shape) if kind == 'zeros'
                    else rng.normal(0, 0.02, shape)).astype(np.float32)
            self.parameters[name] = Tensor(data, requires_grad=True)
        d, patches = config.width, (config.image_size // config.patch_size) ** 2
        weight('patch', (config.patch_size ** 2 * 3, d))
        weight('position', (patches, d))
        for layer in range(config.layers):
            prefix = f'block{layer}.'
            for norm in ['norm1', 'norm2']:
                weight(prefix + norm + '.gain', (d,), 'ones')
                weight(prefix + norm + '.bias', (d,), 'zeros')
            for name, shape in [('q',(d,d)), ('k',(d,d)), ('v',(d,d)), ('projection',(d,d)), ('up',(d,4*d)), ('down',(4*d,d))]:
                weight(prefix + name, shape)
        weight('final.gain', (d,), 'ones')
        weight('final.bias', (d,), 'zeros')
        weight('regions', (d, len(LABELS)))

    @property
    def parameter_count(self):
        return sum(p.data.size for p in self.parameters.values())

    def norm(self, x, name):
        centered = x - x.mean(axis=-1, keepdims=True)
        value = centered * ((centered * centered).mean(axis=-1, keepdims=True) + 1e-5) ** -0.5
        return value * self.parameters[name + '.gain'] + self.parameters[name + '.bias']

    def __call__(self, images):
        c = self.config
        images = np.asarray(images, dtype=np.float32)
        if images.ndim != 4 or images.shape[1:] != (c.image_size, c.image_size, 3) or not np.isfinite(images).all() or np.min(images) < 0 or np.max(images) > 1:
            raise ValueError('Expected finite RGB images [batch, size, size, 3] in [0, 1]')
        batch, p, g, d = images.shape[0], c.patch_size, c.image_size // c.patch_size, c.width

        patches = (images * 2 - 1).reshape(batch,g,p,g,p,3).transpose(0,1,3,2,4,5).reshape(batch,g*g,p*p*3)
        x = Tensor(patches) @ self.parameters['patch'] + self.parameters['position']
        h, head_width = c.heads, d // c.heads
        for layer in range(c.layers):
            prefix = f'block{layer}.'
            normalized = self.norm(x, prefix + 'norm1')
            def project(name):
                return (normalized @ self.parameters[prefix + name]).reshape(batch,g*g,h,head_width).transpose(0,2,1,3)
            q, k, v = project('q'), project('k'), project('v')

            attention = ((q @ k.transpose(0,1,3,2)) * head_width ** -0.5).softmax()
            mixed = (attention @ v).transpose(0,2,1,3).reshape(batch,g*g,d)
            x = x + mixed @ self.parameters[prefix + 'projection']
            hidden = self.norm(x, prefix + 'norm2') @ self.parameters[prefix + 'up']
            hidden = 0.5 * hidden * (1 + (math.sqrt(2/math.pi) * (hidden + 0.044715 * hidden**3)).tanh())
            x = x + hidden @ self.parameters[prefix + 'down']
        return self.norm(x, 'final') @ self.parameters['regions']

    def save(self, destination, metadata):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        header = {'format':'kani-ui-vision-v1', 'config':asdict(self.config), 'labels':LABELS, 'metadata':metadata}
        values = {name:p.data for name,p in self.parameters.items()}
        values['header'] = np.array(json.dumps(header))
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.npz', delete=False) as stream:
                temporary = stream.name
                np.savez_compressed(stream, **values)
                stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary and os.path.exists(temporary): os.unlink(temporary)

    @classmethod
    def load(cls, source):
        with np.load(source, allow_pickle=False) as archive:
            header = json.loads(str(archive['header']))
            if header.get('format') != 'kani-ui-vision-v1' or header.get('labels') != list(LABELS):
                raise ValueError('Not a compatible Kani UI vision checkpoint')
            model = cls(VisionConfig(**header['config']))
            for name, p in model.parameters.items():
                value = archive[name]
                if value.shape != p.data.shape or value.dtype.kind != 'f' or not np.isfinite(value).all():
                    raise ValueError('Invalid vision weights: ' + name)
                p.data[:] = value
        return model, header['metadata']


def synthetic(config, count, seed):
    """Original procedural UI layouts; train/test use different seeds, no real screenshots."""
    rng = np.random.default_rng(seed)
    size, p = config.image_size, config.patch_size
    g = size // p
    images = np.empty((count,size,size,3), dtype=np.float32)
    labels = np.zeros((count,g,g), dtype=np.int64)
    for index in range(count):
        grid = labels[index]
        top = int(rng.integers(1, max(2,g//4)))
        left = int(rng.integers(1, max(2,g//3)))
        bottom = int(rng.integers(1, max(2,g//3)))
        grid[:top] = 1
        grid[top:, :left] = 2
        grid[top:g-bottom, left:] = 3
        if rng.random() < .8: grid[g-bottom:,left:] = 4
        if rng.random() < .45:
            y, x = int(rng.integers(top,max(top+1,g-2))), int(rng.integers(left,max(left+1,g-2)))
            grid[y:min(y+2,g), x:min(x+3,g)] = 5
        dark = rng.random() < .6
        base = float(rng.uniform(.04,.2) if dark else rng.uniform(.7,.87))

        palette = np.clip(base + rng.uniform(-.06,.10,(len(LABELS),3)),0,1)
        image = images[index]
        for y in range(g):
            for x in range(g):
                label = grid[y,x]
                tile = image[y*p:(y+1)*p,x*p:(x+1)*p]
                tile[:] = palette[label]
                ink = np.clip(palette[label] + (.5 if dark else -.5),0,1)
                if label in (2,3,4):
                    for row in range(2,p-1,3):
                        indent = 2 if label == 2 else int(rng.integers(1,max(2,p//3)))
                        length = int(rng.integers(max(indent+1,p//2),p))
                        color = ink.copy()
                        if label == 3: color *= rng.uniform(.5,1,3)
                        if label == 4: color *= np.array([.5,1,.6])
                        tile[row:row+1,indent:length] = color
                elif label == 1: tile[p//2-1:p//2+1,2:4] = ink
                elif label == 5:
                    tile[1:2,1:-1] = ink
                    tile[-4:-2,p//2:-1] = ink
        image += rng.normal(0,.01,image.shape).astype(np.float32)
        np.clip(image,0,1,out=image)
    return images, labels.reshape(count,g*g)


def pillow():
    try:
        from PIL import Image
        return Image
    except ImportError as error:
        raise ValueError('Image decoding needs Pillow: python3 -m pip install -r requirements-vision.txt') from error


def load_image(filename, config):
    Image = pillow()
    with Image.open(filename) as image:
        if image.width * image.height > 40_000_000: raise ValueError('Image exceeds 40 megapixels')
        original = image.size

        ratio = min(config.image_size / image.width, config.image_size / image.height)
        size = tuple(max(1,round(value*ratio)) for value in image.size)
        resized = image.convert('RGB').resize(size,Image.Resampling.BILINEAR)
        canvas = Image.new('RGB',(config.image_size,config.image_size),(0,0,0))
        offset = ((config.image_size-size[0])//2,(config.image_size-size[1])//2)
        canvas.paste(resized,offset)
    return np.asarray(canvas,dtype=np.float32)/255, {'original_size':list(original),'resized_size':list(size),'offset':list(offset)}


def annotated_dataset(manifest, config):
    """JSONL: image, mask (PNG IDs 0..5), split=train|validation. Require disjoint images."""
    Image = pillow()
    root = Path(manifest).resolve().parent
    groups = {'train':[], 'validation':[]}; seen = {}
    for line in Path(manifest).read_text().splitlines():
        if not line.strip(): continue
        record = json.loads(line); split = record['split']
        if split not in groups: raise ValueError('Every image needs a train or validation split')
        filename = root / record['image']
        digest = hashlib.sha256(filename.read_bytes()).hexdigest()
        if digest in seen: raise ValueError('Duplicate screenshot or train/validation leakage')
        seen[digest] = split
        image, geometry = load_image(filename,config)
        with Image.open(root / record['mask']) as mask:
            values = np.asarray(mask)
            if values.ndim != 2 or values.dtype.kind not in 'ui' or tuple(mask.size) != tuple(geometry['original_size']) or np.any(values < 0) or np.any(values >= len(LABELS)):
                raise ValueError('Masks must match image size and contain class IDs 0..5')
            resized = Image.fromarray(values.astype(np.uint8)).resize(tuple(geometry['resized_size']),Image.Resampling.NEAREST)
            canvas = Image.new('L',(config.image_size,config.image_size),0)
            canvas.paste(resized,tuple(geometry['offset']))
            pixels = np.asarray(canvas)
        p, g = config.patch_size, config.image_size//config.patch_size
        patches = pixels.reshape(g,p,g,p).transpose(0,2,1,3).reshape(g*g,p*p)
        target = np.array([np.bincount(row,minlength=len(LABELS)).argmax() for row in patches])
        groups[split].append((image,target))
    if any(len(rows)<2 for rows in groups.values()): raise ValueError('Need at least two distinct screenshots in each split')
    return {key:(np.stack([r[0] for r in rows]),np.stack([r[1] for r in rows])) for key,rows in groups.items()}


def evaluate(model, images, targets, batch=8):
    confusion = np.zeros((len(LABELS),len(LABELS)),dtype=np.int64); losses=[]
    with no_grad():
        for start in range(0,len(images),batch):
            logits = model(images[start:start+batch]); y = targets[start:start+batch]
            losses.append((float(cross_entropy(logits,y).data), y.size))
            np.add.at(confusion,(y.ravel(),logits.data.argmax(-1).ravel()),1)
    union = confusion.sum(0)+confusion.sum(1)-np.diag(confusion)
    iou = np.divide(np.diag(confusion),union,out=np.zeros(len(LABELS)),where=union>0)
    return {'loss':sum(v*n for v,n in losses)/sum(n for _,n in losses), 'patch_accuracy':float(np.trace(confusion)/confusion.sum()),
            'mean_iou':float(iou[union>0].mean()), 'per_class_iou':dict(zip(LABELS,iou.tolist())), 'confusion':confusion.tolist()}


def train(args):
    config = VisionConfig(seed=args.seed); model = VisionModel(config)
    if args.data:
        groups = annotated_dataset(args.data,config); (images,targets),(val_images,val_targets) = groups['train'],groups['validation']
        origin = {'kind':'annotated_screenshots','manifest_sha256':hashlib.sha256(Path(args.data).read_bytes()).hexdigest()}
    else:
        images,targets = synthetic(config,256,args.seed)
        val_images,val_targets = synthetic(config,64,args.seed+1_000_000)
        origin = {'kind':'synthetic_ui','train_seed':args.seed,'validation_seed':args.seed+1_000_000}
    optimizer = AdamW(model.parameters,learning_rate=args.lr)
    rng = np.random.default_rng(args.seed+1)
    start=time.monotonic(); best=float('inf'); history=[]
    baseline=evaluate(model,val_images,val_targets)
    print(json.dumps({'stage':'initial','parameters':model.parameter_count,'validation':baseline}),flush=True)
    for step in range(1,args.steps+1):
        indices=rng.integers(0,len(images),size=args.batch)
        optimizer.zero_grad(); loss=cross_entropy(model(images[indices]),targets[indices]); loss.backward(); optimizer.step()
        if step % args.log_every == 0 or step == args.steps:
            metrics=evaluate(model,val_images,val_targets)
            report={'step':step,'training_loss':float(loss.data),'validation':metrics,'elapsed_seconds':time.monotonic()-start}
            history.append(report); print(json.dumps(report),flush=True)
            if metrics['loss'] < best:
                best=metrics['loss']
                model.save(args.out,{'steps':step,'parameters':model.parameter_count,'data':origin,'validation':metrics,
                    'limitation':'Coarse region classifier. No OCR, captions, code understanding or computer control. Synthetic scores do not establish real screenshot performance.'})
    summary={'parameters':model.parameter_count,'steps':args.steps,'data':origin,'baseline':baseline,'history':history,'best_checkpoint':str(args.out)}
    Path(str(args.out)+'.report.json').write_text(json.dumps(summary,indent=2)+'\n')


def inspect(args):
    model,metadata=VisionModel.load(args.checkpoint)
    image,geometry=load_image(args.image,model.config)
    with no_grad(): probability=model(image[None]).softmax().data[0]
    g=model.config.image_size//model.config.patch_size
    ids=probability.argmax(-1).reshape(g,g)
    report={'model':'Kani UI vision prototype','training':metadata,'geometry':geometry,
        'patch_size':model.config.patch_size,'image_size':model.config.image_size,'labels':LABELS,
        'grid':ids.tolist(),'mean_max_probability':float(probability.max(-1).mean()),
        'counts':{label:int((ids==index).sum()) for index,label in enumerate(LABELS)},
        'limitation':'Predicted coarse layout only; probabilities are not calibrated. This does not read text or understand code. Synthetic-trained weights are unvalidated on real screenshots.'}
    rendered=json.dumps(report,indent=2)
    if args.out: Path(args.out).write_text(rendered+'\n')
    print(rendered)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('train'); p.add_argument('--data',type=Path); p.add_argument('--out',type=Path,default=Path('checkpoints/vision-ui.npz'))
    p.add_argument('--steps',type=int,default=400); p.add_argument('--batch',type=int,default=8); p.add_argument('--lr',type=float,default=.001); p.add_argument('--seed',type=int,default=2026); p.add_argument('--log-every',type=int,default=50)
    p=sub.add_parser('inspect'); p.add_argument('--image',type=Path,required=True); p.add_argument('--checkpoint',type=Path,default=Path('checkpoints/vision-ui.npz')); p.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.command=='train' and not (1<=args.steps<=1_000_000 and 1<=args.batch<=64 and 0<args.lr<=.1 and args.log_every>=1): parser.error('Invalid training settings')
    try: train(args) if args.command=='train' else inspect(args)
    except (ValueError, OSError, KeyError) as error: parser.exit(1,str(error)+'\n')


if __name__ == '__main__': main()
