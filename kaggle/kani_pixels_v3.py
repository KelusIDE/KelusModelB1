"""Kani Pixels v3: an image diffusion model conditioned on labels, trained from scratch on Kaggle GPUs.

No pretrained weights anywhere. EDM (Karras et al. 2022, "Elucidating the Design Space of
Diffusion-Based Generative Models") preconditioning, noise levels, loss weighting and Heun
sampler, with an ADM-style U-Net and classifier-free guidance. Trains on one or two GPUs,
stops itself before Kaggle's 12-hour limit, resumes from its own checkpoint, and exports the
EMA weights as NumPy arrays for Kani. A picture can carry several labels ("cat", "drawing"), so a
prompt can combine them. Open Images (settings 'open_images') gives thousands of subjects.

    python kani_pixels_v3.py prepare --settings settings.json
    torchrun --standalone --nproc_per_node=2 kani_pixels_v3.py train --settings settings.json
    python kani_pixels_v3.py sample --settings settings.json --prompt "a drawing of a cat" --out cats.png
    python kani_pixels_v3.py export --settings settings.json   # Kani files from checkpoint.pt
"""
import argparse
from collections import Counter
import copy
import csv
import io
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SIGMA_DATA, P_MEAN, P_STD = 0.5, -1.2, 1.2
SIGMA_MIN, SIGMA_MAX, RHO = 0.002, 80.0, 7.0
IMAGE_TYPES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
MULTS = {32: [1, 2, 2], 64: [1, 2, 2, 4], 128: [1, 1, 2, 3, 4]}

# Open Images V7: the annotations are CC BY 4.0 and the pictures are listed as CC BY 2.0. Only its 1.74M "boxable"
# training pictures are on the public S3 mirror; each has about six human-verified labels.
OPEN_IMAGES_LISTS = 'https://storage.googleapis.com/openimages/'
OPEN_IMAGES_NAMES = 'v7/oidv7-class-descriptions.csv'
OPEN_IMAGES_LABELS = 'v7/oidv7-train-annotations-human-imagelabels.csv'
OPEN_IMAGES_ROWS = '2018_04/train/train-images-boxable-with-rotation.csv'
OPEN_IMAGES_PICTURE = 'https://s3.amazonaws.com/open-images-dataset/train/{}.jpg'
USER_AGENT = {'User-Agent': 'kani-pixels-v3'}

# A picture with one of these labels is not used at all, so none of them can be asked for. Sexual and undressed content
# and gore: this model is public and people can ask it for anything it knows. Labels are not perfect, so this lowers the
# chance of such pictures; it does not prove there are none.
LEFT_OUT = {
    'abdomen', 'active undergarment', 'adult', 'art model', 'bathing', 'belly dance', 'blood', 'briefs', 'chest hair', 'child model', 'childbirth',
    'competitive swimwear', 'corpse', 'corset', 'fashion model', 'flesh', 'garter', 'gore', 'gunshot', 'hip', 'injury', 'kiss', 'love', 'maillot',
    'model', 'muscle', 'navel', 'pole dance', 'romance', 'showgirl', 'skin', 'stocking', 'thigh', 'tights', 'torso', 'trunks', 'waist', 'wound',
}
LEFT_OUT_WORDS = re.compile(r'nud(?:e|ity|ist)|naked|lingerie|underwear|undergarment|underpants|swimsuit|swimwear|swim brief|bikini|brassiere|\bbra\b|'
                            r'erotic|\bsex|porn|nipple|buttock|panties|thong|fetish|burlesque|stripper|lap dance|boudoir|pin-up|topless|barechested', re.I)
# Pictures of children in these settings are also left out, even with no label above.
CHILDREN = {'baby', 'baby & toddler clothing', 'boy', 'child', 'diaper', 'flower girl', 'girl', 'infant', 'infant bed', 'infant bodysuit', 'pageboy', 'toddler'}
NOT_WITH_CHILDREN = {
    'amateur wrestling', 'artistic gymnastics', 'bath toy', 'bathroom', 'bathtub', 'beach', 'bed', 'bedroom', 'cheerleading', 'cheerleading uniform', 'gymnastics',
    'human body', 'human leg', 'knee', 'leg', 'pool', 'professional wrestling', 'robe', 'scholastic wrestling', 'shower', 'sofa bed', 'surfing', 'swim cap',
    'swimmer', 'swimming', 'swimming pool', 'water park', 'wrestling', 'wrestling singlet',
}
# Words a prompt can leave out, and common words for labels with another name.
PROMPT_FILLER = {'a', 'an', 'the', 'of', 'on', 'in', 'at', 'by', 'to', 'with', 'and', 'or', 'some', 'my', 'me', 'please', 'draw', 'make', 'create', 'show', 'generate',
                 'give', 'paint', 'picture', 'image', 'photo', 'pic', 'small', 'big', 'cute', 'nice', 'very', 'really'}
PROMPT_WORDS = {'bike': 'bicycle', 'plane': 'airplane', 'aeroplane': 'airplane', 'kitten': 'cat', 'kitty': 'cat', 'kitties': 'cat', 'puppy': 'dog', 'puppies': 'dog', 'doggy': 'dog', 'people': 'person',
                'human': 'person', 'guy': 'man', 'lady': 'woman', 'kid': 'child', 'kids': 'child', 'men': 'man', 'women': 'woman', 'mice': 'mouse', 'sketched': 'sketch'}

DEFAULTS = {
    'input_root': '/kaggle/input',
    'dataset_root': None,  # where the DATASETS folders are, when not attached under input_root (the notebook downloads them)
    'datasets': {},
    'open_images': None,  # {'per_label', 'max_images', 'min_label_images'}: thousands of subjects from Open Images, see open_images_plan
    'label_slots': 8,  # most labels one picture keeps, rarest first
    'label_keep': 0.5,  # while training each of a picture's labels is kept this often, so that one or two words in a prompt work
    'prepare_hours': 3.0,  # stop downloading pictures after this long and train on what arrived
    'prepare_threads': 32,
    'session_started': None,  # time.time() at the start of the session: 'hours' then counts from there, downloads included
    'warm_start': True,  # an earlier checkpoint with other subjects: keep what it learned, start the new subjects fresh
    'showcase': [],  # prompts for the sample grids (else subjects spread over the list); 'drawing + cat' combines two labels
    'allow_missing': False,  # True: train even if some of the datasets have no images
    'class_caps': {},  # dataset folder name -> most images per subject from that dataset
    'exclude': [],  # folder names whose images are not used
    'size': 64,
    'max_per_class': 40000,
    'min_per_class': 200,
    'data_dir': '/tmp/kani-pixels-data',
    'run_dir': '/kaggle/working/kani-pixels-v3',
    'hours': 10.5,
    'steps': None,
    'batch': 128,
    'lr': 2e-4,
    'warmup_steps': 2000,
    'channels': 128,
    'mults': None,
    'blocks': 2,
    'attention': [16, 8],
    'dropout': 0.1,
    'label_dropout': 0.1,
    'ema_halflife_kimg': 500,
    'ema_rampup': 0.05,
    'checkpoint_minutes': 30,
    'sample_minutes': 30,
    'sample_steps': 32,
    'guidance': 2.0,
    'seed': 0,
}


def load_settings(path):
    settings = {**DEFAULTS, **json.loads(Path(path).read_text())}
    if settings['mults'] is None:
        if settings['size'] not in MULTS:
            raise SystemExit(f"size must be one of {sorted(MULTS)} (or set mults yourself)")
        settings['mults'] = MULTS[settings['size']]
    return settings




def find_dataset(root, name):
    """input_root/<name>, or the shallowest folder called <name> up to three levels down."""
    if (root / name).is_dir():
        return root / name
    for pattern in (f'*/{name}', f'*/*/{name}', f'*/*/*/{name}'):
        found = sorted(p for p in root.glob(pattern) if p.is_dir())
        if found:
            return found[0]
    return None


def label_for(path, rule):
    """A fixed label, or the image's folder name ('folder'), or its first word ('first-word')."""
    if rule not in ('folder', 'first-word'):
        return rule.lower()
    folder = re.sub(r'[_\-]+', ' ', path.parent.name)
    folder = re.sub(r'\s*\d+$', '', folder).strip().lower()
    return folder.split()[0] if rule == 'first-word' and folder else folder


def collect(settings):
    root, rng = Path(settings['dataset_root'] or settings['input_root']), random.Random(settings['seed'])
    print('attached:', ', '.join(sorted(p.name for p in root.iterdir())) if root.is_dir() else f'{root} is missing')
    by_label, missing = {}, []
    for name, rule in settings['datasets'].items():
        folder = find_dataset(root, name)
        files = sorted(p for p in folder.rglob('*') if p.suffix.lower() in IMAGE_TYPES
                       and not set(p.parts) & set(settings['exclude'])) if folder else []
        if not files:
            print(f'! {name}: no images found')
            missing.append(name)
            continue
        grouped = {}
        for path in files:
            grouped.setdefault(label_for(path, rule), []).append(path)
        for label, paths in grouped.items():
            rng.shuffle(paths)
            by_label.setdefault(label, []).extend(paths[:settings['class_caps'].get(name)])
        print(f'{name}: {len(files):,} images in {folder}')
    # The 2026-11-04 session lost seven of eight datasets, skipped them, and spent 8 GPU hours on anime faces alone.
    if missing and not settings['allow_missing']:
        raise SystemExit(f'Stopped before training: no images for {", ".join(missing)}. See the "!" lines above. '
                         'To train without them on purpose, set allow_missing to True.')
    subjects = {}
    for label, files in sorted(by_label.items()):
        if len(files) < settings['min_per_class']:
            print(f'  left out {label!r}: only {len(files)} images')
            continue
        rng.shuffle(files)
        subjects[label] = files[:settings['max_per_class']]
    return subjects


def load_square(job):
    """The centre square of an image (a path or file object), resized; None for one that will not open."""
    source, size, rotation = job
    from PIL import Image
    try:
        with Image.open(source) as image:
            image.draft('RGB', (size * 2, size * 2))
            image = image.convert('RGB')
            if rotation:
                image = image.rotate(rotation, expand=True)  # the listed angle is counter-clockwise; checked by eye on real pictures
            w, h = image.size
            s = min(w, h)
            left, top = (w - s) // 2, (h - s) // 2
            image = image.crop((left, top, left + s, top + s)).resize((size, size), Image.LANCZOS)
            return np.asarray(image, dtype=np.uint8)
    except Exception:
        return None


def prepared_key(settings):
    """The settings that decide the prepared images; when any of them changes, the images are prepared again."""
    open_images = settings['open_images'] and {k: v for k, v in settings['open_images'].items() if k != 'lists_dir'}
    return {'open_images': open_images, 'label_slots': settings['label_slots'],
            **{k: settings[k] for k in ('datasets', 'class_caps', 'exclude', 'size', 'max_per_class', 'min_per_class', 'seed', 'input_root')}}


def fetch_list(settings, name):
    folder = Path(settings['open_images'].get('lists_dir') or Path(settings['dataset_root'] or '/tmp') / 'open-images-lists')
    path = folder / Path(name).name
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix('.part')
        request = urllib.request.Request(OPEN_IMAGES_LISTS + name, headers=USER_AGENT)
        with urllib.request.urlopen(request, timeout=120) as response, temp.open('wb') as out:
            shutil.copyfileobj(response, out, 1 << 20)
        temp.replace(path)
    return path


def open_images_plan(settings):
    """[(picture id, rotation, its label names)]. Labels with at least min_label_images pictures become subjects. Rarest label first,
    pictures are added until a label has per_label of them, so that people and clothes (half the set) do not swamp the rest."""
    config, rng = settings['open_images'], random.Random(settings['seed'])
    per_label, max_images, least = config.get('per_label', 1000), config.get('max_images', 1200000), config.get('min_label_images', 100)
    with fetch_list(settings, OPEN_IMAGES_NAMES).open(encoding='utf-8', newline='') as f:
        names = {row['LabelName']: row['DisplayName'].lower() for row in csv.DictReader(f)}
    with fetch_list(settings, OPEN_IMAGES_ROWS).open(encoding='utf-8', newline='') as f:
        rotations = {row['ImageID']: int(float(row['Rotation'] or 0)) for row in csv.DictReader(f)}
    labelled = {}
    with fetch_list(settings, OPEN_IMAGES_LABELS).open(encoding='utf-8') as f:
        next(f)
        for line in f:
            picture, _, label, confidence = line.rstrip('\n').split(',')
            if confidence in ('1', '1.0') and picture in rotations and label in names:
                labelled.setdefault(picture, set()).add(names[label])
    blocked = {name for name in names.values() if name in LEFT_OUT or LEFT_OUT_WORDS.search(name)}
    kept = {picture: words for picture, words in labelled.items()
            if not words & blocked and not (words & CHILDREN and words & NOT_WITH_CHILDREN)}
    print(f'Open Images: {len(labelled):,} pictures with labels, {len(labelled) - len(kept):,} left out by the content rules', flush=True)
    counts = Counter(word for words in kept.values() for word in words)
    subjects = {word for word, count in counts.items() if count >= least}
    order = sorted(kept)
    rng.shuffle(order)
    by_label = {}
    for picture in order:
        for word in kept[picture] & subjects:
            by_label.setdefault(word, []).append(picture)
    chosen, have = {}, Counter()
    for word in sorted(by_label, key=lambda w: (counts[w], w)):
        for picture in by_label[word]:
            if have[word] >= per_label:
                break
            if picture not in chosen:
                chosen[picture] = True
                have.update(kept[picture] & subjects)
    picked = list(chosen)
    rng.shuffle(picked)
    picked = picked[:max_images]
    slots = settings['label_slots']
    plan = [(picture, rotations[picture], sorted(kept[picture] & subjects, key=lambda w: (counts[w], w))[:slots]) for picture in picked]
    print(f'Open Images: {len(subjects):,} subjects, {len(plan):,} pictures chosen', flush=True)
    return plan


def load_job(job):
    """(row, image, JPEG bytes); the image is None when the picture could not be fetched or opened."""
    row, source, size, rotation = job
    from PIL import Image
    array = None
    for attempt in range(3):
        try:
            if isinstance(source, Path):
                array = load_square((source, size, rotation))
            else:
                request = urllib.request.Request(OPEN_IMAGES_PICTURE.format(source), headers=USER_AGENT)
                with urllib.request.urlopen(request, timeout=60) as response:
                    array = load_square((io.BytesIO(response.read()), size, rotation))
            break
        except Exception:
            time.sleep(attempt)
    if array is None:
        return row, None, b''
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, 'JPEG', quality=95, subsampling=0)
    return row, array, buffer.getvalue()


def find_pack(settings, key):
    """A folder of prepared images saved by an earlier session of this notebook, in this run or attached as an input."""
    run, root = Path(settings['run_dir']), Path(settings['input_root'])
    found = [run / 'prepared' / 'manifest.json'] + (sorted(root.rglob(f'{run.name}/prepared/manifest.json')) if root.is_dir() else [])
    for manifest in found:
        if manifest.exists() and json.loads(manifest.read_text()).get('key') == key:
            return manifest.parent
    return None


def unpack(pack, out, settings):
    """The saved pictures back into images.npy, which takes minutes instead of a new download. The pack also goes into this session's
    output, so that the next session can attach just this one."""
    from multiprocessing.pool import ThreadPool
    from PIL import Image
    size, started = settings['size'], time.time()
    labels, rows, ends = np.load(pack / 'labels.npy'), np.load(pack / 'rows.npy'), np.load(pack / 'ends.npy')
    out.mkdir(parents=True, exist_ok=True)
    images = np.lib.format.open_memmap(out / 'images.npy', mode='w+', dtype=np.uint8, shape=(len(labels), size, size, 3))

    def decode(job):
        row, jpeg = job
        with Image.open(io.BytesIO(jpeg)) as image:
            images[row] = np.asarray(image.convert('RGB'), dtype=np.uint8)

    def jobs():
        with (pack / 'images.bin').open('rb') as source:
            at = 0
            for row, end in zip(rows.tolist(), ends.tolist()):
                yield row, source.read(end - at)
                at = end

    with ThreadPool(os.cpu_count()) as pool:
        for _ in pool.imap_unordered(decode, jobs(), chunksize=64):
            pass
    images.flush()
    np.save(out / 'labels.npy', labels)
    keep = Path(settings['run_dir']) / 'prepared'
    if pack.resolve() != keep.resolve():
        keep.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(keep, ignore_errors=True)
        shutil.copytree(pack, keep)
    shutil.copy(pack / 'manifest.json', out / 'manifest.json')
    print(f'{len(rows):,} prepared images from {pack} in {time.time() - started:.0f}s', flush=True)


def prepare(settings):
    from multiprocessing.pool import ThreadPool
    out = Path(settings['data_dir'])
    key = prepared_key(settings)
    if (out / 'manifest.json').exists() and json.loads((out / 'manifest.json').read_text()).get('key') == key:
        print(f'{out} is already prepared')
        return
    pack = find_pack(settings, key)
    if pack:
        unpack(pack, out, settings)
        return
    subjects = collect(settings) if settings['datasets'] else {}
    plan = open_images_plan(settings) if settings['open_images'] else []
    if not subjects and not plan:
        raise SystemExit('No images found. Attach datasets with "Add Data" and check DATASETS in the settings cell.')
    size, started = settings['size'], time.time()
    if plan:  # stops here, before hours of nothing, when the pictures cannot be reached from this machine
        probe = [load_job((i, picture, size, rotation)) for i, (picture, rotation, _) in enumerate(plan[:20])]
        working = sum(array is not None for _, array, _ in probe)
        print(f'Open Images: {working} of {len(probe)} test pictures arrived, {time.time() - started:.1f}s', flush=True)
        if working < len(probe) / 2:
            raise SystemExit('Stopped before training: Open Images pictures cannot be downloaded. Is Settings → Internet on?')
    names = sorted(set(subjects) | {word for _, _, words in plan for word in words})
    index, null = {name: i for i, name in enumerate(names)}, len(names)
    files = [(path, 0, [index[name]]) for name, paths in sorted(subjects.items()) for path in paths]
    rows = files + [(picture, rotation, [index[word] for word in words]) for picture, rotation, words in plan]
    out.mkdir(parents=True, exist_ok=True)
    keep = Path(settings['run_dir']) / 'prepared'
    keep.mkdir(parents=True, exist_ok=True)
    images = np.lib.format.open_memmap(out / 'images.npy', mode='w+', dtype=np.uint8, shape=(len(rows), size, size, 3))
    labels = np.full((len(rows), settings['label_slots']), null, dtype=np.int32)
    done_rows, ends, at, broken, deadline = [], [], 0, 0, started + settings['prepare_hours'] * 3600
    jobs = ((i, source, size, rotation) for i, (source, rotation, _) in enumerate(rows))
    with ThreadPool(settings['prepare_threads']) as pool, (keep / 'images.bin').open('wb') as pack_file:
        for k, (row, array, jpeg) in enumerate(pool.imap_unordered(load_job, jobs, chunksize=8)):
            if array is None:
                broken += 1
            else:
                images[row] = array
                labels[row, :len(rows[row][2])] = rows[row][2]
                pack_file.write(jpeg)
                at += len(jpeg)
                done_rows.append(row)
                ends.append(at)
            if (k + 1) % 20000 == 0:
                print(f'  {k + 1:,}/{len(rows):,} images, {broken:,} failed ({time.time() - started:.0f}s)', flush=True)
            if time.time() > deadline:
                print(f'! out of time after {k + 1:,} of {len(rows):,} images; training on those', flush=True)
                pool.terminate()
                break
    images.flush()
    np.save(out / 'labels.npy', labels)
    np.save(keep / 'labels.npy', labels)
    np.save(keep / 'rows.npy', np.array(done_rows, dtype=np.int64))
    np.save(keep / 'ends.npy', np.array(ends, dtype=np.int64))
    (keep / 'open-images-ids.txt').write_text(''.join(f'{picture}\n' for picture, _, _ in plan))
    held = labels[labels < null]
    tally = np.bincount(held, minlength=null)
    manifest = json.dumps({'key': key, 'subjects': names, 'counts': {name: int(tally[i]) for i, name in enumerate(names)}, 'broken': broken}, indent=1)
    if broken > len(rows) * 0.2 and not settings['allow_missing']:
        raise SystemExit(f'Stopped before training: {broken:,} of {len(rows):,} images failed. To train anyway, set allow_missing to True.')
    (out / 'manifest.json').write_text(manifest)
    (keep / 'manifest.json').write_text(manifest)  # last, so that a pack with a manifest is complete
    print(f'{len(done_rows):,} images, {len(names):,} subjects, {broken:,} failed, {time.time() - started:.0f}s, {at / 1e9:.1f} GB saved for later sessions')
    most = sorted(((int(tally[i]), name) for i, name in enumerate(names)), reverse=True)
    print('most common:', ', '.join(f'{name} {count:,}' for count, name in most[:40]))


def real_grid(settings, out, columns=8):
    """A grid of training images, one row per subject (up to 16), to check the data."""
    data = Path(settings['data_dir'])
    names = json.loads((data / 'manifest.json').read_text())['subjects']
    images, labels = np.load(data / 'images.npy', mmap_mode='r'), np.load(data / 'labels.npy')
    rng, rows, shown = np.random.default_rng(0), [], []
    for prompt in showcase(settings, names):
        wanted = labels_for(prompt, names)[0]
        matching = np.flatnonzero(np.all([(labels == i).any(axis=1) for i in wanted], axis=0))
        if len(matching):
            picks = rng.choice(matching, columns)
            rows.append(np.concatenate(list(images[np.sort(picks)]), axis=1))
            shown.append(prompt)
    save_grid(np.concatenate(rows, axis=0), out)
    print('rows:', ', '.join(shown))




class Fourier(nn.Module):
    """Random Fourier features of the noise level."""

    def __init__(self, channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(channels // 2) * scale)

    def forward(self, x):
        x = x.float()[:, None] * (2 * math.pi * self.freqs)[None]
        return torch.cat([x.cos(), x.sin()], dim=1)


class Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.heads = max(1, channels // 64)
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.out = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).permute(1, 0, 2, 4, 3)
        y = F.scaled_dot_product_attention(q, k, v).permute(0, 1, 3, 2).reshape(b, c, h, w)
        return (x + self.out(y)) * math.sqrt(0.5)


class Block(nn.Module):
    """Residual block; the noise level and subject set a scale and shift after the second norm."""

    def __init__(self, cin, cout, emb, dropout, attention):
        super().__init__()
        self.norm1, self.conv1 = nn.GroupNorm(32, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb, cout * 2)
        self.norm2, self.drop = nn.GroupNorm(32, cout), nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.attention = Attention(cout) if attention else None

    def forward(self, x, e):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(e)[:, :, None, None].chunk(2, dim=1)
        h = self.conv2(self.drop(F.silu(self.norm2(h) * (1 + scale) + shift)))
        x = (self.skip(x) + h) * math.sqrt(0.5)
        return self.attention(x) if self.attention else x


class Down(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))


class UNet(nn.Module):
    def __init__(self, size, classes, channels=128, mults=(1, 2, 2, 4), blocks=2, attention=(16, 8), dropout=0.1):
        super().__init__()
        emb = channels * 4
        self.classes = classes
        self.fourier = Fourier(channels)
        self.noise = nn.Sequential(nn.Linear(channels, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.label = nn.Embedding(classes + 1, emb)
        self.inp = nn.Conv2d(3, channels, 3, padding=1)
        self.down, skips, ch, res = nn.ModuleList(), [channels], channels, size
        for level, mult in enumerate(mults):
            for _ in range(blocks):
                self.down.append(Block(ch, channels * mult, emb, dropout, res in attention))
                ch = channels * mult
                skips.append(ch)
            if level < len(mults) - 1:
                self.down.append(Down(ch))
                skips.append(ch)
                res //= 2
        self.mid = nn.ModuleList([Block(ch, ch, emb, dropout, True), Block(ch, ch, emb, dropout, False)])
        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(mults))):
            for _ in range(blocks + 1):
                self.up.append(Block(ch + skips.pop(), channels * mult, emb, dropout, res in attention))
                ch = channels * mult
            if level > 0:
                self.up.append(Up(ch))
                res *= 2
        self.out_norm, self.out = nn.GroupNorm(32, ch), nn.Conv2d(ch, 3, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def condition(self, labels):
        """One label per picture, or a row of labels added together; `classes` fills unused places, and a row of nothing but that means no label."""
        if labels.dim() == 1:
            return self.label(labels)
        present = labels != self.classes
        count = present.sum(1, keepdim=True)
        total = (self.label(labels) * present.unsqueeze(-1)).sum(1) / count.clamp(min=1).sqrt()
        return torch.where(count > 0, total, self.label.weight[self.classes].unsqueeze(0))

    def forward(self, x, c_noise, labels):
        e = F.silu(self.noise(self.fourier(c_noise)) + self.condition(labels))
        h = self.inp(x)
        hs = [h]
        for layer in self.down:
            h = layer(h, e) if isinstance(layer, Block) else layer(h)
            hs.append(h)
        for layer in self.mid:
            h = layer(h, e)
        for layer in self.up:
            h = layer(torch.cat([h, hs.pop()], dim=1), e) if isinstance(layer, Block) else layer(h)
        return self.out(F.silu(self.out_norm(h)))


def build(settings, classes):
    return UNet(settings['size'], classes, settings['channels'], settings['mults'], settings['blocks'],
                settings['attention'], settings['dropout'])


def denoise(net, x, sigma, labels):
    """EDM's D(x; sigma): the network output is mixed with the input, scaled for every noise level."""
    s = sigma.float().reshape(-1, 1, 1, 1)
    c_skip = SIGMA_DATA ** 2 / (s ** 2 + SIGMA_DATA ** 2)
    c_out = s * SIGMA_DATA / (s ** 2 + SIGMA_DATA ** 2).sqrt()
    c_in = 1 / (s ** 2 + SIGMA_DATA ** 2).sqrt()
    return c_skip * x + c_out * net(c_in * x, s.flatten().log() / 4, labels).float()


def edm_loss(net, x, labels):
    sigma = (torch.randn(x.shape[0], device=x.device) * P_STD + P_MEAN).exp()
    weight = ((sigma ** 2 + SIGMA_DATA ** 2) / (sigma * SIGMA_DATA) ** 2)[:, None, None, None]
    noisy = x + torch.randn_like(x) * sigma[:, None, None, None]
    return (weight * (denoise(net, noisy, sigma, labels) - x) ** 2).mean()


@torch.no_grad()
def sample(net, labels, size, steps=32, guidance=2.0, seed=0, chunk=64):
    """Images in [-1, 1] by EDM's deterministic Heun sampler with classifier-free guidance."""
    device = next(net.parameters()).device
    i = torch.arange(steps, dtype=torch.float64)
    t = (SIGMA_MAX ** (1 / RHO) + i / max(steps - 1, 1) * (SIGMA_MIN ** (1 / RHO) - SIGMA_MAX ** (1 / RHO))) ** RHO
    t = torch.cat([t, torch.zeros(1, dtype=torch.float64)]).tolist()
    generator = torch.Generator(device=device).manual_seed(seed)
    results = []
    for start in range(0, len(labels), chunk):
        y = torch.tensor(labels[start:start + chunk], device=device)
        null = torch.full_like(y, net.classes)
        n = len(y)

        def guided(x, sigma):
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                both = denoise(net, torch.cat([x, x]), torch.full((2 * n,), sigma, device=device), torch.cat([y, null]))
            cond, uncond = both.chunk(2)
            return uncond + guidance * (cond - uncond)

        x = torch.randn(n, 3, size, size, device=device, generator=generator) * t[0]
        for k in range(steps):
            d = (x - guided(x, t[k])) / t[k]
            nxt = x + (t[k + 1] - t[k]) * d
            if t[k + 1] > 0:
                nxt = x + (t[k + 1] - t[k]) * (d + (nxt - guided(nxt, t[k + 1])) / t[k + 1]) / 2
            x = nxt
        results.append(x.clamp(-1, 1).float().cpu())
    return torch.cat(results)




def to_uint8(images):
    return ((images.permute(0, 2, 3, 1).numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)


def save_grid(array, path, scale=2):
    from PIL import Image
    image = Image.fromarray(array)
    image.resize((image.width * scale, image.height * scale), Image.NEAREST).save(path)


def spread(names, rows=16):
    """Up to `rows` subjects spread evenly over the sorted list."""
    if len(names) <= rows:
        return list(names)
    return [names[int(i)] for i in np.linspace(0, len(names) - 1, rows).round()]


def labels_for(prompt, subjects):
    """(label numbers, words not understood) for a prompt such as "a sketch of two dogs". Longest phrases first; plurals work."""
    known = {}
    for i, name in enumerate(subjects):
        known[name] = i
    for i, name in enumerate(subjects):
        known.setdefault(re.sub(r'\s*\(.*\)$', '', name), i)  # "trunk (tree)" also answers to "trunk"
    words = re.findall(r"[a-z0-9'&-]+", prompt.lower())
    found, unknown, at = [], [], 0
    while at < len(words):
        filler = words[at] in PROMPT_FILLER  # "picture" alone is nothing, but "picture frame" is a subject
        for length in range(min(4, len(words) - at), 1 if filler else 0, -1):
            phrase = ' '.join(words[at:at + length])
            forms = [phrase] + ([phrase[:-1]] if phrase.endswith('s') else []) + ([phrase[:-2]] if phrase.endswith('es') else [])
            forms += [PROMPT_WORDS[form] for form in forms if form in PROMPT_WORDS]
            hit = next((known[form] for form in forms if form in known), None)
            if hit is not None:
                found.append(hit)
                at += length
                break
        else:
            if not filler:
                unknown.append(words[at])
            at += 1
    return list(dict.fromkeys(found)), unknown


def not_drawable(prompt):
    """Is the prompt asking for something left out of training? Refused, rather than drawn without the word."""
    text = prompt.lower()
    return bool(LEFT_OUT_WORDS.search(text)) or bool(set(re.findall(r"[a-z-]+", text)) & (LEFT_OUT - {'love', 'model', 'skin', 'hip', 'muscle', 'blood', 'stocking', 'waist'}))


def label_rows(wanted, slots, null):
    """Equal-length rows for sample(): each row of label numbers, filled up with `null`."""
    return [(row + [null] * slots)[:slots] for row in wanted]


def showcase(settings, subjects):
    """The prompts to show: the settings' list where the model knows every word, else subjects spread over the list."""
    prompts = []
    for prompt in settings.get('showcase') or []:
        wanted, unknown = labels_for(prompt, subjects)
        if wanted and not unknown:
            prompts.append(prompt)
    return prompts or spread(subjects)


def sample_grid(net, settings, subjects, path, columns=8, seed=0):
    rows = showcase(settings, subjects)
    wanted = [labels_for(prompt, subjects)[0] for prompt in rows for _ in range(columns)]
    labels = label_rows(wanted, settings['label_slots'], len(subjects))
    images = to_uint8(sample(net, labels, settings['size'], settings['sample_steps'], settings['guidance'], seed))
    grid = images.reshape(len(rows), columns, *images.shape[1:]).transpose(0, 2, 1, 3, 4)
    save_grid(grid.reshape(len(rows) * settings['size'], columns * settings['size'], 3), path)




def find_checkpoint(settings):
    """checkpoint.pt in run_dir, else in an attached earlier output of this notebook."""
    run = Path(settings['run_dir'])
    if (run / 'checkpoint.pt').exists():
        return run / 'checkpoint.pt'
    root = Path(settings['input_root'])
    found = sorted(root.rglob(f'{run.name}/checkpoint.pt'), key=lambda path: len(path.parts)) if root.is_dir() else []
    return found[0] if found else None


def warm(net, trained, old_subjects, new_subjects):
    """Load weights trained on other subjects: everything but the label table as it is, and each subject's row where the old list had it."""
    table = net.label.weight.detach().cpu().clone()
    for i, name in enumerate(new_subjects):
        if name in old_subjects:
            table[i] = trained['label.weight'][old_subjects.index(name)]
    table[len(new_subjects)] = trained['label.weight'][len(old_subjects)]  # the "no label" row
    net.load_state_dict({**trained, 'label.weight': table})
    return len(set(new_subjects) & set(old_subjects))


def save_checkpoint(path, net, ema, opt, scaler, step, seen, subjects, settings):
    temp = path.with_suffix('.tmp')
    torch.save({'net': net.state_dict(), 'ema': ema.state_dict(), 'opt': opt.state_dict(), 'scaler': scaler.state_dict(),
                'step': step, 'images_seen': seen, 'subjects': subjects, 'settings': settings}, temp)
    temp.replace(path)


def export(ema, settings, subjects, step, seen):
    """kani-pixels-v3.npz (EMA weights) and kani-pixels-v3.json (everything needed to rebuild the model)."""
    run = Path(settings['run_dir'])
    np.savez(run / 'kani-pixels-v3.npz', **{k: v.detach().float().cpu().numpy() for k, v in ema.state_dict().items()})
    config = {'model': 'Kani Pixels v3', 'subjects': subjects, 'steps': step, 'images_seen': seen,
              'parameters': sum(p.numel() for p in ema.parameters()), 'sigma_data': SIGMA_DATA,
              **{k: settings[k] for k in ('size', 'channels', 'mults', 'blocks', 'attention', 'dropout', 'sample_steps', 'guidance', 'label_slots')}}
    (run / 'kani-pixels-v3.json').write_text(json.dumps(config, indent=1) + '\n')
    print(f'exported {run / "kani-pixels-v3.npz"} ({config["parameters"]:,} parameters, step {step:,})', flush=True)


def train(settings):
    distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    cuda = torch.cuda.is_available()
    if distributed:
        torch.distributed.init_process_group('nccl' if cuda else 'gloo')
    rank = torch.distributed.get_rank() if distributed else 0
    world = torch.distributed.get_world_size() if distributed else 1
    local = int(os.environ.get('LOCAL_RANK', '0'))
    device = torch.device('cuda', local) if cuda else torch.device('cpu')
    if cuda:
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
    main = rank == 0
    if settings['batch'] % world:
        raise SystemExit(f"batch {settings['batch']} must divide evenly over {world} GPUs")
    per = settings['batch'] // world

    data = Path(settings['data_dir'])
    subjects = json.loads((data / 'manifest.json').read_text())['subjects']
    images, labels = np.load(data / 'images.npy', mmap_mode='r'), np.load(data / 'labels.npy')
    valid = np.flatnonzero(labels[:, 0] < len(subjects))
    run = Path(settings['run_dir'])
    if main:
        (run / 'samples').mkdir(parents=True, exist_ok=True)

    torch.manual_seed(settings['seed'])
    net = build(settings, len(subjects)).to(device).to(memory_format=torch.channels_last)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=settings['lr'], betas=(0.9, 0.999), eps=1e-8)
    scaler = torch.amp.GradScaler('cuda', enabled=cuda)
    step = seen = 0
    checkpoint = find_checkpoint(settings)
    if checkpoint:
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state['subjects'] == subjects:
            net.load_state_dict(state['net'])
            ema.load_state_dict(state['ema'])
            opt.load_state_dict(state['opt'])
            scaler.load_state_dict(state['scaler'])
            step, seen = state['step'], state['images_seen']
            if main:
                print(f'resumed from {checkpoint}: step {step:,}, {seen / 1e6:.2f}M images seen', flush=True)
        else:
            same = all(state['settings'][k] == settings[k] for k in ('size', 'channels', 'mults', 'blocks', 'attention'))
            if not settings['warm_start'] or not same:
                raise SystemExit(f'{checkpoint} was trained on different subjects or another model size; use the same settings or a new run_dir')
            shared = warm(net, state['net'], state['subjects'], subjects)
            warm(ema, state['ema'], state['subjects'], subjects)
            if main:  # a new optimizer and step count: the label table changed shape, and the learning rate warms up again
                print(f'warm start from {checkpoint} (it had seen {state["images_seen"] / 1e6:.2f}M images): {shared} of its '
                      f'{len(state["subjects"])} subjects are among the {len(subjects):,} now', flush=True)
    elif main:
        print('fresh start: no earlier checkpoint found', flush=True)
    model = nn.parallel.DistributedDataParallel(net, device_ids=[local] if cuda else None) if distributed else net
    if main:
        print(json.dumps({'parameters': sum(p.numel() for p in net.parameters()), 'subjects': len(subjects),
                          'images': len(valid), 'gpus': world, 'batch': settings['batch'], 'device': str(device)}), flush=True)

    rng = np.random.default_rng([settings['seed'], rank, step])
    null, halflife = len(subjects), settings['ema_halflife_kimg'] * 1000
    started = last_checkpoint = last_sample = time.time()
    budget = settings['hours'] * 3600 - (time.time() - settings['session_started'] if settings['session_started'] else 0)
    stop, stopping = torch.zeros(1, device=device), False
    total, count, window = torch.zeros((), device=device), 0, time.time()
    while True:
        index = np.sort(rng.choice(valid, per))
        x = torch.from_numpy(images[index]).to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 127.5 - 1
        flip = torch.rand(per, device=device) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(3), x).contiguous(memory_format=torch.channels_last)
        y = torch.from_numpy(labels[index]).to(device).long()
        score = torch.rand(y.shape, device=device).masked_fill(y == null, -1)
        keep = (score >= 1 - settings['label_keep']) | (score == score.max(dim=1, keepdim=True).values)  # at least one stays
        y = torch.where(keep, y, null)
        y = torch.where(torch.rand(per, 1, device=device) < settings['label_dropout'], null, y)
        for group in opt.param_groups:
            group['lr'] = settings['lr'] * min(1.0, (step + 1) / settings['warmup_steps'])
        with torch.autocast(device.type, dtype=torch.float16, enabled=cuda):
            loss = edm_loss(model, x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        seen += settings['batch']

        beta = 0.5 ** (settings['batch'] / max(min(halflife, seen * settings['ema_rampup']), 1e-8))
        with torch.no_grad():
            for e, p in zip(ema.parameters(), net.parameters()):
                e.lerp_(p, 1 - beta)
        total += loss.detach()
        count += 1

        if step % 50 == 0:
            done = time.time() - started > budget or (settings['steps'] and step >= settings['steps'])
            stop.fill_(float(bool(done)))
            if distributed:
                torch.distributed.broadcast(stop, 0)
            stopping = bool(stop.item())
        if main and (step % 200 == 0 or stopping):
            rate = count * settings['batch'] / (time.time() - window)
            status = {'step': step, 'images_seen': seen, 'loss': round(total.item() / count, 5),
                      'images_per_second': round(rate), 'hours_left': round(max(0, budget - (time.time() - started)) / 3600, 2)}
            print(json.dumps(status), flush=True)
            (run / 'status.json').write_text(json.dumps({**status, 'state': 'running', 'updated': time.time()}))
            total, count, window = torch.zeros((), device=device), 0, time.time()
        if main and time.time() - last_sample > settings['sample_minutes'] * 60:
            sample_grid(ema, settings, subjects, run / 'samples' / f'{seen // 1000:08d}k.png')
            last_sample = time.time()
        if main and time.time() - last_checkpoint > settings['checkpoint_minutes'] * 60:
            save_checkpoint(run / 'checkpoint.pt', net, ema, opt, scaler, step, seen, subjects, settings)
            last_checkpoint = time.time()
            print(f'checkpoint at step {step:,}', flush=True)
        if stopping:
            break

    if main:
        save_checkpoint(run / 'checkpoint.pt', net, ema, opt, scaler, step, seen, subjects, settings)
        export(ema, settings, subjects, step, seen)
        sample_grid(ema, settings, subjects, run / 'samples' / f'{seen // 1000:08d}k.png')
        (run / 'status.json').write_text(json.dumps({'step': step, 'images_seen': seen, 'state': 'stopped', 'updated': time.time()}))
        print(f'stopped after {(time.time() - started) / 3600:.2f} h: step {step:,}, {seen / 1e6:.2f}M images seen', flush=True)
    if distributed:
        torch.distributed.destroy_process_group()




def load_trained(settings):
    """The EMA model from kani-pixels-v3.npz, or from checkpoint.pt when there is no export yet."""
    run = Path(settings['run_dir'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if (run / 'kani-pixels-v3.npz').exists():
        config = json.loads((run / 'kani-pixels-v3.json').read_text())
        net = build({**settings, **config}, len(config['subjects']))
        with np.load(run / 'kani-pixels-v3.npz') as weights:
            net.load_state_dict({k: torch.from_numpy(weights[k]) for k in weights.files})
        return net.to(device).eval(), {**settings, **config}, config['subjects']
    checkpoint = find_checkpoint(settings)
    if checkpoint is None:
        raise SystemExit('No kani-pixels-v3.npz or checkpoint.pt yet; train first.')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    net = build(state['settings'], len(state['subjects']))
    net.load_state_dict(state['ema'])
    return net.to(device).eval(), state['settings'], state['subjects']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['prepare', 'real', 'train', 'sample', 'export'])
    parser.add_argument('--settings', default='settings.json')
    parser.add_argument('--subject', help='sample: which subject (see kani-pixels-v3.json)')
    parser.add_argument('--prompt', help='sample: words naming subjects, such as "a drawing of a cat"')
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--guidance', type=float, help='sample: override the guidance strength')
    parser.add_argument('--out', default='samples.png')
    args = parser.parse_args()
    settings = load_settings(args.settings)
    if args.command == 'prepare':
        prepare(settings)
    elif args.command == 'real':
        real_grid(settings, args.out)
    elif args.command == 'train':
        train(settings)
    elif args.command == 'export':
        checkpoint = find_checkpoint(settings)
        if checkpoint is None:
            raise SystemExit('No checkpoint.pt found.')
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        net = build(state['settings'], len(state['subjects']))
        net.load_state_dict(state['ema'])
        Path(settings['run_dir']).mkdir(parents=True, exist_ok=True)
        export(net, {**state['settings'], 'run_dir': settings['run_dir']}, state['subjects'], state['step'], state['images_seen'])
    else:
        net, trained, subjects = load_trained(settings)
        if args.prompt:
            if not_drawable(args.prompt):
                raise SystemExit('This model was not trained to draw that.')
            wanted, unknown = labels_for(args.prompt, subjects)
            if not wanted:
                raise SystemExit(f'No subject in {args.prompt!r}; see the subjects in kani-pixels-v3.json')
            subject = ' + '.join(subjects[i] for i in wanted)
            if unknown:
                print(f'not used, not a subject: {", ".join(unknown)}')
        else:
            text = (args.subject or '').lower()
            subject = next((s for s in subjects if s == text), None) or next((s for s in subjects if text and text in s), None)
            if subject is None:
                raise SystemExit(f'Pick --subject from: {", ".join(subjects)}')
            wanted = [subjects.index(subject)]
        rows = label_rows([wanted] * args.count, max(trained.get('label_slots', 8), len(wanted)), len(subjects))
        images = to_uint8(sample(net, rows, trained['size'], trained['sample_steps'],
                                 args.guidance if args.guidance is not None else trained['guidance'], args.seed))
        columns = min(8, args.count)
        rows = -(-args.count // columns)
        pad = np.zeros((rows * columns - len(images), *images.shape[1:]), dtype=np.uint8)
        grid = np.concatenate([images, pad]).reshape(rows, columns, *images.shape[1:]).transpose(0, 2, 1, 3, 4)
        save_grid(grid.reshape(rows * trained['size'], columns * trained['size'], 3), args.out, scale=4)
        print(json.dumps({'subject': subject, 'out': args.out}))


if __name__ == '__main__':
    main()
