"""Training images for Kani's pixel model: Open Images photos (CC BY 2.0) shrunk to 64x64.

Subjects are the Open Images boxable classes that are also Quick, Draw! subjects (data/quickdraw/
all-categories.txt), plus people, so the pixel model and Kani Sketch cover the same everyday things.
For each subject, up to --per-class training photos with a human-verified label are downloaded from
the Open Images bucket, center-cropped, resized to 64x64 and kept only in that form; the original
JPEG is never written to disk. Author, license and source URL of every photo are recorded.

    python3 data/make_pixels.py --per-class 300      # writes data/pixels-v1/
"""
import argparse
import concurrent.futures
import csv
import io
import json
from pathlib import Path
import random
import ssl
import urllib.request

import numpy as np

try:
    import certifi
    TLS = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    TLS = ssl.create_default_context()

ROOT = Path(__file__).resolve().parent.parent
META = 'https://storage.googleapis.com/openimages'
FILES = {'classes': f'{META}/v7/oidv7-class-descriptions-boxable.csv',
         'labels': f'{META}/2018_04/train/train-annotations-human-imagelabels-boxable.csv',
         'images': f'{META}/2018_04/train/train-images-boxable-with-rotation.csv'}
BUCKET = 'https://open-images-dataset.s3.amazonaws.com/train/{}.jpg'
EXTRA = {'person', 'man', 'woman', 'girl', 'boy', 'flower', 'fruit', 'food', 'tree', 'house', 'building'}
SIZE = 64


def fetch(url, path):
    if not path.exists():
        with urllib.request.urlopen(url, timeout=120, context=TLS) as response, open(path.with_suffix('.part'), 'wb') as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        path.with_suffix('.part').rename(path)
    return path


def thumbnail(data, rotation):
    from PIL import Image
    image = Image.open(io.BytesIO(data)).convert('RGB')
    if rotation:
        image = image.rotate(-rotation, expand=True)
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    return np.asarray(image.crop((left, top, left + side, top + side)).resize((SIZE, SIZE), Image.LANCZOS), dtype=np.uint8)


def download(item):
    image_id, rotation = item
    try:
        with urllib.request.urlopen(BUCKET.format(image_id), timeout=60, context=TLS) as response:
            return image_id, thumbnail(response.read(), rotation)
    except Exception as error:
        return image_id, error


def build(out, per_class=300, seed=2038, workers=16):
    raw = ROOT / 'data/openimages'
    raw.mkdir(parents=True, exist_ok=True)
    paths = {name: fetch(url, raw / Path(url).name) for name, url in FILES.items()}
    quickdraw = {line.strip().lower() for line in (ROOT / 'data/quickdraw/all-categories.txt').read_text().splitlines() if line.strip()}
    classes = {}
    with open(paths['classes'], newline='', encoding='utf-8') as stream:
        for label, name in csv.reader(stream):
            if name.lower() in quickdraw or name.lower() in EXTRA:
                classes[label] = name
    rng = random.Random(seed)
    candidates = {label: [] for label in classes}
    with open(paths['labels'], newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            if row['LabelName'] in candidates and row['Confidence'] == '1':
                candidates[row['LabelName']].append(row['ImageID'])
    chosen, taken = {}, set()
    for label, ids in sorted(candidates.items()):
        rng.shuffle(ids)
        picks = [i for i in ids if i not in taken][:per_class]
        taken.update(picks)
        for image_id in picks:
            chosen[image_id] = label
    info = {}
    with open(paths['images'], newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            if row['ImageID'] in chosen:
                info[row['ImageID']] = {k: row.get(k, '') for k in ('Author', 'License', 'OriginalURL', 'OriginalLandingURL', 'Rotation')}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = sorted(set(classes[label] for label in chosen.values()))
    index = {name: i for i, name in enumerate(names)}
    images, labels, sources, failed = [], [], [], 0
    items = [(i, int(float(info.get(i, {}).get('Rotation') or 0))) for i in chosen]
    with concurrent.futures.ThreadPoolExecutor(workers) as pool, open(out / 'sources.jsonl', 'w', encoding='utf-8') as credits:
        for count, (image_id, result) in enumerate(pool.map(download, items), 1):
            if isinstance(result, Exception):
                failed += 1
                continue
            images.append(result)
            labels.append(index[classes[chosen[image_id]]])
            credits.write(json.dumps({'image_id': image_id, 'subject': classes[chosen[image_id]], **info.get(image_id, {})}) + '\n')
            if count % 2000 == 0:
                print(json.dumps({'downloaded': len(images), 'failed': failed, 'of': len(items)}), flush=True)
    np.save(out / 'images.npy', np.stack(images))
    np.save(out / 'labels.npy', np.asarray(labels, dtype=np.int16))
    summary = {'subjects': names, 'images': len(images), 'failed': failed, 'size': SIZE, 'per_class': per_class,
               'license': 'Open Images photos, CC BY 2.0 (author and source per image in sources.jsonl); labels CC BY 4.0'}
    (out / 'manifest.json').write_text(json.dumps(summary, indent=1) + '\n')
    return {k: summary[k] for k in ('images', 'failed')} | {'subjects': len(names)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default=str(ROOT / 'data/pixels-v1'))
    parser.add_argument('--per-class', type=int, default=300)
    args = parser.parse_args()
    print(json.dumps(build(args.out, args.per_class)))
