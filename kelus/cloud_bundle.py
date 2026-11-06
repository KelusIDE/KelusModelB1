"""Build a private, allowlisted cloud-training bundle; does not upload anything."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

from .model import Model

MODULES = ('__init__', 'model', 'tensor', 'tokenizer', 'gpu', 'gpu_train', 'grow',
           'corpus', 'pretrain', 'optim', 'data', 'curriculum', 'languages', 'progress')


def build(root, source, out):
    root, source, out = Path(root), Path(source), Path(out)
    if out.exists():
        raise ValueError('Bundle already exists; choose a new output filename')
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as temp:
        temp = Path(temp)
        snapshot = temp / 'snapshot.npz'


        with source.open('rb') as stream, snapshot.open('wb') as target:
            shutil.copyfileobj(stream, target, 1 << 20)
        with snapshot.open('rb') as stream:
            original_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        model, header, states = Model.load(snapshot)
        del states
        parent = temp / 'parent.npz'
        model.save(parent, {**header.get('metadata', {}), 'snapshot_sha256': original_hash})
        details = {'format': 'kani-cloud-bundle-v1', 'source_step': header.get('metadata', {}).get('steps'),
                   'source_parameters': model.parameter_count, 'source_sha256': original_hash,
                   'included': {}, 'note': 'Own Kani weights, code and existing corpus; no credentials or external model weights.'}
        del model
        files = [(root/'kelus'/f'{name}.py', f'kelus/{name}.py') for name in MODULES]
        files += [(root/'data/big-v1'/name, f'data/big-v1/{name}') for name in
                  ('train.bin', 'validation.bin', 'manifest.json', 'tokenizer.json', 'sources.json', 'documents.jsonl')]
        files += [(parent, 'checkpoints/parent.npz'), (root/'LICENSE', 'LICENSE')]

        for folder in ('data/articles-v2', 'data/foundation'):
            for name in ('sources.json', 'manifest.json', 'documents.jsonl', 'train-documents.json', 'validation-documents.json'):
                path = root/folder/name
                if path.exists():
                    files.append((path, f'{folder}/{name}'))
            for path in (root/folder).rglob('*'):
                if path.is_file() and ('license' in path.name.lower() or 'notice' in path.name.lower()):
                    files.append((path, str(path.relative_to(root))))
        archive_path = temp/'bundle.zip'
        with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for path, destination in files:
                with path.open('rb') as stream:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                details['included'][destination] = {'sha256': digest, 'bytes': path.stat().st_size}
                archive.write(path, destination)
            archive.writestr('bundle.json', json.dumps(details, indent=2)+'\n')
        archive_path.replace(out)
    out.with_suffix('.json').write_text(json.dumps(details, indent=2)+'\n')
    return {key: value for key, value in details.items() if key != 'included'} | {'out': str(out), 'bytes': out.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('training-runs/kani-big-v1/latest.npz'))
    parser.add_argument('--out', type=Path, default=Path('training-runs/cloud/kani-cloud.zip'))
    args = parser.parse_args()
    print(json.dumps(build(Path(__file__).resolve().parents[1], args.source, args.out), indent=2))


if __name__ == '__main__':
    main()
