"""Stage the published APK's web assets verbatim for the matching iOS build."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--apk', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
if a.out.exists():
    raise SystemExit('Choose a new staging directory; existing assets are preserved.')
entries = {}
with zipfile.ZipFile(a.apk) as archive:
    for name in archive.namelist():
        if not name.startswith('assets/public/') or name.endswith('/'):
            continue
        relative = PurePosixPath(name.removeprefix('assets/public/'))
        if relative.is_absolute() or '..' in relative.parts or not relative.parts:
            raise SystemExit('Unsafe APK asset path')
        entries[relative] = archive.read(name)
if PurePosixPath('index.html') not in entries:
    raise SystemExit('APK contains no web entry point')
for relative, data in entries.items():
    target = a.out.joinpath(*relative.parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
receipt = {'apk_sha256': hashlib.sha256(a.apk.read_bytes()).hexdigest(),
           'assets': {str(name): hashlib.sha256(data).hexdigest() for name, data in entries.items()}}
a.out.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(f'Staged {len(entries)} unchanged APK web assets into {a.out}')
