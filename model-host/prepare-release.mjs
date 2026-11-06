import { createHash } from 'node:crypto';
import { copyFile, mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url)), root = resolve(here, '..');
const stage = process.argv[2];
if (!stage) throw new Error('Usage: node model-host/prepare-release.mjs <staging-directory>');
await mkdir(stage, { recursive: true });
const catalog = { repository: 'KelusIDE/KelusModelB1', tag: 'models-2026-11-05', models: {} };
const folders = ['model', ...(await readdir(join(root, 'site/models'))).sort().map(id => `models/${id}`)];
for (const directory of folders) {
  const id = directory === 'model' ? 'seal-3' : directory.split('/')[1];
  const manifestPath = join(root, 'site', directory, 'b1.json');
  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  const bytes = await readFile(join(root, 'site', directory, 'b1.bin'));
  const sha256 = createHash('sha256').update(bytes).digest('hex');
  if (sha256 !== manifest.weights_sha256 || (manifest.bytes && bytes.length !== manifest.bytes)) {
    throw new Error(`Model export failed verification: ${id}`);
  }
  const asset = `${id}.${sha256.slice(0, 16)}.bin`;
  await writeFile(join(stage, asset), bytes);
  await copyFile(manifestPath, join(stage, `${id}.b1.json`));
  catalog.models[id] = { name: manifest.name, directory, asset, bytes: bytes.length, sha256 };
  console.log(`${id}: verified ${bytes.length} bytes`);
}
for (const name of ['LICENSE', 'NOTICE', 'LICENSING.md', 'MODEL_CARD.md']) await copyFile(join(root, name), join(stage, name));
const json = JSON.stringify(catalog, null, 2) + '\n';
await writeFile(join(here, 'catalog.json'), json);
await writeFile(join(stage, 'catalog.json'), json);
const sums = [];
for (const name of (await readdir(stage)).sort()) {
  if (name === 'SHA256SUMS') continue;
  sums.push(`${createHash('sha256').update(await readFile(join(stage, name))).digest('hex')}  ${name}`);
}
await writeFile(join(stage, 'SHA256SUMS'), sums.join('\n') + '\n');
console.log(`Prepared ${Object.keys(catalog.models).length} models in ${stage}`);
