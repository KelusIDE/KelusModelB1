


import { readFile, readdir, mkdir, copyFile, rename, rm, writeFile } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { modelBytes, validateManifest } from '../src/model-validation.mjs';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');


const DEFAULTS = { 'otter-4-12': '../site/models/otter-4-12', 'seal-3': '../site/model' };
const jobs = process.argv[2] ? { [process.argv[2]]: process.argv[3] } : DEFAULTS;
const packaged = {};
for (const [id, from] of Object.entries(jobs)) {
  if (!/^[a-z0-9-]+$/.test(id) || !from) throw new Error('Usage: package-model.mjs [<id> <export dir>]');
  const source = resolve(root, from), destination = resolve(root, 'public/models', id);
  if (source === destination) throw new Error('Choose a source export outside public/models.');
  const manifest = JSON.parse(await readFile(resolve(source, 'b1.json'), 'utf8'));
  const weights = await readFile(resolve(source, 'b1.bin'));
  validateManifest(manifest, weights.byteLength);
  if (createHash('sha256').update(weights).digest('hex') !== manifest.weights_sha256) throw new Error(`${id}: model checksum mismatch.`);
  const stage = destination + '.staging';
  await rm(stage, { recursive: true, force: true });
  await mkdir(stage, { recursive: true });
  await copyFile(resolve(source, 'b1.json'), resolve(stage, 'b1.json'));
  await copyFile(resolve(source, 'b1.bin'), resolve(stage, 'b1.bin'));
  await rm(destination, { recursive: true, force: true });
  await rename(stage, destination);
  packaged[id] = { name: manifest.name, parameters: manifest.parameters, bytes: modelBytes(manifest) };
  console.log(`Packaged ${id}: ${manifest.name} · ${manifest.parameters.toLocaleString()} parameters · ${(modelBytes(manifest) / 1e6).toFixed(0)} MB · SHA256 ${manifest.weights_sha256}`);
}


const index = resolve(root, 'public/models/index.json');
const existing = process.argv[2] ? await readFile(index, 'utf8').then(JSON.parse, () => ({})) : {};
if (!process.argv[2]) {
  for (const entry of await readdir(resolve(root, 'public/models'), { withFileTypes: true })) {
    if (entry.isDirectory() && !(entry.name in packaged)) {
      await rm(resolve(root, 'public/models', entry.name), { recursive: true, force: true });
      console.log(`Removed ${entry.name}: no longer offered`);
    }
  }
}
await writeFile(index, JSON.stringify({ ...existing, ...packaged }, null, 1) + '\n');
