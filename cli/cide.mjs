import { readFile } from 'node:fs/promises';
import { Model } from '../site/b1.js';

let cached;
export async function loadCide() {
  if (cached) return cached;
  try {
    const builds = JSON.parse(await readFile(new URL('../kani-builds.json', import.meta.url), 'utf8'));
    const build = builds.find(item => item.id === 'cide');
    if (!build?.export) throw new Error('No Cide coding build is configured.');
    const directory = new URL(build.export + '/', new URL('../', import.meta.url));
    const manifest = JSON.parse(await readFile(new URL('b1.json', directory), 'utf8'));
    if (manifest.specialist !== 'coding') throw new Error('The installed checkpoint is not a Cide coding release.');
    const bytes = await readFile(new URL('b1.bin', directory));
    cached = new Model(manifest, bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
    return cached;
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
    throw new Error('The configured Cide build is not exported. Run node model-host/download.mjs cide from the repository root. See kani-builds.json and CIDE.md.');
  }
}
