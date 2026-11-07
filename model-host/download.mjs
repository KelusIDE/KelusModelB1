import { createReadStream, realpathSync } from 'node:fs';
import { mkdir, open, rename, rm, stat } from 'node:fs/promises';
import { createHash, randomUUID } from 'node:crypto';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import catalog from './catalog.json' with { type: 'json' };

const ROOT = fileURLToPath(new URL('../', import.meta.url));
const ALIASES = { latest: 'otter-4-12', cide: 'cide-4-5' };

async function matches(file, model) {
  try { if ((await stat(file)).size !== model.bytes) return false; }
  catch (error) { if (error.code === 'ENOENT') return false; throw error; }
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(file)) hash.update(chunk);
  return hash.digest('hex') === model.sha256;
}

export async function downloadModel(id, { root = ROOT, fetcher = fetch, models = catalog.models, log = console.log } = {}) {
  id = ALIASES[id] || id;
  const model = Object.hasOwn(models, id) && models[id];
  if (!model) throw new Error(`Unknown model: ${id}. Choose ${Object.keys(models).join(', ')}.`);
  const destination = resolve(root, 'site', model.directory, 'b1.bin');
  if (await matches(destination, model)) { log(`${id}: already verified`); return destination; }
  await mkdir(dirname(destination), { recursive: true });
  const temporary = destination + '.' + randomUUID() + '.download';
  let file;
  try {
    const url = `https://github.com/${catalog.repository}/releases/download/${catalog.tag}/${model.asset}`;
    const response = await fetcher(url, { signal: AbortSignal.timeout(30 * 60 * 1000) });
    if (!response.ok || !response.body) throw new Error(`${id}: download failed (${response.status}).`);
    file = await open(temporary, 'wx');
    const hash = createHash('sha256');
    let bytes = 0;
    const reader = response.body.getReader();
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        bytes += value.byteLength;
        if (bytes > model.bytes) throw new Error(`${id}: unexpected download size.`);
        hash.update(value);
        let offset = 0;
        while (offset < value.byteLength) {
          const { bytesWritten } = await file.write(value, offset, value.byteLength - offset);
          if (!bytesWritten) throw new Error(`${id}: could not write model.`);
          offset += bytesWritten;
        }
      }
    } finally { await reader.cancel(); }
    if (bytes !== model.bytes || hash.digest('hex') !== model.sha256) throw new Error(`${id}: checksum verification failed.`);
    await file.close(); file = null;
    await rename(temporary, destination);
    log(`${id}: downloaded and verified (${bytes} bytes)`);
    return destination;
  } finally { await file?.close(); await rm(temporary, { force: true }); }
}

// Node resolves symlinks for import.meta.url but not argv, and macOS temp folders and ~/.kani/current are symlinks.
if (process.argv[1] && realpathSync(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  const ids = args.includes('--all') ? Object.keys(catalog.models) : args.length ? args : ['latest', 'cide'];
  try { for (const id of [...new Set(ids)]) await downloadModel(id); }
  catch (error) { console.error(error.message); process.exitCode = 1; }
}
