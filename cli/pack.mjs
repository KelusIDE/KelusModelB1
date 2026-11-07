#!/usr/bin/env node
// Builds what the installer downloads: the CLI, the site modules it imports, and each model's manifest. Weights stay out;
// model-host/download.mjs fetches them. Writes dist/kani-cli-VERSION.tar.gz and dist/install.sh with the sum pinned.
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { realpathSync } from 'node:fs';
import { copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');

/** Every local file reachable from the entry points through import statements. */
export async function importClosure(entries, root = ROOT) {
  const found = new Set(), pending = [...entries];
  while (pending.length) {
    const file = pending.pop();
    if (found.has(file)) continue;
    found.add(file);
    if (!/\.m?js$/.test(file)) continue;
    const source = await readFile(path.join(root, file), 'utf8');
    for (const match of source.matchAll(/(?:from\s+|import\s*\(\s*|^import\s+)['"](\.{1,2}\/[^'"]+)['"]/gm))
      pending.push(path.relative(root, path.resolve(root, path.dirname(file), match[1])));
  }
  return [...found].sort();
}

export async function pack(out = path.join(HERE, 'dist')) {
  const { version } = JSON.parse(await readFile(path.join(HERE, 'package.json'), 'utf8'));
  const catalog = JSON.parse(await readFile(path.join(ROOT, 'model-host/catalog.json'), 'utf8'));
  const exports = new Set(Object.values(catalog.models).map(model => `site/${model.directory}`));
  const builds = JSON.parse(await readFile(path.join(ROOT, 'kani-builds.json'), 'utf8')).filter(build => exports.has(build.export));
  const files = [...await importClosure(['cli/kani.mjs', 'cli/cide.mjs', 'model-host/download.mjs']), 'cli/package.json', 'model-host/catalog.json',
    'LICENSE', 'NOTICE', ...[...exports].map(directory => `${directory}/b1.json`)];
  const stage = await mkdtemp(path.join(os.tmpdir(), 'kani-cli-'));
  try {
    for (const file of new Set(files)) {
      await mkdir(path.dirname(path.join(stage, file)), { recursive: true });
      await copyFile(path.join(ROOT, file), path.join(stage, file));
    }
    await writeFile(path.join(stage, 'kani-builds.json'), JSON.stringify(builds, null, 2) + '\n');
    // Without this, Node before 22.7 would not read the site's .js files as modules.
    await writeFile(path.join(stage, 'package.json'), '{ "private": true, "type": "module" }\n');
    await mkdir(out, { recursive: true });
    const archive = path.join(out, `kani-cli-${version}.tar.gz`);
    execFileSync('tar', ['--no-xattrs', '--no-mac-metadata', '--format', 'ustar', '-czf', archive, '-C', stage, '.'].filter(arg => process.platform === 'darwin' || !arg.startsWith('--no-')),
      { env: { ...process.env, COPYFILE_DISABLE: '1' } });
    const sha256 = createHash('sha256').update(await readFile(archive)).digest('hex');
    const script = (await readFile(path.join(HERE, 'install.sh'), 'utf8')).replaceAll('@VERSION@', version).replaceAll('@SHA256@', sha256);
    await writeFile(path.join(out, 'install.sh'), script);
    return { archive, script: path.join(out, 'install.sh'), sha256, version, files: files.length + 2 };
  } finally { await rm(stage, { recursive: true, force: true }); }
}

if (process.argv[1] && realpathSync(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const made = await pack(process.argv[2] && path.resolve(process.argv[2]));
  console.log(`${made.archive}\n${made.script}\nSHA-256 ${made.sha256}`);
}
