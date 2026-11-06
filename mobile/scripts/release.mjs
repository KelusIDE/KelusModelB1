

import { createHash } from 'node:crypto';
import { copyFile, readFile, writeFile } from 'node:fs/promises';
import { execFileSync, spawnSync } from 'node:child_process';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { REPOSITORY, MANIFEST_URL, parseUpdate } from '../src/update-manifest.mjs';
import { inspectApk, validateReleaseApk } from './apk-info.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const repo = 'KelusIDE/KelusModelB1';
const publish = process.argv.includes('--publish');
const notes = process.argv.slice(2).filter(arg => arg !== '--publish');
if (notes.some(n => n.startsWith('--'))) throw new Error('Unknown release option. Use --publish or supply release-note text.');
const { version } = JSON.parse(await readFile(resolve(root, 'package.json'), 'utf8'));
const gradle = await readFile(resolve(root, 'android/app/build.gradle'), 'utf8');
const versionCode = Number(/versionCode (\d+)/.exec(gradle)?.[1]), versionName = /versionName "([^"]+)"/.exec(gradle)?.[1];
if (versionName !== version) throw new Error('package.json and build.gradle versions must match.');
const tag = `v${version}`;
const gh = args => execFileSync('gh', [...args, '--repo', repo], { encoding: 'utf8' });
let existing;
if (publish) {

  const releases = JSON.parse(gh(['release', 'list', '--limit', '100', '--json', 'tagName,isDraft']));
  existing = releases.find(release => release.tagName === tag);
  if (existing && !existing.isDraft) throw new Error(`${tag} is already published. Increase the version; never overwrite an installed release.`);
  const latestResponse = await fetch(MANIFEST_URL, { signal: AbortSignal.timeout(30000), cache: 'no-store' });
  if (!latestResponse.ok) throw new Error(`Cannot verify the current update channel (HTTP ${latestResponse.status}).`);
  const latest = parseUpdate(await latestResponse.text());
  if (!latest || latest.versionCode >= versionCode) throw new Error('The release must be newer than the current public update manifest.');
}
for (const command of ['test', 'apk']) {
  const result = spawnSync(process.platform === 'win32' ? 'npm.cmd' : 'npm', ['run', command], { cwd: root, stdio: 'inherit', shell: process.platform === 'win32' });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status || 1);
}
const built = resolve(root, `artifacts/kani-${version}-debug.apk`), apk = resolve(root, `artifacts/kani-${version}.apk`);
const info = await inspectApk(built);
const certificate = JSON.parse(await readFile(resolve(root, 'signing-certificate.json'), 'utf8'));
validateReleaseApk(info, version, versionCode, certificate);
const bytes = await readFile(built);
const manifest = { versionCode, versionName: version, apk: `${REPOSITORY}download/${tag}/kani-${version}.apk`,
  sha256: createHash('sha256').update(bytes).digest('hex'), size: bytes.length, notes: notes.length ? notes : ['Improvements and fixes'] };
if (!parseUpdate(JSON.stringify(manifest))) throw new Error('The app would not accept this update manifest.');
await copyFile(built, apk);
const manifestPath = resolve(root, 'artifacts/kani-update.json'), notesPath = resolve(root, `artifacts/release-${version}.md`);
await writeFile(manifestPath, JSON.stringify(manifest, null, 2) + '\n');
await writeFile(notesPath, [`Kani ${version} for Android (build ${versionCode}).`, '', ...manifest.notes.map(note => `- ${note}`), '',
  'Install over your existing Kani app; do not uninstall it. App ID and signing certificate match Kani 0.5.0, preserving app data.',
  'Kani 0.5.4 and older cannot download updates in the app (their downloads stop with "That update information is invalid"), so install this version by hand once: download the APK below on your phone, open it, and tap Update. From 0.5.5 on, Kani offers and installs updates in the app. Android still asks you to confirm installation.',
  '', 'This is a sideload/test build signed with the existing Android debug certificate, not a Play Store release.',
  `APK SHA-256: ${manifest.sha256}`, '', 'Built from the local mobile working tree; the repository tag alone is not a snapshot of its uncommitted source.'].join('\n') + '\n');
console.log(`Prepared ${apk}\nCertificate SHA-256: ${info.signers[0]}\nAPK SHA-256: ${manifest.sha256}`);
if (!publish) {
  console.log('No remote release changed. Use npm run release:publish -- "Release note" to build and publish.');
} else {
  if (!existing) {
    gh(['release', 'create', tag, apk, manifestPath, '--draft', '--latest=false', '--target', 'main', '--title', `Kani ${version}`, '--notes-file', notesPath]);
  }
  const staged = JSON.parse(gh(['release', 'view', tag, '--json', 'isDraft,assets']));
  if (!staged.isDraft) throw new Error('Expected an unpublished draft.');

  for (const [name, hash, size] of [[`kani-${version}.apk`, manifest.sha256, manifest.size],
    ['kani-update.json', createHash('sha256').update(await readFile(manifestPath)).digest('hex'), (await readFile(manifestPath)).length]]) {
    const asset = staged.assets.find(asset => asset.name === name);
    if (!asset || asset.state !== 'uploaded' || asset.size !== size || asset.digest !== `sha256:${hash}`) {
      throw new Error(`Draft asset ${name} does not match the verified build. Draft left unpublished; inspect it before retrying.`);
    }
  }
  gh(['release', 'edit', tag, '--draft=false', '--latest', '--notes-file', notesPath]);
  const response = await fetch(MANIFEST_URL, { signal: AbortSignal.timeout(30000), cache: 'no-store' });
  const publicManifest = response.ok ? parseUpdate(await response.text()) : null;
  if (!publicManifest || publicManifest.sha256 !== manifest.sha256 || publicManifest.versionCode !== versionCode) {
    throw new Error('Release published, but the public latest manifest could not yet be verified. Check GitHub before retrying.');
  }
  console.log(`Published and verified ${REPOSITORY}tag/${tag}`);
}
