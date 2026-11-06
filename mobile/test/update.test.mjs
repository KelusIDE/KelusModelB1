
import test from 'node:test';
import assert from 'node:assert/strict';
import { MANIFEST_URL, isNewer, megabytes, parseUpdate, MAX_APK_BYTES } from '../src/update-manifest.mjs';

const good = { versionCode: 7, versionName: '0.5.2', apk: 'https://github.com/KelusIDE/KelusModelB1/releases/download/v0.5.2/kani-0.5.2.apk',
  sha256: 'a'.repeat(64), size: 170_526_381, notes: ['Faster replies', 3, 'Fixes'] };

test('a release manifest is read, checked and compared with the installed build', () => {
  assert.equal(MANIFEST_URL, 'https://github.com/KelusIDE/KelusModelB1/releases/latest/download/kani-update.json');
  const update = parseUpdate(JSON.stringify(good));
  assert.deepEqual(update.notes, ['Faster replies', 'Fixes']);
  assert.ok(isNewer(update, '6'));
  assert.ok(!isNewer(update, '7'));
  assert.ok(!isNewer(update, '8'));
  assert.ok(!isNewer(null, '1'));
  assert.equal(megabytes(update.size), '171 MB');
});

test('anything else is ignored: another site, a bad checksum, a broken file', () => {
  for (const bad of [
    { ...good, apk: 'https://evil.example/kani.apk' },
    { ...good, apk: 'https://github.com/someone-else/Kani/releases/download/v1/kani.apk' },
    { ...good, apk: 'http://github.com/KelusIDE/KelusModelB1/releases/download/v0.5.2/kani.apk' },
    { ...good, sha256: 'xyz' }, { ...good, versionCode: 0 }, { ...good, versionCode: 1.5 }, { ...good, size: -1 },
    { ...good, size: MAX_APK_BYTES + 1 }, { ...good, size: 1.2 }, { ...good, versionCode: 2147483648 },
    { ...good, versionName: '' }, { ...good, versionName: '<script>' },
    { ...good, apk: good.apk + '?redirect=https://evil.example' }, { ...good, apk: good.apk + '#fragment' },
    { ...good, apk: 'https://github.com/KelusIDE/KelusModelB1/releases/download/../../../other/kani.apk' },
    { ...good, apk: good.apk.replace('v0.5.2/', '%2e%2e/') },
  ]) assert.equal(parseUpdate(JSON.stringify(bad)), null, JSON.stringify(bad).slice(0, 80));
  assert.equal(parseUpdate('<html>Not found</html>'), null);
  assert.equal(parseUpdate('null'), null);
});
test('unknown or malformed installed build never authorizes an update', () => {
  for (const build of ['', undefined, null, 'bad', '0', '-1', '1.5']) assert.equal(isNewer(good, build), false);
});
