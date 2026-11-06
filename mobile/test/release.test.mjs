import test from 'node:test';
import assert from 'node:assert/strict';
import { validateReleaseApk } from '../scripts/apk-info.mjs';
const certificate = { packageName: 'com.kelus.kani', sha256: 'a'.repeat(64) };
const info = { packageName: certificate.packageName, versionCode: 8, versionName: '0.5.3', signers: [certificate.sha256] };
test('release validates the actual APK, rejecting stale version, wrong package, and wrong signing key', () => {
  assert.doesNotThrow(() => validateReleaseApk(info, '0.5.3', 8, certificate));
  for (const change of [{ versionCode: 7 }, { versionName: '0.5.2' }, { packageName: 'com.other.app' }, { signers: [] }, { signers: ['b'.repeat(64)] }]) {
    assert.throws(() => validateReleaseApk({ ...info, ...change }, '0.5.3', 8, certificate));
  }
});
