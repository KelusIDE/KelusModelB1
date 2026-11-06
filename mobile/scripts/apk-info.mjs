import { execFileSync } from 'node:child_process';
import { readdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import { homedir } from 'node:os';

export async function inspectApk(file) {
  const sdk = process.env.ANDROID_HOME || process.env.ANDROID_SDK_ROOT || resolve(homedir(), 'Library/Android/sdk');
  const versions = (await readdir(resolve(sdk, 'build-tools'))).filter(v => /^\d+\.\d+\.\d+$/.test(v))
    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  if (!versions.length) throw new Error('Install Android SDK Build Tools first.');
  const tools = resolve(sdk, 'build-tools', versions.at(-1));
  const env = { ...process.env };
  if (!env.JAVA_HOME && process.platform === 'darwin') env.JAVA_HOME = execFileSync('/usr/libexec/java_home', ['-v', '21'], { encoding: 'utf8' }).trim();
  const badging = execFileSync(resolve(tools, 'aapt'), ['dump', 'badging', file], { encoding: 'utf8', env });
  const metadata = /^package: name='([^']+)' versionCode='(\d+)' versionName='([^']+)'/m.exec(badging);
  if (!metadata) throw new Error('Could not read the actual APK version.');

  const certificates = execFileSync(resolve(tools, process.platform === 'win32' ? 'apksigner.bat' : 'apksigner'), ['verify', '--print-certs', file], { encoding: 'utf8', env });
  const signers = [...certificates.matchAll(/Signer #\d+ certificate SHA-256 digest: ([a-f0-9]{64})/g)].map(match => match[1]);
  if (!signers.length) throw new Error('The APK has no verified signing certificate.');
  return { packageName: metadata[1], versionCode: Number(metadata[2]), versionName: metadata[3], signers };
}

export function validateReleaseApk(info, version, versionCode, certificate) {
  if (info.packageName !== certificate.packageName) throw new Error('The APK has the wrong application ID.');
  if (info.versionName !== version || info.versionCode !== versionCode) throw new Error('The APK version does not match the release. Build it again.');
  if (info.signers.length !== 1 || info.signers[0] !== certificate.sha256) throw new Error('The APK is signed with a different key. Existing Kani installations cannot update to it.');
}
