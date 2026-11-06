import { spawnSync } from 'node:child_process';
import { mkdir, copyFile, access, readFile } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const env = { ...process.env };
if (!env.JAVA_HOME && process.platform === 'darwin') {
  const java = spawnSync('/usr/libexec/java_home', ['-v', '21'], { encoding: 'utf8' });
  if (java.status === 0) env.JAVA_HOME = java.stdout.trim();
}
if (!env.ANDROID_HOME && !env.ANDROID_SDK_ROOT && process.platform === 'darwin') {
  const sdk = resolve(env.HOME, 'Library/Android/sdk');
  await access(sdk); env.ANDROID_HOME = sdk;
}
const build = spawnSync(process.platform === 'win32' ? 'gradlew.bat' : './gradlew',
  ['assembleDebug', '--no-daemon', '--max-workers=2'], { cwd: resolve(root, 'android'), env, stdio: 'inherit', shell: process.platform === 'win32' });
if (build.error) throw build.error;
if (build.status !== 0) process.exit(build.status || 1);
await mkdir(resolve(root, 'artifacts'), { recursive: true });
const { version } = JSON.parse(await readFile(resolve(root, 'package.json'), 'utf8'));
const target = resolve(root, `artifacts/kani-${version}-debug.apk`);
await copyFile(resolve(root, 'android/app/build/outputs/apk/debug/app-debug.apk'), target);
console.log(`Android test APK: ${target}`);
