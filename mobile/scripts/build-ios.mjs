import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { cpSync, mkdtempSync, readFileSync, rmSync, mkdirSync } from 'node:fs';
import { resolve } from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
function run(command, args, env = {}) {
  const result = spawnSync(command, args, { cwd: root, stdio: 'inherit', env: { ...process.env, ...env } });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}
const args = process.argv.slice(2);
const ipa = args.includes('--ipa');
let apk;
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--ipa') continue;
  if (args[i] === '--apk' && args[i + 1]) { apk = resolve(args[++i]); continue; }
  throw new Error('Usage: npm run ios:build -- [--ipa] [--apk FILE]');
}

// Check before building/copying web assets or downloading native dependencies.
const xcode = spawnSync('xcodebuild', ['-version'], { encoding: 'utf8' });
const major = Number(xcode.stdout?.match(/^Xcode (\d+)/m)?.[1]);
if (process.platform !== 'darwin' || xcode.status !== 0 || major < 26 || !major) {
  console.error('Kani iOS requires full Xcode 26+ with the iOS SDK. Command Line Tools alone cannot build it.');
  console.error('Install and launch Xcode, then select it in Xcode → Settings → Locations → Command Line Tools.');
  process.exit(1);
}

run('bundle', ['check']);
if (apk) {
  mkdirSync(resolve(root, 'artifacts'), { recursive: true });
  const staging = mkdtempSync(resolve(root, 'artifacts/ios-apk-'));
  // The extractor refuses to overwrite an existing directory.
  rmSync(staging, { recursive: true });
  run('python3', ['scripts/ios-apk-assets.py', '--apk', apk, '--out', staging]);
  run('bundle', ['exec', 'npx', 'cap', 'sync', 'ios'], { KANI_IOS_WEB_DIR: staging });
} else {
  run('npm', ['run', 'sync:ios']);
}
const configuration = ipa ? 'Release' : 'Debug';
run('xcodebuild', [
  '-workspace', 'ios/App/App.xcworkspace', '-scheme', 'App',
  '-configuration', configuration, '-sdk', 'iphoneos', '-destination', 'generic/platform=iOS',
  '-derivedDataPath', 'artifacts/ios-derived', 'CODE_SIGNING_ALLOWED=NO', 'CODE_SIGNING_REQUIRED=NO', 'build',
]);
const app = resolve(root, `artifacts/ios-derived/Build/Products/${configuration}-iphoneos/App.app`);
if (ipa) {
  const { version } = JSON.parse(readFileSync(resolve(root, 'package.json'), 'utf8'));
  const output = resolve(root, `artifacts/kani-${version}-unsigned.ipa`);
  const staging = mkdtempSync(resolve(root, 'artifacts/ios-payload-'));
  try {
    mkdirSync(resolve(staging, 'Payload'));
    cpSync(app, resolve(staging, 'Payload/Kani.app'), { recursive: true, verbatimSymlinks: true });
    run('/usr/bin/ditto', ['-c', '-k', '--keepParent', resolve(staging, 'Payload'), output]);
    console.log(`Unsigned IPA for re-signing with a sideload tool: ${output}`);
  } finally { rmSync(staging, { recursive: true, force: true }); }
} else {
  console.log(`Unsigned device build: ${app}`);
}
