import test from 'node:test';
import assert from 'node:assert/strict';
import { createUpdateController } from '../src/update-controller.mjs';
const update = { versionCode: 8, versionName: '0.5.3', apk: 'https://github.com/KelusIDE/KelusModelB1/releases/download/v0.5.3/kani-0.5.3.apk', sha256: 'a'.repeat(64), size: 170000000, notes: ['Update fixes'] };
const flush = () => new Promise(resolve => setImmediate(resolve));
async function setup(t, overrides = {}, supported = true) {
  const calls = []; let allowed = false, progress;
  const native = {
    cleanup: async () => { calls.push('cleanup'); },
    fetchManifest: async () => { calls.push('manifest'); return { text: JSON.stringify(update) }; },
    addListener: async (_, callback) => { progress = callback; return { remove: async () => calls.push('remove') }; },
    download: async options => { calls.push(['download', options]); progress({ loaded: 85e6, total: 170e6 }); return { path: '/cache/updates/kani-8.apk' }; },
    install: async options => { calls.push(['install', options]); return { permission: !allowed, started: allowed }; },
    canInstall: async () => ({ allowed }), ...overrides,
  };
  const app = { getInfo: async () => ({ build: '7', version: '0.5.2' }), addListener: async () => ({ remove: async () => {} }) };
  const controller = createUpdateController(native, app, supported);
  t.after(controller.mount()); await flush();
  return { controller, calls, native, allow: () => { allowed = true; } };
}
test('startup checks after cleanup, offers update, never downloads without a tap', async t => {
  const { controller, calls } = await setup(t);
  assert.deepEqual(calls, ['cleanup', 'manifest']);
  assert.equal(controller.getSnapshot().phase, 'available');
});
test('denying install permission returns to app without reopening settings; retry reuses verified download', async t => {
  const { controller, calls, allow } = await setup(t);
  await controller.start(); assert.equal(controller.getSnapshot().phase, 'permission');
  await controller.resume(); assert.equal(controller.getSnapshot().phase, 'available');
  assert.equal(calls.filter(c => c[0] === 'install').length, 1);
  allow(); await controller.start(); assert.equal(controller.getSnapshot().phase, 'installing');
  assert.equal(calls.filter(c => c[0] === 'download').length, 1);
  await controller.resume(); assert.equal(controller.getSnapshot().phase, 'available');
  assert.equal(calls.filter(c => c[0] === 'install').length, 2);
});
test('granting permission opens installer and carries verification metadata again', async t => {
  const { controller, calls, allow } = await setup(t);
  await controller.start(); allow(); await controller.resume();
  assert.equal(controller.getSnapshot().phase, 'installing');
  const options = calls.filter(c => c[0] === 'install').at(-1)[1];
  assert.equal(options.sha256, update.sha256); assert.equal(options.size, update.size); assert.equal(options.versionCode, 8);
});
test('malformed manifest and network errors do not claim the installed app is current', async t => {
  const { controller, native } = await setup(t);
  native.fetchManifest = async () => ({ text: '<html>Unavailable</html>' });
  await controller.check(); assert.equal(controller.getSnapshot().phase, 'error'); assert.match(controller.getSnapshot().problem, /invalid/);
  native.fetchManifest = async () => { throw new Error('Offline'); };
  await controller.check(); assert.equal(controller.getSnapshot().phase, 'error');
});
test('listener failure and corrupt download are recoverable and never launch installer', async t => {
  const { controller, native, calls } = await setup(t, { addListener: async () => { throw new Error('Listener unavailable'); } });
  await controller.start(); assert.equal(controller.getSnapshot().phase, 'error');
  native.addListener = async () => ({ remove: async () => {} });
  native.download = async () => { throw new Error('Checksum mismatch'); };
  await controller.start(); assert.equal(controller.getSnapshot().phase, 'error');
  assert.equal(calls.filter(c => c[0] === 'install').length, 0);
});
test('rapid repeated taps start only one download', async t => {
  let finish;
  const { controller, calls } = await setup(t, { download: () => new Promise(resolve => { calls.push('download'); finish = resolve; }) });
  const first = controller.start(); await flush(); await controller.start();
  assert.equal(calls.filter(c => c === 'download').length, 1);
  finish({ path: '/cache/updates/kani-8.apk' }); await first;
});
test('iOS and browsers never call the Android updater', async t => {
  const { controller, calls } = await setup(t, {}, false);
  await controller.check(); await controller.start(); await controller.resume();
  assert.deepEqual(calls, []);
});
test('Later survives automatic checks; a manual check can show the offer again', async t => {
  const { controller } = await setup(t);
  controller.hide(); await controller.check(true); assert.equal(controller.getSnapshot().hidden, true);
  await controller.check(false); assert.equal(controller.getSnapshot().hidden, false);
});
