'use strict';
// Drives the real `kelus stdio` process with a tiny untrained checkpoint.
const assert = require('node:assert');
const { execFileSync } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const { Bridge, explain } = require('../bridge');

const ROOT = path.resolve(__dirname, '..', '..');
const PYTHON = process.env.KELUS_PYTHON || 'python3';
const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'kelus-bridge-'));
const checkpoint = path.join(temp, 'tiny.npz');
execFileSync(PYTHON, ['-c', `from kelus.model import Config, Model
Model(Config(context=64, width=8, heads=2, layers=1, seed=7)).save(${JSON.stringify(checkpoint)}, {"steps": 1})`], { cwd: ROOT });
test.after(() => fs.rmSync(temp, { recursive: true, force: true }));

function started(options = {}) {
  const bridge = new Bridge({ python: PYTHON, cwd: ROOT, checkpoint, ...options });
  const status = new Promise(resolve => bridge.on('status', s => s.state !== 'starting' && resolve(s)));
  bridge.start();
  return { bridge, status };
}

function reply(bridge, id) {
  const events = [];
  return new Promise(resolve => bridge.on('event', event => {
    if (event.id !== id) return;
    events.push(event);
    if (event.type === 'done' || event.type === 'error') resolve(events);
  }));
}

test('streams a reply and reports readiness', async () => {
  const { bridge, status } = started();
  const ready = await status;
  assert.strictEqual(ready.state, 'ready');
  assert.strictEqual(ready.info.checkpoint, 'tiny');
  const events = reply(bridge, 'a');
  assert.ok(bridge.chat('a', [{ role: 'user', content: 'hello!' }], { maxTokens: 8 }));
  const result = await events;
  assert.deepStrictEqual([result[0].type, result[result.length - 1].type], ['start', 'done']);
  assert.strictEqual(result[0].source, 'model');
  bridge.stop();
});

test('labels local helpers and rejects invalid conversations', async () => {
  const { bridge, status } = started();
  await status;
  const helper = reply(bridge, 'calc');
  bridge.chat('calc', [{ role: 'user', content: 'what is 2 + 2?' }]);
  const events = await helper;
  assert.strictEqual(events[0].source, 'local_calculator');
  assert.strictEqual(events[1].text, '2 + 2 = 4');
  const invalid = reply(bridge, 'bad');
  bridge.chat('bad', [{ role: 'assistant', content: 'hi' }]);
  assert.strictEqual((await invalid)[0].type, 'error');
  bridge.stop();
});

test('cancels a reply in progress', async () => {
  const { bridge, status } = started();
  await status;
  const events = reply(bridge, 'c');
  bridge.chat('c', [{ role: 'user', content: 'write a python function' }], { maxTokens: 50 });
  bridge.cancel('c');
  const done = (await events).pop();
  assert.strictEqual(done.type, 'done');
  assert.strictEqual(done.cancelled, true);
  bridge.stop();
});

test('explains a missing interpreter or checkpoint', async () => {
  const missing = started({ python: '/nonexistent/python3' });
  assert.match((await missing.status).error, /Python was not found/);
  const absent = started({ checkpoint: path.join(temp, 'absent.npz') });
  assert.strictEqual((await absent.status).state, 'error');
  assert.match(explain("ModuleNotFoundError: No module named 'numpy'", 'python3'), /pip install numpy/);
});
