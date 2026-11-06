

import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, rmSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { extname, join, normalize } from 'node:path';

const CHROME = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const root = new URL('../../', import.meta.url).pathname;
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));


const page = `(async () => {
  const b1 = await import('/site/b1.js'), { gpuModel } = await import('/site/gpu.js');
  await b1.enableKernels();
  const results = {};
  for (const id of ['seal-3', 'otter-4-12']) {
    const cpu = await b1.loadModel('/mobile/public/models/' + id + '/');
    const gpu = await gpuModel(cpu);
    if (!gpu) return null;
    const replies = [];
    for (const text of ['hi', 'Write a Python function that checks if a number is prime.', 'what is a variable?']) {
      const messages = [{ role: 'user', content: text }];
      let fast = ''; for await (const piece of gpu.generate(messages, { maxTokens: 80 })) fast += piece;
      replies.push([fast, [...cpu.generate(messages, { maxTokens: 80 })].join('')]);
    }
    // Two conversations in turn: the GPU's cache follows the one it is given.
    const a = [257, 259, ...cpu.tokenizer.encode('hello there'), 260], b = [257, 259, ...cpu.tokenizer.encode('write a loop'), 260];
    const ca = {}, cb = {};
    await gpu.nextLogits(a, ca); await gpu.nextLogits(b, cb);
    const again = await gpu.nextLogits([...a, 300], ca), expected = cpu.nextLogits([...a, 300], {});
    let worst = 0; for (let i = 0; i < expected.length; i++) worst = Math.max(worst, Math.abs(expected[i] - again[i]));
    results[id] = { name: gpu.name, replies, switched: worst };
    gpu.destroy();
  }
  return results;
})()`;

test('the GPU engine passes its check and writes the CPU engine\'s replies', { timeout: 240000 }, async t => {
  if (!existsSync(CHROME)) return t.skip('Chrome is not installed');
  const server = createServer(async (req, res) => {
    const file = join(root, normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname)));
    const types = { '.js': 'text/javascript', '.json': 'application/json' };
    try { const body = await readFile(file.startsWith(root) && !file.endsWith('/') ? file : join(root, 'mobile/index.html')); res.writeHead(200, { 'Content-Type': types[extname(file)] || 'text/html' }); res.end(body); }
    catch { res.writeHead(404); res.end(); }
  }).listen(0, '127.0.0.1');
  await new Promise(resolve => server.once('listening', resolve));
  const profile = mkdtempSync(join(tmpdir(), 'kani-gpu-test-')), port = 9400 + Math.floor(Math.random() * 400);
  const chrome = spawn(CHROME, ['--headless=new', '--enable-unsafe-webgpu', `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, 'about:blank'], { stdio: 'ignore' });
  t.after(() => { chrome.kill(); server.close(); setTimeout(() => rmSync(profile, { recursive: true, force: true }), 1000); });
  let target;
  for (let i = 0; i < 100 && !target; i++) {
    await sleep(200);
    target = await fetch(`http://127.0.0.1:${port}/json/new?http://127.0.0.1:${server.address().port}/`, { method: 'PUT' }).then(r => r.json()).catch(() => null);
  }
  assert.ok(target, 'Chrome started');
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise(resolve => ws.addEventListener('open', resolve));
  const pending = new Map();
  ws.addEventListener('message', event => { const message = JSON.parse(event.data); pending.get(message.id)?.(message); });
  let id = 0;
  const send = (method, params) => new Promise(resolve => { pending.set(++id, resolve); ws.send(JSON.stringify({ id, method, params })); });
  await sleep(1000);
  const reply = await send('Runtime.evaluate', { expression: page, awaitPromise: true, returnByValue: true });
  ws.close();
  assert.equal(reply.result.exceptionDetails, undefined, reply.result.exceptionDetails?.exception?.description);
  const results = reply.result.result.value;
  if (!results) return t.skip('Chrome found no GPU adapter here');
  for (const [model, { replies, switched }] of Object.entries(results)) {
    for (const [gpu, cpu] of replies) assert.equal(gpu, cpu, model);
    assert.ok(switched < 1e-3, `${model}: logits after switching conversations differ by ${switched}`);
  }
});
