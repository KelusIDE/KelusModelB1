import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { Worker } from 'node:worker_threads';
import { modelBytes, validateManifest } from '../src/model-validation.mjs';
const manifest = JSON.parse(await readFile(new URL('../public/models/seal-3/b1.json', import.meta.url)));
const otter = JSON.parse(await readFile(new URL('../public/models/otter-4-12/b1.json', import.meta.url)));
test('packaged model is valid; truncated weights and corrupt tensor layouts fail', () => {
  validateManifest(manifest, manifest.parameters * 4);
  assert.throws(() => validateManifest(manifest, manifest.parameters * 4 - 4));
  assert.throws(() => validateManifest({ ...manifest, tensors: [] }, manifest.parameters * 4));
  assert.throws(() => validateManifest({ ...manifest, tensors: [{ offset: -1, shape: [manifest.parameters] }] }, manifest.parameters * 4));
});
test('real worker loads bundled weights, streams a reply, and answers the next turn with no network', { timeout: 60000 }, async t => {
  const worker = new Worker(new URL('./worker-harness.mjs', import.meta.url));
  t.after(() => worker.terminate());
  const events = [];
  worker.on('message', data => events.push(data));
  const until = predicate => new Promise((resolve, reject) => {
    const timer = setTimeout(() => done(new Error('Worker timed out')), 45000);
    function done(error, event) { clearTimeout(timer); worker.off('message', receive); worker.off('error', failed); error ? reject(error) : resolve(event); }
    function receive(event) { if (event.type === 'error') done(new Error(event.error)); else if (predicate(event)) done(null, event); }
    function failed(error) { done(error); }
    worker.on('message', receive); worker.on('error', failed);
  });
  let next = until(e => e.type === 'ready');
  worker.postMessage({ type: 'load', base: 'https://kani.test/models/seal-3/' });
  const ready = await next;
  assert.equal(ready.model.name, manifest.name); assert.equal(ready.model.sha256, manifest.weights_sha256);

  assert.equal(ready.model.device, 'cpu'); assert.match(ready.model.note, /CPU/);
  next = until(e => e.type === 'done' && e.id === 'greeting');
  worker.postMessage({ type: 'generate', id: 'greeting', messages: [{ role: 'user', content: 'hi' }] });
  await next;
  const reply = events.filter(e => e.type === 'token' && e.id === 'greeting').at(-1).text;
  assert.ok(reply.length > 0);
  next = until(e => e.type === 'done' && e.id === 'calculator');
  worker.postMessage({ type: 'generate', id: 'calculator', messages: [{ role: 'user', content: 'hi' }, { role: 'assistant', content: reply }, { role: 'user', content: 'what is 12 * 7?' }] });
  const result = await next;
  assert.equal(result.source, 'local_calculator');
  assert.match(events.filter(e => e.id === 'calculator' && e.type === 'token').at(-1).text, /84/);


  next = until(e => e.type === 'done' && e.id === 'fact');
  worker.postMessage({ type: 'generate', id: 'fact', search: true, messages: [{ role: 'user', content: 'when was Albert Einstein born?' }] });
  const fact = await next;
  const shown = events.filter(e => e.id === 'fact' && e.type === 'token').map(e => e.text);
  assert.equal(fact.source, 'model_web');
  assert.ok(shown.every(text => !text.startsWith('SEARCH:')), shown[0]);
  assert.match(shown.at(-1), /^Searching the web for “[^”]+”…[\s\S]*Sources:\n\[1\] Albert Einstein — https:\/\/en\.wikipedia\.org\/wiki\/Albert_Einstein$/);


  next = until(e => e.type === 'done' && e.id === 'offline');
  worker.postMessage({ type: 'generate', id: 'offline', search: false, messages: [{ role: 'user', content: 'when was Albert Einstein born?' }] });
  assert.equal((await next).source, 'search_unavailable');
});

test('Kani Otter 4.12 is packaged as int8: about a quarter of float32, and a damaged layout is refused', () => {
  assert.equal(otter.quantization, 'int8-per-channel');
  validateManifest(otter, modelBytes(otter));
  assert.ok(modelBytes(otter) < otter.parameters * 4 / 3.5, `${modelBytes(otter)} bytes`);
  assert.throws(() => validateManifest(otter, modelBytes(otter) - 4));
  const moved = otter.tensors.map(t => t.dtype === 'int8' && t.name === 'block0.q' ? { ...t, byte_offset: t.byte_offset + 4 } : t);
  assert.throws(() => validateManifest({ ...otter, tensors: moved }, modelBytes(otter)));
  assert.throws(() => validateManifest({ ...otter, quantization: 'int4' }, modelBytes(otter)));
});
test('the real worker loads Kani Otter 4.12, answers, and finds places near you', { timeout: 180000 }, async t => {
  const worker = new Worker(new URL('./worker-harness.mjs', import.meta.url));
  t.after(() => worker.terminate());
  const events = [];
  const until = predicate => new Promise((resolve, reject) => {
    worker.on('message', event => { events.push(event); if (event.type === 'error') reject(new Error(event.error)); else if (predicate(event)) resolve(event); });
    worker.on('error', reject);
  });
  const ready = until(e => e.type === 'ready');
  worker.postMessage({ type: 'load', base: 'https://kani.test/models/otter-4-12/' });
  assert.equal((await ready).model.name, 'Kani Otter 4.12');
  const done = until(e => e.type === 'done' && e.id === 'hello');
  worker.postMessage({ type: 'generate', id: 'hello', messages: [{ role: 'user', content: 'Write a Python function that adds two numbers.' }] });
  await done;
  assert.match(events.filter(e => e.type === 'token' && e.id === 'hello').at(-1).text, /def \w+\(/);



  const located = until(e => e.type === 'locate' && e.id === 'pizza');
  const placed = until(e => e.type === 'done' && e.id === 'pizza');
  worker.postMessage({ type: 'generate', id: 'pizza', search: true, messages: [{ role: 'user', content: 'any good pizza places near me?' }] });
  await located;
  worker.postMessage({ type: 'location', near: { lat: 40.748441, lng: -73.985664 } });
  const finished = await placed;
  assert.equal(finished.source, 'places');
  const request = events.find(e => e.type === 'places-request');
  assert.match(request.body.query, /pizza/i);
  assert.deepEqual(request.body.near, { lat: 40.75, lng: -73.99 });
  const shown = events.filter(e => e.type === 'token' && e.id === 'pizza').map(e => e.text);
  assert.ok(shown.every(text => !text.startsWith('PLACES:')), shown[0]);
  assert.match(shown.at(-1), /1\. Joe’s Pizza · ★ 4\.6 \(18,234\) · \$ · Open now[\s\S]*Results from Google Maps$/);
  assert.match(finished.saved, /^I looked up .*pizza.* near you\. Search again in Maps: https:\/\/www\.google\.com\/maps\/search\/\?api=1&query=/);


  const off = until(e => e.type === 'done' && e.id === 'pizza-off');
  worker.postMessage({ type: 'generate', id: 'pizza-off', search: false, messages: [{ role: 'user', content: 'any good pizza places near me?' }] });
  assert.equal((await off).source, 'search_unavailable');
});
