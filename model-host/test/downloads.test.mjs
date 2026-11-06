import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { serve } from '../index.ts';
import catalog from '../catalog.json' with { type: 'json' };
import { loadWeights } from '../../site/b1.js';

const model = catalog.models['otter-4-12'];
const url = `https://models.example/v1/otter-4-12/${model.sha256}/b1.bin`;
const request = (options = {}) => new Request(url, options);

test('only known checksum-pinned model exports can be downloaded', async () => {
  const fetcher = () => { throw new Error('Must not access network'); };
  for (const path of ['/https://example.com', url.replace(model.sha256, '0'.repeat(64)), url + '?url=https://example.com']) {
    assert.equal((await serve(new Request(new URL(path, url)), fetcher)).status, 404);
  }
  assert.equal((await serve(request({ method: 'POST' }), fetcher)).status, 405);
  assert.equal((await serve(request({ method: 'HEAD' }), fetcher)).headers.get('Content-Length'), String(model.bytes));
  const preflight = await serve(request({ method: 'OPTIONS' }), fetcher);
  assert.equal(preflight.status, 204);
  assert.equal(preflight.headers.get('Access-Control-Allow-Origin'), '*');
});

test('streams GitHub bytes through a CORS response without forwarding credentials', async () => {
  const calls = [];
  const response = await serve(request({ headers: { Authorization: 'secret', Cookie: 'secret' } }), async (target, options) => {
    calls.push(target);
    assert.equal(options.redirect, 'manual');
    assert.equal(options.headers.Authorization, undefined);
    assert.equal(options.headers.Cookie, undefined);
    if (calls.length === 1) return new Response(null, { status: 302, headers: { Location: 'https://release-assets.githubusercontent.com/allowed' } });
    return new Response(new ReadableStream({ start(controller) { controller.enqueue(Uint8Array.of(1, 2, 3)); controller.close(); } }), {
      headers: { 'Content-Length': String(model.bytes) },
    });
  });
  assert.equal(calls[0], `https://github.com/${catalog.repository}/releases/download/${catalog.tag}/${model.asset}`);
  assert.equal(response.headers.get('X-Weights-Sha256'), model.sha256);
  assert.match(response.headers.get('Cache-Control'), /immutable/);
  assert.equal(response.headers.get('Access-Control-Allow-Origin'), '*');
  assert.deepEqual([...new Uint8Array(await response.arrayBuffer())], [1, 2, 3]);
});

test('blocks unrelated redirects, errors, wrong lengths and invalid byte ranges', async () => {
  assert.equal((await serve(request(), async () => new Response(null, { status: 302, headers: { Location: 'https://evil.example/file' } }))).status, 502);
  assert.equal((await serve(request(), async () => new Response('missing', { status: 404 }))).status, 502);
  assert.equal((await serve(request(), async () => new Response('short', { headers: { 'Content-Length': '5' } }))).status, 502);
  for (const range of ['bytes=9999999999-', 'bytes=20-10', 'bytes=0-1,4-5', 'garbage']) {
    assert.equal((await serve(request({ headers: { Range: range } }), () => { throw new Error('No network'); })).status, 416);
  }
  const partial = await serve(request({ headers: { Range: 'bytes=0-2' } }), async (_, options) => {
    assert.equal(options.headers.Range, 'bytes=0-2');
    return new Response(Uint8Array.of(1, 2, 3), { status: 206, headers: { 'Content-Length': '3', 'Content-Range': `bytes 0-2/${model.bytes}` } });
  });
  assert.equal(partial.status, 206);
  assert.equal(partial.headers.get('Content-Length'), '3');
});

test('conditional requests return 304 without downloading from GitHub', async () => {
  const response = await serve(request({ headers: { 'If-None-Match': `W/"${model.sha256}"` } }), () => { throw new Error('No network'); });
  assert.equal(response.status, 304);
});

test('verified browser cache avoids a second download and repairs damaged cache', async () => {
  const originalCache = globalThis.caches, originalLocation = globalThis.location;
  const stored = new Map();
  globalThis.location = new URL('https://kelus-b1.web.app');
  globalThis.caches = { open: async () => ({
    match: async key => stored.get(key)?.clone(),
    put: async (key, value) => stored.set(key, value),
    delete: async key => stored.delete(key),
  }) };
  const bytes = Uint8Array.of(1, 2, 3, 4), base = 'models/test/';
  const manifest = { bytes: 4, weights_sha256: createHash('sha256').update(bytes).digest('hex') };
  let downloads = 0;
  const fetcher = async () => { downloads++; return new Response(bytes); };
  try {
    await loadWeights(base, manifest, fetcher);
    await loadWeights(base, manifest, fetcher);
    assert.equal(downloads, 1);
    stored.set(base + 'b1.bin', new Response(Uint8Array.of(4, 3, 2, 1), { headers: { 'x-weights-sha256': manifest.weights_sha256 } }));
    await loadWeights(base, manifest, fetcher);
    assert.equal(downloads, 2);
    stored.clear();
    await assert.rejects(loadWeights(base, manifest, async () => new Response(Uint8Array.of(4, 3, 2, 1))), /verification/);
    assert.equal(stored.size, 0);
    await assert.rejects(loadWeights(base, manifest, async () => new Response('x')), /incomplete/);
  } finally { globalThis.caches = originalCache; globalThis.location = originalLocation; }
});
