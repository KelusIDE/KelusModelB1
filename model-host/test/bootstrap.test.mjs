import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { downloadModel } from '../download.mjs';

const bytes = Buffer.from('model fixture');
const models = { fixture: { directory: 'models/fixture', asset: 'fixture.bin', bytes: bytes.length,
  sha256: createHash('sha256').update(bytes).digest('hex') } };

test('bootstrap verifies weights, reuses valid local copies and rejects unknown models', async () => {
  const root = await mkdtemp(join(tmpdir(), 'kani-bootstrap-'));
  let calls = 0;
  const options = { root, models, log: () => {}, fetcher: async url => {
    calls++; assert.match(url, /^https:\/\/github.com\/KelusIDE\/KelusModelB1\/releases\/download\/models-2026-11-05\//);
    return new Response(bytes);
  } };
  try {
    const file = await downloadModel('fixture', options);
    assert.deepEqual(await readFile(file), bytes);
    await downloadModel('fixture', options);
    assert.equal(calls, 1);
    await assert.rejects(downloadModel('../unknown', options), /Unknown model/);
    assert.equal(calls, 1);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('bad or interrupted downloads preserve an existing file and remove temporary data', async () => {
  const root = await mkdtemp(join(tmpdir(), 'kani-bootstrap-'));
  const directory = join(root, 'site/models/fixture');
  await mkdir(directory, { recursive: true });
  await writeFile(join(directory, 'b1.bin'), 'keep existing');
  try {
    for (const response of [() => new Response('bad'), () => new Response(Buffer.alloc(bytes.length)), () => new Response(Buffer.alloc(bytes.length + 1)),
      () => new Response(new ReadableStream({ start(controller) { controller.error(new Error('disconnected')); } }))]) {
      await assert.rejects(downloadModel('fixture', { root, models, log: () => {}, fetcher: async () => response() }));
      assert.equal(await readFile(join(directory, 'b1.bin'), 'utf8'), 'keep existing');
      assert.deepEqual(await readdir(directory), ['b1.bin']);
    }
  } finally { await rm(root, { recursive: true, force: true }); }
});
