// Reproducible CPU inference benchmark. Fixed tokens measure speed, not answer quality.
import fs from 'node:fs/promises';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { Model, conversation } from '../site/b1.js';
const directory = process.argv[2] ? path.resolve(process.argv[2]) : new URL('../site/model/', import.meta.url);
const manifest = JSON.parse(await fs.readFile(directory instanceof URL ? new URL('b1.json', directory) : path.join(directory, 'b1.json'), 'utf8'));
const data = await fs.readFile(directory instanceof URL ? new URL('b1.bin', directory) : path.join(directory, 'b1.bin'));
const model = new Model(manifest, data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
const samples = [];
for (let repeat = 0; repeat < 6; repeat++) {
  const tokens = conversation([{role:'user',content:'Write a Python function to reverse a string.'}], model.tokenizer), cache = {};
  const start = performance.now();
  for (let step = 0; step < 64; step++) { model.nextLogits(tokens, cache); tokens.push(97); }
  if (repeat) samples.push(performance.now() - start);
  await new Promise(resolve => setImmediate(resolve));
}
const median = [...samples].sort((a,b) => a-b)[2];
console.log(JSON.stringify({ parameters: manifest.parameters, fixed_steps: 64, median_ms: median, steps_per_second: 64000/median, samples_ms: samples, note: 'One prompt prefill plus 63 cached decode steps; does not measure answer quality.' }, null, 2));
