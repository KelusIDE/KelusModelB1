
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { Model, enableKernels, disableKernels, looping } from '../../site/b1.js';

const open = async () => {
  const manifest = JSON.parse(await readFile(new URL('../../site/model/b1.json', import.meta.url), 'utf8'));
  const weights = await readFile(new URL('../../site/model/b1.bin', import.meta.url));
  return new Model(manifest, weights.buffer.slice(weights.byteOffset, weights.byteOffset + weights.byteLength));
};

for (const fused of [true, false]) {
  test(`kernel (${fused ? 'fused' : 'plain'} build) and JavaScript paths agree on logits and greedy replies`, async () => {
    disableKernels();
    const plain = await open();
    assert.equal(plain.kernel, undefined);
    assert.ok(await enableKernels({ fused }), 'WebAssembly SIMD is available in this Node');
    const fast = await open();
    assert.ok(fast.kernel);
    const ids = [257, 259, ...fast.tokenizer.encode('Write a Python function that adds two numbers.'), 260];
    const a = plain.nextLogits(ids, {}), b = fast.nextLogits(ids, {});
    for (let i = 0; i < a.length; i++) assert.ok(Math.abs(a[i] - b[i]) < 1e-3, `logit ${i}: ${a[i]} vs ${b[i]}`);
    for (const prompt of ['hi', 'What is a variable?', 'Write a Python function that checks whether a number is even.']) {
      const messages = [{ role: 'user', content: prompt }];
      assert.equal([...fast.generate(messages, { maxTokens: 60 })].join(''), [...plain.generate(messages, { maxTokens: 60 })].join(''), prompt);
    }
    disableKernels();
  });
}

test('a prompt read eight tokens at a time gives exactly the logits of reading it one token at a time', async () => {
  disableKernels();
  assert.ok(await enableKernels());
  const model = await open();
  const ids = [257, 259, ...model.tokenizer.encode('Explain what a for loop does, then write one in Python that prints 1 to 10.'), 260];
  assert.ok(ids.length > 16 && ids.length % 8, 'covers full batches and a remainder');
  const batched = model.nextLogits(ids, {}), cache = {};
  let single;
  for (let n = 1; n <= ids.length; n++) single = model.nextLogits(ids.slice(0, n), cache);
  assert.deepEqual(Array.from(batched), Array.from(single));
  disableKernels();
});

test('a greedy loop ends the reply, as in kelus/model.py', () => {
  const date = [32, 74, 117, 108, 121, 32, 50, 54, 44];
  assert.ok(looping([5, 6, ...Array(4).fill(date).flat()]));
  assert.ok(!looping([5, 6, ...Array(3).fill(date).flat()]));
  assert.ok(looping(Array(32).fill(65)));
  assert.ok(!looping(Array(31).fill(65)));
  assert.ok(!looping(Array.from({ length: 200 }, (_, i) => i)));
  assert.ok(!looping([...Array(8).fill([1, 2, 3, 4]).flat(), 7]));
});
