import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { compile, run, formatError } from '../public/lus/compiler/index.js';
import { LUS_EXAMPLES } from '../src/lus/examples.ts';

for (const example of LUS_EXAMPLES) {
  test(`${example.label} compiles to WebAssembly and produces the expected output`, async () => {
    const bytes = compile(example.code);
    assert.ok(WebAssembly.validate(bytes));
    const lines = []; const answers = example.input.split('\n'); let next = 0;
    await run(bytes, line => lines.push(line), () => answers[next++]);
    assert.deepEqual(lines, example.expected);
  });
}
test('compiler reports a useful source location for a type error', () => {
  const source = 'fn main {\n    let score = 1\n    score = 2\n}';
  try { compile(source); assert.fail('accepted a mutation of let'); }
  catch(error) { assert.match(formatError(error, source), /main\.lus:3:/); assert.match(error.message, /var/); }
});
test('runtime errors are surfaced', async () => {
  await assert.rejects(run(compile('fn main {\n    let zero = 0\n    say 1 / zero\n}'), () => {}), /zero/);
});
test('served compiler source matches the documented snapshot', () => {
  const manifest=JSON.parse(readFileSync(new URL('../public/lus/SOURCE.json',import.meta.url)));
  for(const [file,hash] of Object.entries(manifest.files)) assert.equal(createHash('sha256').update(readFileSync(new URL('../public/lus/compiler/'+file,import.meta.url))).digest('hex'),hash);
});
