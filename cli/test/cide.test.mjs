import test from 'node:test';
import assert from 'node:assert/strict';
import { CIDE_CALL, maybeCide, codingContext, cideAnswer } from '../../site/cide.js';
import { loadCide } from '../cide.mjs';

const messages = [{ role: 'user', content: 'Fix the addition function.' }];
test('the default expert loads the configured current coding release', async () => {
  const model = await loadCide();
  assert.equal(model.manifest.name, 'Cide 4.5');
  assert.equal(model.manifest.specialist, 'coding');
  assert.equal(model.parameterCount, 205588480);
});
const expert = { config: { context: 96 }, tokenizer: { encode: text => [...text] },
  *generate(received) { assert.deepEqual(received, messages); yield 'CIDE'; } };

test('only possible control sequences are held; ordinary and quoted replies stream', () => {
  for (const text of ['', 'C', 'CI', 'CID', 'CIDE', 'CIDE\n']) assert.equal(maybeCide(text), true);
  for (const text of ['CIDER', 'Cide', 'CIDE is a name', '```\nCIDE', 'hello', 'SEARCH: cats']) assert.equal(maybeCide(text), false);
  assert.equal(CIDE_CALL, 'CIDE');
});

test('specialist receives the original task; its output never recursively calls tools', async () => {
  let loads = 0, text = '';
  for await (const piece of cideAnswer(messages, { load: async () => { loads++; return expert; }, maxTokens: 24 })) text += piece;
  assert.equal(loads, 1);
  assert.equal(text, 'CIDE');
});

test('cancelling before or during load prevents inference', async () => {
  let stopped = true;
  for await (const _ of cideAnswer(messages, { load: () => assert.fail('loaded after cancellation'), cancelled: () => stopped })) assert.fail('yielded after cancellation');
  stopped = false;
  for await (const _ of cideAnswer(messages, { load: () => { stopped = true; return expert; }, cancelled: () => stopped })) assert.fail('yielded after cancellation');
});

test('long latest source is rejected; old complete exchanges may be omitted', () => {
  assert.throws(() => codingContext(expert, [{ role: 'user', content: 'x'.repeat(100) }], 24), /too long/);
  const history = [{ role: 'user', content: 'x'.repeat(70) }, { role: 'assistant', content: 'Earlier reply.' }, ...messages];
  assert.deepEqual(codingContext(expert, history, 24), messages);
});

test('missing weights give an error and never fall back to impersonating Cide', async () => {
  await assert.rejects(async () => {
    for await (const _ of cideAnswer(messages, { load: () => { throw new Error('Cide is not installed'); } })) {}
  }, /not installed/);
});

test('coding output can finish beyond chat limits; an explicit limit is respected', async () => {
  const complete = '<!doctype html>' + 'x'.repeat(1200) + '</html>';
  const budgets = [];
  const model = { config: { context: 2048 }, tokenizer: expert.tokenizer,
    *generate(received, { maxTokens }) {
      assert.deepEqual(received, messages);
      budgets.push(maxTokens);
      yield complete.slice(0, maxTokens);
    } };
  let full = '', limited = '';
  for await (const text of cideAnswer(messages, { load: async () => model })) full += text;
  for await (const text of cideAnswer(messages, { load: async () => model, maxTokens: 128 })) limited += text;
  assert.equal(full, complete);
  assert.equal(limited.length, 128);
  assert.deepEqual(budgets, [4096, 128]);
});
