import test from 'node:test';
import assert from 'node:assert/strict';
import { completedContext, restoreChats, updateHistory, MAX_CHATS } from '../src/history.mjs';
const pair = (id, status = 'complete', source = 'model') => [
  { id: id + 'u', role: 'user', content: id },
  { id: id + 'a', role: 'assistant', content: 'answer', status, source },
];
const chat = n => ({ id: `${n}`, title: `Chat ${n}`, model: 'Kani Seal 3', updated: n, messages: pair(`${n}`) });
test('restores a saved conversation without changing its content or model identity', () => {
  assert.deepEqual(restoreChats(JSON.stringify([chat(1)])), [chat(1)]);
  assert.deepEqual(restoreChats(null), []);
});
test('corrupt saved data raises instead of silently replacing existing chats', () => {
  for (const value of ['{', '{}', '[null]', JSON.stringify([{ ...chat(1), messages: [{ role: 'system' }] }])]) {
    assert.throws(() => restoreChats(value));
  }
});
test('stopped, failed, and unavailable-search turns never pollute the next prompt', () => {
  const successful = pair('good');
  assert.deepEqual(completedContext([...pair('bad', 'failed'), ...successful, ...pair('partial', 'stopped'), ...pair('search', 'complete', 'search_unavailable')]), successful);
});
test('translated turns reach the model in English', () => {
  const context = completedContext([
    { id: 'a', role: 'user', content: 'Hola', english: 'Hello' },
    { id: 'b', role: 'assistant', content: 'Hola, ¿qué tal?', english: 'Hi, how are you?', status: 'complete', source: 'model' },
    { id: 'c', role: 'user', content: 'Hi' },
    { id: 'd', role: 'assistant', content: 'Hello!', status: 'complete', source: 'model' }]);
  assert.deepEqual(context.map(m => m.content), ['Hello', 'Hi, how are you?', 'Hi', 'Hello!']);
});
test('context trimming preserves user/assistant pairs', () => {
  const result = completedContext(Array.from({ length: 12 }, (_, i) => pair(String(i))).flat());
  assert.equal(result.length, 14); assert.equal(result[0].role, 'user'); assert.equal(result.at(-1).role, 'assistant');
});
test('updating one conversation does not duplicate it; retention is bounded', () => {
  const history = Array.from({ length: MAX_CHATS }, (_, i) => chat(i));
  assert.equal(updateHistory(history, chat(5)).length, MAX_CHATS);
  assert.equal(updateHistory(history, chat(5))[0].id, '5');
  assert.equal(updateHistory(history, chat(100)).length, MAX_CHATS);
  assert.deepEqual(updateHistory(history, { ...chat(100), messages: [] }), history);
});
