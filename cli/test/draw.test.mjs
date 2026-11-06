import assert from 'node:assert';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { drawRequest, fixSpelling, isWorkspaceRequest, unknownSubject } from '../workspace-chat.mjs';

test('drawing requests go to Kani Painter', () => {
  for (const text of ['draw a red hot air balloon', 'Paint a sunset over the ocean', 'can you draw a snowy village at night',
    'make a picture of a city skyline', 'generate an image of deep space with two planets', 'show me a picture of a forest'])
    assert.strictEqual(drawRequest(text), text, text);
  assert.strictEqual(drawRequest('/draw a pine tree'), 'a pine tree');
  assert.strictEqual(drawRequest('/draw'), null);
  assert.ok(isWorkspaceRequest('draw a cactus'));
});

test('other requests stay with the chat model', () => {
  for (const text of ['how do I draw a circle in python?', 'what is a drawing tablet', 'write a function that draws', 'hello', 'drawer'])
    assert.strictEqual(drawRequest(text), null, text);
});

test('requests for things the painter never learned get an honest reply instead of a guess', () => {
  const subjects = JSON.parse(readFileSync(new URL('../../kelus/paint_subjects.json', import.meta.url), 'utf8'));
  for (const text of ['a person', 'a human waving', 'two red cars in the city', 'the ocean at sunset', 'an orange kitten'])
    assert.strictEqual(unknownSubject(text, subjects), null, text);
  for (const text of ['a dragon', 'my mom', 'a unicorn with wings'])
    assert.match(unknownSubject(text, subjects), /hasn't learned to draw that yet.*people, cats, dogs/, text);
});

test('a one-letter typo in a subject is corrected, the same way as kelus.paint.fix_spelling', () => {
  const known = new Set(['banana', 'elephant', 'house', 'horse', 'apple']);
  assert.strictEqual(fixSpelling('draw me a bnana', known), 'draw me a banana');
  assert.strictEqual(fixSpelling('draw an elephnat', known), 'draw an elephant');
  assert.strictEqual(fixSpelling('draw a hosue', known), 'draw a house');
  assert.strictEqual(fixSpelling('draw a hose', known), 'draw a hose');
});

test('common words are not corrected into subjects', () => {
  const known = new Set(['chair', 'tree', 'banana']), common = new Set(['hair', 'three', 'pink']);
  assert.strictEqual(fixSpelling('draw a girl with pink hair', known, common), 'draw a girl with pink hair');
  assert.strictEqual(fixSpelling('draw three apples', known, common), 'draw three apples');
  assert.strictEqual(fixSpelling('draw a bnana', known, common), 'draw a banana');
});
