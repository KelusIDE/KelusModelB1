
import test from 'node:test';
import assert from 'node:assert/strict';
import { Repeats } from '../../site/b1.js';

const run = (text, pieces = [text]) => {
  const repeats = new Repeats();
  let shown = '';
  for (const piece of pieces) {
    shown += repeats.push(piece);
    if (repeats.stopped) break;
  }
  return { shown: shown + repeats.finish(), stopped: repeats.stopped };
};
const byChar = text => run(text, [...text]);

test('a list item said again ends the reply before it, with no trailing blank lines', () => {
  const text = '1. Drink plenty of water\n2. Drink plenty of fluids\n\n3. Drink plenty of water\n4. Drink plenty';
  for (const result of [run(text), byChar(text)]) assert.deepEqual(result, { shown: '1. Drink plenty of water\n2. Drink plenty of fluids', stopped: true });
});

test('a sentence starting with the same eight words as an earlier one ends the reply', () => {
  const first = "I'm currently working on a project to develop a smart home system that can adapt to your home's needs.";
  const text = `I'm working on a new project. ${first}\n\nI'm currently working on a project to develop a smart home system that can adapt to your needs.`;
  assert.deepEqual(byChar(text), { shown: `I'm working on a new project. ${first}`, stopped: true });
});

test('a sentence is held while it could be a repeat, then shown once it differs', () => {
  const repeats = new Repeats();
  assert.equal(repeats.push('The cat sat on the mat. The cat sat on the'), 'The cat sat on the mat.');
  assert.equal(repeats.push(' sofa today. Then it slept.'), ' The cat sat on the sofa today. Then it slept.');
  assert.equal(repeats.finish(), '');
  assert.equal(repeats.stopped, false);
});

test('short sentences may come back, and nothing is lost when the reply ends on its own', () => {
  const text = 'Yes. Yes. Okay then, see you later. Okay then, see you';
  assert.deepEqual(byChar(text), { shown: text, stopped: false });
});

test('code between fences is never held or stopped', () => {
  const text = "Here it is:\n```python\nprint('hello world again now')\nprint('hello world again now')\n```\nprint('hello world again now')\n";
  assert.deepEqual(byChar(text), { shown: text, stopped: false });
});
