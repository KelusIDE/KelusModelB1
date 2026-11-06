import assert from 'node:assert';
import test from 'node:test';
import { Renderer, WORDMARK, banner, box, highlight, inputArea, theme, width } from '../ui.mjs';

const plain = theme({ mode: 0 });
const color = theme({ mode: 3 });
const CODE = '```python\ndef reverse(text):\n    return text[::-1]\n```';

function render(text, { t = plain, columns = 40, pieces = [...text] } = {}) {
  const renderer = new Renderer(t, columns);
  return renderer.begin() + pieces.map(piece => renderer.push(piece)).join('') + renderer.end();
}

test('code fences become a box and never stream as raw backticks', () => {
  assert.strictEqual(render(CODE), [
    '  ╭─ python ' + '─'.repeat(28),
    '  │ def reverse(text):',
    '  │     return text[::-1]',
    '  ╰─',
  ].join('\n'));
  assert.strictEqual(render(CODE, { pieces: [CODE] }), render(CODE));
  assert.ok(!render(CODE).includes('```'));
});

test('an unterminated block still closes', () => {
  assert.ok(render('```js\nconsole.log(1)').endsWith('  │ console.log(1)\n  ╰─'));
});

test('prose wraps at word boundaries with an indent and drops inline backticks', () => {
  const lines = render('Recursion is when a function calls itself. It needs a `base case` to stop.', { columns: 30 }).split('\n');
  assert.ok(lines.every(line => line.startsWith('  ') && width(line) <= 30), lines.join('|'));
  assert.ok(lines.join(' ').includes('base case'));
  assert.ok(!lines.join('').includes('`'));
});

test('colors are applied only when the theme has color', () => {
  assert.strictEqual(highlight(plain, 'def f(): return 1', 'python'), 'def f(): return 1');
  assert.match(highlight(color, 'def f(): return 1', 'python'), /\x1b\[38;2;197;134;192mdef/);
  assert.strictEqual(width(color.gradient('KANI')), 4);
  assert.strictEqual(banner(plain, 80).split('\n').length, WORDMARK.length);
});

test('boxes keep every line the same width', () => {
  const lines = box(color, ['one', color.color('teal', 'two')], 30, 'Title').split('\n');
  assert.deepStrictEqual(lines.map(width), [30, 30, 30, 30]);
});

test('the input box wraps long text and tracks the caret', () => {
  const area = inputArea(plain, { text: 'x'.repeat(30), cursor: 27, size: 30, placeholder: 'Ask', status: '' });
  assert.strictEqual(area.lines.filter(line => line.startsWith('│')).length, 2);
  assert.deepStrictEqual([area.row, area.column], [2, 7]);
  const empty = inputArea(plain, { text: '/n', cursor: 2, size: 60, placeholder: 'Ask', status: '', suggestions: [['/new', 'start a new chat']] });
  assert.match(empty.lines[empty.lines.length - 1], /\/new\s+start a new chat/);
});
