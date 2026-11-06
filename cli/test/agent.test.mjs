import assert from 'node:assert';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { candidateCode, doTask, editPrompt, javascriptOutline, merge, parseTask, promptFor, sandboxTest, signaturePrefix } from '../agent.mjs';
import { Workspace } from '../workspace.mjs';

const OPTIONS = { python: 'python3', maxTokens: 120 };

async function project(files) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'kani-agent-'));
  for (const [name, text] of Object.entries(files)) {
    await fs.mkdir(path.dirname(path.join(root, name)), { recursive: true });
    await fs.writeFile(path.join(root, name), text);
  }
  test.after(() => fs.rm(root, { recursive: true, force: true }));
  return Workspace.open(root);
}


function scripted(...replies) {
  return { config: {context:4096}, tokenizer: {encode:s=>Array.from(s)}, calls: [], *generate(messages, options) { this.calls.push({ prompt: messages[0].content, ...options }); yield replies.shift() ?? ''; } };
}

async function collect(generator) {
  let text = '';
  for await (const piece of generator) text += piece;
  return text;
}

test('parses the target, task, name, and flags', () => {
  const task = parseTask('src/utils.py add a function scale(x) that returns x * 4 --tries 2 --test "python3 -m unittest -q"');
  assert.deepStrictEqual([task.file, task.language, task.name, task.tries, task.test], ['src/utils.py', 'Python', 'scale', 2, 'python3 -m unittest -q']);
  assert.strictEqual(promptFor(task), 'Write a Python function scale(x) that returns x * 4. Return only the function.');
  assert.throws(() => parseTask('notes.txt add things'), /Python \(\.py\) or JavaScript/);
});

test('finds JavaScript functions past braces in strings and comments', () => {
  const source = 'const a = 1;\nfunction f(x) {\n  const s = "}{";\n  // }\n  return x;\n}\nexport function g() {\n  return `${1}}`;\n}\n';
  assert.deepStrictEqual(javascriptOutline(source), [['f', 2, 6], ['g', 7, 9]]);
});

test('merges by replacing a function or appending one', () => {
  const original = 'def a():\n    return 1\n\ndef b():\n    return 2\n';
  assert.strictEqual(merge(original, 'def a():\n    return 9\n', ['a', 1, 2]), 'def a():\n    return 9\n\ndef b():\n    return 2\n');
  assert.ok(merge(original, 'def c():\n    return 3\n', null).endsWith('return 2\n\n\ndef c():\n    return 3\n'));
  assert.strictEqual(candidateCode('```python\ndef c():\n    return 3\n```'), 'def c():\n    return 3\n');
});

test('rejects bad candidates, then stages one that passes sandboxed tests', async () => {
  const workspace = await project({
    'utils.py': 'def keep():\n    return 1\n\n\ndef scale(x):\n    return x\n',
    'test_utils.py': 'import unittest\nfrom utils import scale, keep\n\nclass T(unittest.TestCase):\n    def test(self):\n        self.assertEqual(scale(3), 12)\n        self.assertEqual(keep(), 1)\n',
  });
  const model = scripted('```python\ndef scale(x:\n```', '```python\ndef grow(x):\n    return x * 4\n```',
    '```python\ndef scale(x):\n    return x * 3\n```', '```python\ndef scale(x):\n    return x * 4\n```');
  const out = await collect(doTask(workspace, model, 'utils.py make scale(x) return x * 4 --test "python3 -m unittest -q"', OPTIONS, {}));
  assert.match(out, /replace scale\(\) at lines 5-6/);
  assert.match(out, /✗ syntax/);
  assert.match(out, /✗ named grow, expected scale/);
  assert.match(out, /✗ tests failed/);
  assert.match(out, /✓ tests passed in the sandbox/);
  assert.deepStrictEqual(model.calls.map(call => call.temperature), [0, 0.7, 0.7, 0.7]);
  assert.match(model.calls[0].prompt, /Current Python function:\ndef scale\(x\):\n    return x/);
  assert.match(model.calls[0].prompt, /Reference test_utils.py/);
  assert.match(model.calls[3].prompt, /return x \* 3/);
  assert.match(model.calls[3].prompt, /AssertionError: 9 != 12/);
  const id = out.match(/Proposal ([a-f0-9-]{36})/)[1];
  const proposal = await workspace.proposal(id);
  assert.strictEqual(proposal.content, 'def keep():\n    return 1\n\n\ndef scale(x):\n    return x * 4\n');
  assert.strictEqual((await workspace.read('utils.py')).text.includes('x * 4'), false);
});

test('stages nothing when no candidate passes', async () => {
  const workspace = await project({ 'm.js': 'export function keep() {\n  return 1;\n}\n' });
  const out = await collect(doTask(workspace, scripted('no code here', '```js\nfunction other() { return 2; }\n```'), 'm.js add function twice(x) that returns x * 2 --tries 2', OPTIONS, {}));
  assert.match(out, /No candidate passed every check after 2 attempts/);
  await assert.rejects(fs.stat(path.join(workspace.root, '.kani/proposals')));
});

test('the sandbox blocks network access and writes outside the copy', { skip: process.platform !== 'darwin' }, async () => {
  const workspace = await project({ 'a.py': 'x = 1\n' });
  const outside = path.join(os.tmpdir(), `kani-outside-${process.pid}`);
  const write = await sandboxTest(workspace, 'a.py', 'x = 2\n', `echo hi > ${outside}`);
  assert.strictEqual(write.passed, false);
  await assert.rejects(fs.stat(outside));
  const network = await sandboxTest(workspace, 'a.py', 'x = 2\n', `python3 -c "import socket; socket.create_connection(('1.1.1.1', 80), timeout=3)"`);
  assert.strictEqual(network.passed, false);
  const inside = await sandboxTest(workspace, 'a.py', 'x = 2\n', 'python3 -c "import a; assert a.x == 2" && echo ok > made.txt');
  assert.strictEqual(inside.passed, true, inside.out);
});


test('cancelling after the last generated token still prevents staging', async () => {
  const workspace=await project({}); const cancel={cancelled:false};
  const model={config:{context:2048},tokenizer:{encode:s=>Array.from(s)},*generate(){yield 'def square(x):\n    return x * x\n';cancel.cancelled=true;}};
  const out=await collect(doTask(workspace,model,'a.py square(x) returns x * x',OPTIONS,cancel));
  assert.match(out,/Cancelled; no proposal saved/);
  await assert.rejects(fs.stat(path.join(workspace.root,'.kani/proposals')));
});

test('required source and failure are never silently truncated', () => {
  const model=scripted();
  model.config.context=180;
  assert.throws(()=>editPrompt(model,parseTask('a.py square(x) returns x*x'),'x'.repeat(180),[],null,60),/Nothing was staged/);
  const task=parseTask('a.py square(x) returns x*x');
  const result=editPrompt(model,task,null,[{path:'b.py',line:1,text:'x'.repeat(180)}],null,60);
  assert.equal(result.included,0);
  assert.match(result.prompt,/square/);
});

test('fixes the stated signature so the model cannot garble names', async () => {
  assert.strictEqual(signaturePrefix(parseTask('u.py add scale_value(x, y) that returns x * y')), '```python\ndef scale_value(x, y):');
  assert.strictEqual(signaturePrefix(parseTask('u.py add a function named boost_sum that takes left and right and returns left + right')), '```python\ndef boost_sum(left, right):');
  assert.strictEqual(signaturePrefix(parseTask('m.js add a function named $pick that doubles its input')), '```javascript\nfunction $pick(');
  assert.strictEqual(signaturePrefix(parseTask('u.py add something useful')), '');
  const workspace = await project({ 'u.py': 'x = 1\n' });
  const model = scripted('```python\ndef scale(x):\n    return x * 4\n```');
  const out = await collect(doTask(workspace, model, 'u.py add scale(x) that returns x * 4 --tries 1', OPTIONS, {}));
  assert.strictEqual(model.calls[0].prefix, '```python\ndef scale(x):');
  assert.match(out, /Signature fixed from your request: def scale\(x\):/);
});
