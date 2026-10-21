/**
 * /do: a checked edit loop around a small model.
 *
 * The control flow is ordinary code, not model reasoning: find the target, gather
 * evidence, ask the model for one function, check it, insert it, optionally run
 * the user's test command in a sandbox, and resample until something passes.
 * The result is a normal proposal for /diff and /apply; nothing is written directly.
 */
import { spawn } from 'node:child_process';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { checkSyntax } from './syntax.mjs';
import { clean, proposalDiff } from './workspace.mjs';

const LANGUAGES = { '.py': 'Python', '.js': 'JavaScript', '.mjs': 'JavaScript', '.cjs': 'JavaScript' };
const TEST_TIMEOUT = 60_000;
const COPY_LIMIT = 16 * 1024 * 1024;

/** `/do PATH TASK [--test "CMD"] [--tries N]` */
export function parseTask(args) {
  let text = args.trim();
  const options = { test: null, tries: 4 };
  const flag = /\s--(test|tries)\s+(?:"([^"]*)"|'([^']*)'|(\S+))\s*$/;
  for (let match; (match = text.match(flag));) {
    const value = match[2] ?? match[3] ?? match[4];
    if (match[1] === 'test') options.test = value;
    else options.tries = Math.max(1, Math.min(8, Number(value) || 4));
    text = text.slice(0, match.index).trim();
  }
  const parts = text.match(/^(?:"([^"]+)"|'([^']+)'|(\S+))\s+([\s\S]+)$/);
  if (!parts) throw new Error('Usage: /do PATH TASK [--test "COMMAND"] [--tries N]');
  const file = parts[1] || parts[2] || parts[3];
  const language = LANGUAGES[path.extname(file).toLowerCase()];
  if (!language) throw new Error('/do edits Python (.py) or JavaScript (.js/.mjs/.cjs) files.');
  const task = parts[4].trim();
  const named = task.match(/\b(?:named|called)\s+([A-Za-z_$][\w$]*)/) || task.match(/\b([A-Za-z_$][\w$]*)\s*\(/) ||
    task.match(/\b(?:def|function)\s+(?!(?:named|called|that|to|which)\b)([A-Za-z_$][\w$]*)/);
  return { file, language, task, name: named ? named[1] : null, ...options };
}

/** The phrasing the model was trained on: "Write a Python function name(args) that ...". */
export function promptFor({ language, task }) {
  const body = task.replace(/^(?:please\s+)?(?:add|write|create|make|implement)\s+(?:a\s+|an\s+)?(?:python\s+|javascript\s+|js\s+)?(?:function\s+)?/i, '');
  return `Write a ${language} function ${body.replace(/[.\s]+$/, '')}. Return only the function.`;
}

/**
 * The signature stated in the request, used as a decoding prefix so the model cannot
 * garble the function name or parameters. Parameters come from `name(a, b)` or from
 * "takes a and b and ..."; otherwise only the name is fixed.
 */
export function signaturePrefix({ language, task, name }) {
  if (!name) return '';
  const stated = task.match(new RegExp(`\\b${name.replace(/\$/g, '\\$')}\\s*\\(([^)]*)\\)`));
  const takes = task.match(/\btakes\s+([A-Za-z_$][\w$]*(?:(?:,\s*|\s+and\s+)[A-Za-z_$][\w$]*)*)\s+and\s+(?:returns|adds|subtracts|multiplies|divides|squares|doubles)\b/);
  const params = stated ? stated[1].trim() : takes ? takes[1].split(/,\s*|\s+and\s+/).join(', ') : null;
  if (language === 'Python') return params === null ? `\`\`\`python\ndef ${name}(` : `\`\`\`python\ndef ${name}(${params}):`;
  return params === null ? `\`\`\`javascript\nfunction ${name}(` : `\`\`\`javascript\nfunction ${name}(${params}) {`;
}

/** Preserve required source and feedback; omit only whole optional reference lines. */
export function editPrompt(model, task, source, references, failure, maxTokens) {
  let prompt = promptFor(task);
  if (source !== null) prompt += `\nCurrent ${task.language} function:\n${source}`;
  if (failure) prompt += `\nPrevious attempt:\n${failure.code || '(no complete code)'}\nCheck failure:\n${failure.reason}\nCorrect this failure while satisfying the original request.`;
  const budget = model.config.context - Math.min(maxTokens, Math.max(16, Math.floor(model.config.context / 3)));
  const count = text => model.tokenizer.encode(text).length + 3;
  if (count(prompt) > budget) throw new Error(`The task, source and check feedback need ${count(prompt)} tokens; only ${budget} fit. Use a smaller function or a larger-context model. Nothing was staged.`);
  let included = 0;
  for (const ref of references) {
    const next = `\nReference ${ref.path}:${ref.line}: ${ref.text}`;
    if (count(prompt + next) > budget) break;
    prompt += next; included++;
  }
  return { prompt, included };
}

function run(command, args, { input = '', cwd, timeout = 10_000, env } = {}) {
  return new Promise(resolve => {
    const child = spawn(command, args, { cwd, env, stdio: ['pipe', 'pipe', 'pipe'] });
    let out = '', timedOut = false;
    const keep = chunk => { if (out.length < 20_000) out += chunk; };
    const timer = setTimeout(() => { timedOut = true; child.kill('SIGKILL'); }, timeout);
    child.stdout.on('data', keep);
    child.stderr.on('data', keep);
    child.stdin.on('error', () => {});
    child.on('error', error => { clearTimeout(timer); resolve({ code: -1, out: error.message, timedOut }); });
    child.on('close', code => { clearTimeout(timer); resolve({ code, out, timedOut }); });
    child.stdin.end(input);
  });
}

const PY_OUTLINE = `
import ast, json, sys
source = sys.stdin.read()
tree = ast.parse(source)
print(json.dumps([[n.name, n.lineno, n.end_lineno] for n in tree.body
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]))
`;

/** Top-level functions as [name, firstLine, lastLine]; parsed, never executed. */
async function outline(language, source, python) {
  if (language === 'Python') {
    const result = await run(python, ['-I', '-c', PY_OUTLINE], { input: source });
    if (result.code !== 0) throw new Error(result.out.trim().split('\n').pop() || 'Python could not parse the source.');
    return JSON.parse(result.out);
  }
  return javascriptOutline(source);
}

/** Function declarations found by a small scanner that skips strings and comments. */
export function javascriptOutline(source) {
  const lines = source.split('\n');
  const found = [];
  lines.forEach((line, index) => {
    const match = line.match(/^(?:export\s+)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)\s*\(/);
    if (!match) return;
    let depth = 0, started = false, quote = null, comment = null;
    for (let i = index; i < lines.length; i++) {
      const text = lines[i];
      for (let j = 0; j < text.length; j++) {
        const c = text[j], next = text[j + 1];
        if (comment === 'line') break;
        if (comment === 'block') { if (c === '*' && next === '/') { comment = null; j++; } continue; }
        if (quote) { if (c === '\\') j++; else if (c === quote) quote = null; continue; }
        if (c === '/' && next === '/') { comment = 'line'; break; }
        if (c === '/' && next === '*') { comment = 'block'; j++; continue; }
        if (c === '"' || c === "'" || c === '`') { quote = c; continue; }
        if (c === '{') { depth++; started = true; }
        if (c === '}') depth--;
        if (started && depth === 0) { found.push([match[1], index + 1, i + 1]); return; }
      }
      if (comment === 'line') comment = null;
    }
  });
  return found;
}

/** Extract exactly one code block (or bare code) from a reply. */
export function candidateCode(reply) {
  const blocks = [...reply.matchAll(/```[\w+#.-]*[^\n]*\n([\s\S]*?)(?:```|$)/g)].map(m => m[1]);
  const code = (blocks.length ? blocks[0] : reply).replace(/\s+$/, '');
  if (!code.trim()) throw new Error('empty reply');
  return code + '\n';
}

/** Insert the function, or replace the existing one of the same name. */
export function merge(original, code, existing) {
  if (original === null) return code;
  const lines = original.replace(/\n*$/, '').split('\n');
  if (existing) {
    const [, first, last] = existing;
    return [...lines.slice(0, first - 1), ...code.replace(/\n$/, '').split('\n'), ...lines.slice(last)].join('\n') + '\n';
  }
  return lines.join('\n') + '\n\n\n' + code;
}

const SANDBOX = root => `(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* (subpath ${JSON.stringify(root)}) (literal "/dev/null") (literal "/dev/tty") (regex #"^/dev/fd/"))`;

/**
 * Run the user's test command against a copy of the project that contains the change.
 * macOS sandbox: no network, and writes only inside the temporary copy.
 */
export async function sandboxTest(workspace, relativePath, content, command) {
  if (process.platform !== 'darwin') return { ran: false, passed: false, out: 'Sandboxed tests need macOS sandbox-exec; tests were not run.' };
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'kani-test-')));
  try {
    const { files } = await workspace.files();
    let bytes = 0;
    for (const name of files) {
      const source = path.join(workspace.root, name);
      const info = await fs.stat(source).catch(() => null);
      if (!info?.isFile() || (bytes += info.size) > COPY_LIMIT) continue;
      await fs.mkdir(path.dirname(path.join(root, name)), { recursive: true });
      await fs.copyFile(source, path.join(root, name));
    }
    await fs.mkdir(path.dirname(path.join(root, relativePath)), { recursive: true });
    await fs.writeFile(path.join(root, relativePath), content);
    const env = { PATH: process.env.PATH, HOME: root, TMPDIR: root, PYTHONDONTWRITEBYTECODE: '1', LANG: 'en_US.UTF-8' };
    const result = await run('/usr/bin/sandbox-exec', ['-p', SANDBOX(root), '/bin/sh', '-c', command], { cwd: root, timeout: TEST_TIMEOUT, env });
    return { ran: true, passed: result.code === 0 && !result.timedOut, out: result.timedOut ? 'Timed out after 60 s.' : result.out };
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
}

const tail = (text, lines = 8) => clean(text).trim().split('\n').slice(-lines).map(line => '      ' + line.slice(0, 200)).join('\n');

/** Yields text for the chat view; stages a proposal only when a candidate passes every check. */
export async function* doTask(workspace, model, args, options, cancel) {
  const task = parseTask(args);
  const target = await workspace.resolve(task.file, true);
  let file = null;
  try { file = await workspace.read(target.rel); } catch (error) { if (error.code !== 'ENOENT') throw error; }
  const functions = file ? await outline(task.language, file.text, options.python) : [];
  const existing = task.name ? functions.find(([name]) => name === task.name) : null;

  yield `● Plan · ${task.language} · ${target.rel}${file ? '' : ' (new file)'}\n`;
  yield `  1. ${existing ? `replace ${task.name}() at lines ${existing[1]}-${existing[2]}` : task.name ? `add ${task.name}()` : 'add the requested function'}\n`;
  yield `  2. generate up to ${task.tries} candidates; check syntax${task.name ? ' and the function name' : ''}\n`;
  yield `  3. ${task.test ? `run \`${task.test}\` in a sandbox (no network, temporary copy)` : 'no test command given: syntax checks only (add --test "CMD")'}\n`;
  yield `  4. stage a proposal for review; nothing is written until /apply\n\n`;

  yield `● Context\n`;
  yield `  ${file ? `${target.rel}: ${file.lines.length} lines; functions: ${functions.map(([name]) => name).join(', ') || 'none'}` : `${target.rel} does not exist yet`}\n`;
  let references = [];
  if (task.name) {
    const { hits } = await workspace.search(task.name, 6);
    references = hits.filter(hit => hit.path !== target.rel);
    yield hits.length ? hits.map(hit => `  ${hit.path}:${hit.line} ${clean(hit.text).trim().slice(0, 120)}`).join('\n') + '\n' : `  no other references to ${task.name}\n`;
  }
  const source = existing ? file.lines.slice(existing[1] - 1, existing[2]).join('\n') : null;
  let failure = null;

  const prefix = signaturePrefix(task);
  if (prefix) yield `  Signature fixed from your request: ${clean(prefix.split('\n')[1])}\n`;
  const seen = new Set();
  for (let attempt = 1; attempt <= task.tries; attempt++) {
    if (cancel.cancelled) { yield 'Cancelled; no proposal saved.\n'; return; }
    const temperature = attempt === 1 ? 0 : 0.7;
    let request;
    try { request = editPrompt(model, task, source, references, failure, options.maxTokens); }
    catch (error) { yield `${clean(error.message)}\n`; return; }
    yield `  Model input: ${source === null ? 'new function' : 'complete current function'}, ${request.included}/${references.length} reference lines${failure ? ', previous code and check failure' : ''}.\n`;
    yield `● Attempt ${attempt}/${task.tries} · temperature ${temperature}\n`;
    let reply = '';
    for (const text of model.generate([{ role: 'user', content: request.prompt }], { maxTokens: options.maxTokens, temperature, prefix })) {
      if (cancel.cancelled) break;
      reply += text;
      await new Promise(resolve => setImmediate(resolve));
    }
    if (cancel.cancelled) { yield 'Cancelled; no proposal saved.\n'; return; }
    let code;
    try { code = candidateCode(reply); } catch { failure = { code: '', reason: 'Return exactly one complete function, in a code block.' }; yield '  ✗ no code in the reply\n'; continue; }
    if (seen.has(code)) { yield '  · same as an earlier candidate; resampling\n'; continue; }
    seen.add(code);
    yield code.replace(/\n$/, '').split('\n').map(line => '    │ ' + line).join('\n') + '\n';

    let defined;
    try { defined = await outline(task.language, code, options.python); }
    catch (error) { failure = { code, reason: clean(error.message).slice(0, 500) }; yield `  ✗ syntax: ${clean(error.message).slice(0, 160)}\n`; continue; }
    if (defined.length !== 1) { failure = { code, reason: `Expected exactly one function, got ${defined.length}.` }; yield `  ✗ expected exactly one function, got ${defined.length}\n`; continue; }
    if (task.name && defined[0][0] !== task.name) { failure = { code, reason: `Function must be named ${task.name}, not ${defined[0][0]}.` }; yield `  ✗ named ${defined[0][0]}, expected ${task.name}\n`; continue; }
    const merged = merge(file?.text ?? null, code, existing ?? (functions.find(([name]) => name === defined[0][0]) || null));
    try { await checkSyntax(target.rel, merged, options.python); } catch (error) { failure = { code, reason: clean(error.message).slice(0, 500) }; yield `  ✗ file no longer parses: ${clean(error.message).split('\n')[0]}\n`; continue; }
    yield `  ✓ syntax and name\n`;

    if (task.test) {
      const result = await sandboxTest(workspace, target.rel, merged, task.test);
      if (!result.ran) { yield `  ! ${result.out}\nNo proposal saved because the requested tests did not run.\n`; return; }
      else if (!result.passed) { failure = { code, reason: `Test command: ${task.test}\n${tail(result.out)}` }; yield `  ✗ tests failed\n${tail(result.out)}\n`; continue; }
      else yield `  ✓ tests passed in the sandbox\n`;
    }

    if (cancel.cancelled) { yield 'Cancelled; no proposal saved.\n'; return; }
    // Recheck the file after generation, then stage like /draft does.
    let now = null;
    try { now = await workspace.read(target.rel); } catch (error) { if (error.code !== 'ENOENT') throw error; }
    if ((now?.hash ?? null) !== (file?.hash ?? null)) throw new Error('The file changed while /do was running. Nothing was staged.');
    const proposal = await workspace.stage(target.rel, merged);
    yield `\n● Proposal ${proposal.id}\n${proposalDiff(proposal)}\n\n`;
    yield `Review it. /apply ${proposal.id} writes it; /discard ${proposal.id} removes it.\n`;
    yield task.test ? 'Your tests passed against this change in a sandboxed copy.\n' : 'Only syntax was checked; behavior is untested.\n';
    return;
  }
  yield `\nNo candidate passed every check after ${task.tries} attempts. Nothing was staged.\n`;
  yield `The model may not know this task yet; try a precise spec like "name(x) that returns x * 4".\n`;
}
