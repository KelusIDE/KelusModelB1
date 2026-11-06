import { checkSyntax } from './syntax.mjs';
import { doTask } from './agent.mjs';
import { extractSvg, svgPrompt } from '../site/svg.js';
import { webAnswer, webRequest } from '../site/web.js';
import path from 'node:path';
import fs from 'node:fs/promises';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
const exec = promisify(execFile);
import { clean, extractCode, pathArgument, proposalDiff } from './workspace.mjs';

export const WORKSPACE_COMMANDS = [
  ['/web', 'online Wikipedia search: QUERY; or /web read TITLE/URL'],
  ['/svg', 'generate a vector image proposal: PATH.svg DESCRIPTION'],
  ['/draw', 'paint a picture with Kani Painter: DESCRIPTION (or just say "draw …")'],
  ['/do', 'checked edit: PATH TASK [--test "CMD"] [--tries N]'],
  ['/vision', 'experimental screenshot regions: PATH (requires Python/Pillow)'],
  ['/files', 'list project files (optional name filter)'],
  ['/analyze', 'inspect project or /analyze PATH'],
  ['/read', 'read PATH [start:end], with line numbers'],
  ['/search', 'find literal text in project files'],
  ['/ask', 'ask about PATH[:start-end] QUESTION'],
  ['/draft', 'propose a complete file: PATH REQUEST'],
  ['/diff', 'show a saved proposal: ID'],
  ['/apply', 'apply a reviewed proposal: ID'],
  ['/discard', 'remove a proposal: ID'],
];
const known = new Set(WORKSPACE_COMMANDS.map(([name]) => name));
const DRAW = /^(?:(?:please|can you|could you|would you)\s+)?(?:draw|paint|sketch)\b|^(?:(?:please|can you|could you)\s+)?(?:make|generate|create|show)\s+(?:me\s+)?(?:a|an)\s+(?:picture|image|drawing|painting|scene)\b/i;

export function drawRequest(input) {
  const text = input.trim();
  if (/^\/draw\b/.test(text)) return text.slice(5).trim() || null;
  return DRAW.test(text) ? text : null;
}
export function isWorkspaceRequest(input) {
  return Boolean(webRequest(input)) || Boolean(drawRequest(input)) || known.has(input.split(/\s/)[0]) || /^(?:please\s+)?(?:analy[sz]e|inspect)\s+(?:(?:the|this|my)\s+)?(?:files|project|workspace|repo)(?:\s+please)?[.!?]*$/i.test(input.trim());
}
const reserve = (model, options) => Math.min(options.maxTokens, Math.max(16, Math.floor(model.config.context / 3)));
const count = (model, text) => model.tokenizer.encode(text).length + 3;

export function contextPrompt(model, question, file, { maxTokens = 240, start = 1, end = file.lines.length } = {}) {
  const budget = model.config.context - reserve(model, { maxTokens });
  const base = `Question: ${question}\nSource: ${file.path}\n`;
  if (count(model, base) >= budget) throw new Error('The question is too long for this model context.');
  let prompt = base, last = start - 1;
  for (let i = start - 1; i < Math.min(end, file.lines.length); i++) {
    const next = `${i + 1}: ${file.lines[i]}\n`;
    if (count(model, prompt + next) > budget) break;
    prompt += next; last = i + 1;
  }
  if (last < start) throw new Error('No complete source line fits this model context. Choose shorter lines or a model with a larger context.');
  return { prompt, note: `Model context: ${file.path}:${start}-${last} · ${count(model, prompt)}/${model.config.context} tokens${last < Math.min(end, file.lines.length) ? ' · later lines omitted' : ''}.` };
}

async function* generate(model, prompt, options, run) {
  for (const text of model.generate([{ role: 'user', content: prompt }], options)) {
    if (run.cancelled) return;
    yield text;
    await new Promise(resolve => setImmediate(resolve));
  }
}

let painter = null;
async function loadPainter(directory) {
  if (painter?.directory === directory) return painter.model;
  const { Model, enableKernels } = await import('../site/b1.js');
  await enableKernels();
  let manifest, weights;
  try {
    manifest = JSON.parse(await fs.readFile(path.join(directory, 'b1.json'), 'utf8'));
    weights = await fs.readFile(path.join(directory, 'b1.bin'));
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
    throw new Error(`No Kani Painter model in ${directory}. After training, export it: python3 -m kelus.export_web --checkpoint checkpoints/kani-painter-v4b.npz --out ${directory}`);
  }
  painter = { directory, model: new Model(manifest, weights.buffer.slice(weights.byteOffset, weights.byteOffset + weights.byteLength)) };
  return painter.model;
}


export function unknownSubject(description, subjects) {
  const known = new Set(subjects.words);
  if ((description.toLowerCase().match(/[a-z]+/g) || []).some(word => known.has(word))) return null;
  const things = subjects.objects, settings = subjects.settings;
  return "Kani Painter hasn't learned to draw that yet, so it would only guess. It can draw "
    + `${things.slice(0, -1).join(', ')} and ${things.at(-1)}, in scenes like ${settings.slice(0, -1).join(', ')} `
    + `or ${settings.at(-1)}. Try: draw a person in a meadow at sunset.`;
}


async function* pixels(workspace, description, options, run) {

  const subjects = JSON.parse(await fs.readFile(path.join(options.engineRoot, 'kelus/paint_subjects.json'), 'utf8').catch(() => 'null'));
  if (subjects && !unknownSubject(description, subjects)) return false;
  const target = await workspace.resolve(`.kani/drawings/pixels-${Date.now()}-${Math.random().toString(16).slice(2, 8)}.png`, true, true);
  await fs.mkdir(path.dirname(target.full), { recursive: true, mode: 0o700 });

  const venv = path.join(options.engineRoot, '.venv/bin/python');
  const python = await fs.access(venv).then(() => venv, () => options.python);
  let done;
  try {
    done = await exec(python, ['-m', 'kelus.pixel_dit', 'sample', '--run', path.join(options.engineRoot, 'training-runs/kani-pixels-v2'),
      '--prompt', description, '--out', target.full, '--count', '4', '--columns', '2', '--scale', '4'],
    { cwd: options.engineRoot, timeout: 120000, maxBuffer: 1024 * 1024 });
  } catch (error) {
    throw new Error(`Pixel generation failed: ${(error.stderr || error.message).trim().split('\n').pop()}`);
  }
  const { subject } = JSON.parse(done.stdout.trim().split('\n').pop());
  if (!subject) return false;
  if (run.cancelled) { await fs.unlink(target.full); return true; }
  yield { source: 'pixels', omitted: 0 };
  yield { text: `Kani Pixels v2 painted ${subject}: four tries at 64×64 pixels, from a diffusion model trained from scratch on Open Images photos. Experimental and blurry.\nImage: ${clean(target.full)}` };
  return true;
}


async function sketch(workspace, description, options) {
  const checkpoint = options.sketchModel || path.join(options.engineRoot, 'training-runs/kani-sketch-v1/best.npz');
  try { await fs.access(checkpoint); } catch { return null; }
  const slug = description.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 40) || 'drawing';
  const target = await workspace.resolve(`.kani/drawings/sketch-${slug}-${new Date().toISOString().replace(/[-:]/g, '').slice(0, 15)}.png`, true, true);
  await fs.mkdir(path.dirname(target.full), { recursive: true, mode: 0o700 });
  try {
    const { stdout } = await exec(options.python, ['-m', 'kelus.sketch', 'draw', '--checkpoint', checkpoint, '--prompt', description, '--out', target.full],
      { cwd: options.engineRoot, timeout: 120000, maxBuffer: 1024 * 1024 });
    const result = JSON.parse(stdout.trim().split('\n').pop());
    return `Kani Sketch drew ${/^[aeiou]/i.test(result.subject) ? 'an' : 'a'} ${clean(result.subject)}: a doodle, learned from 414,000 Quick, Draw! drawings.\nImage: ${clean(target.full)}`;
  } catch (error) {
    if (/unknown subject/.test(error.stdout || '') || /unknown subject/.test(error.stderr || '')) return null;
    throw error;
  }
}


function near(a, b) {
  if (a === b || Math.abs(a.length - b.length) > 1) return a === b;
  if (a.length === b.length) {
    const diff = [...a].map((c, i) => c !== b[i] ? i : -1).filter(i => i >= 0);
    return diff.length === 1 || (diff.length === 2 && diff[1] === diff[0] + 1 && a[diff[0]] === b[diff[1]] && a[diff[1]] === b[diff[0]]);
  }
  const [short, long] = a.length < b.length ? [a, b] : [b, a];
  let i = 0;
  while (i < short.length && short[i] === long[i]) i++;
  return short.slice(i) === long.slice(i + 1);
}


export function fixSpelling(description, known, common = new Set()) {
  return description.replace(/[A-Za-z]+/g, word => {
    const lower = word.toLowerCase();
    if (lower.length < 4 || known.has(lower) || common.has(lower)) return word;
    const close = [...known].filter(w => w.length >= 4 && near(lower, w));
    return close.length === 1 ? close[0] : word;
  });
}


async function* paint(workspace, description, options, run) {
  let subjects = null;
  try { subjects = JSON.parse(await fs.readFile(path.join(options.engineRoot, 'kelus/paint_subjects.json'), 'utf8')); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
  const known = new Set(subjects?.words || []);
  try { for (const word of (await fs.readFile(path.join(options.engineRoot, 'data/quickdraw/all-categories.txt'), 'utf8')).toLowerCase().match(/[a-z]+/g) || []) known.add(word); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
  let common = new Set();
  try { common = new Set((await fs.readFile(path.join(options.engineRoot, 'kelus/english_words.txt'), 'utf8')).split('\n')); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
  description = fixSpelling(description, known, common);
  const refusal = subjects && unknownSubject(description, subjects);
  if (refusal) {

    const sketched = await sketch(workspace, description, options);
    if (sketched) { yield { source: 'sketch', omitted: 0 }; yield { text: sketched }; return; }
    yield { source: 'painter', omitted: 0 }; yield { text: refusal + ' Kani Sketch, which doodles 345 everyday subjects from Quick, Draw!, has not learned it either.' }; return;
  }
  const model = await loadPainter(options.painterDir);
  yield { source: 'painter', omitted: 0 };
  yield { text: `Kani Painter is writing a drawing program for: ${clean(description)}\n` };
  let reply = '';
  for await (const text of generate(model, description, { ...options, maxTokens: 900, temperature: 0, prefix: '```python\n' }, run)) reply += text;
  if (run.cancelled) { yield { text: 'Cancelled; nothing saved.' }; return; }
  const program = (reply.match(/```(?:python)?\n([\s\S]*?)(?:```|$)/) || [, reply])[1].trimEnd() + '\n';
  const slug = description.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 40) || 'drawing';
  const stamp = new Date().toISOString().replace(/[-:]/g, '').slice(0, 15);
  const source = await workspace.resolve(`.kani/drawings/${slug}-${stamp}.py`, true, true);
  await fs.mkdir(path.dirname(source.full), { recursive: true, mode: 0o700 });
  await fs.writeFile(source.full, program, { flag: 'wx', mode: 0o600 });
  const image = source.full.replace(/\.py$/, '.png');
  try {
    await exec(options.python, ['-m', 'kelus.paint', source.full, '--out', image], { cwd: options.engineRoot, timeout: 30000, maxBuffer: 1024 * 1024 });
  } catch (error) {
    const reason = (error.stderr || error.message).trim().split('\n').pop();
    yield { text: `The painter's program did not render: ${clean(reason)}\nProgram kept for review: ${clean(source.full)}` };
    return;
  }
  yield { text: `Image: ${clean(image)}\nProgram (${program.split('\n').length - 1} lines): ${clean(source.full)}\nRendered by Kani Paint's checked interpreter; the program never ran as Python.` };
}


export async function* workspaceReply(workspace, model, input, options, run) {
  const online=webRequest(input);
  if(online) {
    yield {source:'web',omitted:0};
    const controller=new AbortController(), timer=setInterval(()=>{if(run.cancelled)controller.abort();},100);
    try { const result=await webAnswer(online,fetch,controller.signal); if(!run.cancelled)yield {text:clean(result.text)}; }
    finally {clearInterval(timer);}
    return;
  }
  const drawing = drawRequest(input);
  if (drawing !== null || /^\/draw\s*$/.test(input.trim())) {
    if (!drawing) throw new Error('Usage: /draw DESCRIPTION, for example /draw a sunset over the ocean with two sailboats');
    if (!['pixels', 'painter'].includes(options.imageModel || 'painter')) throw new Error('Image model must be pixels or painter');

    if (options.imageModel === 'pixels' && (yield* pixels(workspace, drawing, options, run))) return;
    yield* paint(workspace, drawing, options, run);
    return;
  }
  const parts = input.trim().match(/^(\S+)(?:\s+([\s\S]*))?$/);
  let command = parts[1], args = parts[2] || '';
  if (!command.startsWith('/')) { command = '/analyze'; args = ''; }
  yield { source: 'workspace', omitted: 0 };
  let result;
  switch (command) {
    case '/svg': {
      const [name, description] = pathArgument(args);
      if (!description || !name.toLowerCase().endsWith('.svg')) throw new Error('Usage: /svg PATH.svg DESCRIPTION');
      await workspace.resolve(name, true);
      let before = null;
      try { before = await workspace.read(name); } catch (error) { if (error.code !== 'ENOENT') throw error; }
      const prompt = svgPrompt(description);
      yield { text: `Generating SVG for ${clean(name)} from the loaded model…\n` };
      let reply = '';
      for await (const text of generate(model, prompt, options, run)) reply += text;
      if (run.cancelled) { yield { text: 'Cancelled; no proposal saved.' }; return; }
      let content;
      try { content = extractSvg(reply) + '\n'; }
      catch (error) { throw new Error(`The model did not produce a complete, supported SVG. ${error.message} No file was staged.`); }
      let now = null;
      try { now = await workspace.read(name); } catch (error) { if (error.code !== 'ENOENT') throw error; }
      if ((now?.hash ?? null) !== (before?.hash ?? null)) throw new Error('File changed while generating. No proposal saved.');
      const proposal = await workspace.stage(name, content);
      const preview = await workspace.resolve(`.kani/previews/${proposal.id}.svg`, true, true);
      await fs.mkdir(path.dirname(preview.full), { recursive: true, mode: 0o700 });
      await fs.writeFile(preview.full, content, { flag: 'wx', mode: 0o600 });
      yield { text: `Validated SVG preview: ${clean(preview.full)}\n` + proposalDiff(proposal)
        + `\n\nReview the image and source. /apply ${proposal.id} saves ${clean(name)}; /discard ${proposal.id} removes the proposal.` };
      return;
    }
    case '/vision': {
      const [name] = pathArgument(args), target = await workspace.resolve(name);
      const info = await fs.stat(target.full);
      if (!info.isFile() || info.size > 20 * 1024 * 1024) throw new Error('Expected an image smaller than 20 MiB.');
      try { await fs.access(options.visionModel); } catch { throw new Error('No vision checkpoint. In the Kani repository run: python3 -m kelus.vision train'); }
      const { stdout } = await exec(options.python, ['-m', 'kelus.vision', 'inspect', '--image', target.full, '--checkpoint', options.visionModel], { cwd: options.engineRoot, timeout: 30000, maxBuffer: 1024 * 1024 });
      const report = JSON.parse(stdout);
      result = `Experimental UI region prediction: ${name}\nTraining data: ${report.training.data.kind}\n` +
        report.grid.map(row => row.map(id => ['B','T','S','E','R','D'][id]).join(' ')).join('\n') +
        '\nLegend: B background, T toolbar, S sidebar, E editor, R terminal, D dialog\n' + report.limitation;
      break;
    }
    case '/files': {
      const listing = await workspace.files();
      const matches = listing.files.filter(f => f.toLowerCase().includes(args.toLowerCase()));
      result = `${matches.length} matching files · ${listing.source}\n` + matches.slice(0, 150).join('\n') + (matches.length > 150 || listing.truncated ? '\nListing capped; narrow the name filter.' : '');
      break;
    }
    case '/analyze': result = await workspace.analyze(args ? pathArgument(args)[0] : undefined); break;
    case '/read': {
      const [name, range] = pathArgument(args), file = await workspace.read(name);
      if (range && !/^\d+(?::\d+)?$/.test(range)) throw new Error('Usage: /read PATH [start:end]');
      const [a, b] = range.split(':').map(Number), start = range ? a : 1, end = range.includes(':') ? b : Math.min(start + 79, file.lines.length);
      if (start < 1 || end < start || start > file.lines.length) throw new Error('Line range is outside this file.');
      const stop = Math.min(end, start + 199, file.lines.length);
      result = file.lines.slice(start - 1, stop).map((line, i) => `${file.path}:${start + i} ${line.slice(0, 1000)}${line.length > 1000 ? ' [line clipped]' : ''}`).join('\n');
      if (stop < file.lines.length) result += `\nShowing ${start}-${stop}/${file.lines.length} lines.`;
      break;
    }
    case '/search': {
      const { hits, truncated, skipped } = await workspace.search(args);
      result = hits.map(h => `${h.path}:${h.line} ${h.text}`).join('\n') || 'No matches in eligible text files.';
      if (truncated) result += '\nSearch limit reached; narrow the query.';
      if (skipped) result += `\nSkipped ${skipped} unreadable, binary or oversized files.`;
      break;
    }
    case '/do': {
      for await (const text of doTask(workspace, model, args, options, run)) yield { text };
      return;
    }
    case '/diff': result = proposalDiff(await workspace.proposal(args.trim())); break;
    case '/apply': result = await workspace.apply(args.trim()); break;
    case '/discard': await workspace.discard(args.trim()); result = 'Proposal discarded. Project files were not changed.'; break;
    case '/ask': {
      const [spec, question] = pathArgument(args);
      if (!question) throw new Error('Usage: /ask PATH[:start-end] QUESTION');
      const match = spec.match(/^(.*):(\d+)-(\d+)$/), name = match ? match[1] : spec;
      const file = await workspace.read(name), start = match ? Number(match[2]) : 1, end = match ? Number(match[3]) : file.lines.length;
      if (start < 1 || start > end || start > file.lines.length) throw new Error('Invalid source range.');
      const { prompt, note } = contextPrompt(model, question, file, { ...options, start, end });
      yield { text: clean(note) + '\n\nModel response (may be inaccurate):\n' };
      for await (const text of generate(model, prompt, options, run)) yield { text };
      return;
    }
    case '/draft': {
      const [name, request] = pathArgument(args);
      if (!request) throw new Error('Usage: /draft PATH REQUEST');
      await workspace.resolve(name, true);
      let file = null;
      try { file = await workspace.read(name); } catch (e) { if (e.code !== 'ENOENT') throw e; }

      const prompt = `Write ${name}. ${request}\nReturn one complete code block.${file ? `\nCurrent file:\n${file.text}` : ''}`;
      const room = model.config.context - reserve(model, options);
      if (count(model, prompt) > room) throw new Error(`The whole request/file needs ${count(model, prompt)} tokens; this model has room for ${room}. Use a smaller file or a larger-context model. Nothing changed.`);
      yield { text: `Generating a proposal for ${clean(name)} from the loaded model…\n` };
      let reply = '';
      for await (const text of generate(model, prompt, options, run)) reply += text;
      if (run.cancelled) { yield { text: 'Cancelled; no proposal saved.' }; return; }
      const content = extractCode(reply);
      const syntax = await checkSyntax(name, content, options.python);

      let now = null;
      try { now = await workspace.read(name); } catch (e) { if (e.code !== 'ENOENT') throw e; }
      if ((now?.hash ?? null) !== (file?.hash ?? null)) throw new Error('File changed while generating. No proposal saved.');
      if (run.cancelled) { yield { text: 'Cancelled; no proposal saved.' }; return; }
      const proposal = await workspace.stage(name, content);
      yield { text: '\n' + proposalDiff(proposal) + `\n\nProposal ${proposal.id}\nReview for correctness. /apply ${proposal.id} writes it; /discard ${proposal.id} removes it.\n${syntax.message}` };
      return;
    }
    default: throw new Error('Unknown workspace command.');
  }
  yield { text: clean(result) };
}
