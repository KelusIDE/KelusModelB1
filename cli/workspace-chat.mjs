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
export function isWorkspaceRequest(input) {
  return Boolean(webRequest(input)) || known.has(input.split(/\s/)[0]) || /^(?:please\s+)?(?:analy[sz]e|inspect)\s+(?:(?:the|this|my)\s+)?(?:files|project|workspace|repo)(?:\s+please)?[.!?]*$/i.test(input.trim());
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

/** Explicit tools work even when the model has never learned a tool-call syntax. */
export async function* workspaceReply(workspace, model, input, options, run) {
  const online=webRequest(input);
  if(online) {
    yield {source:'web',omitted:0};
    const controller=new AbortController(), timer=setInterval(()=>{if(run.cancelled)controller.abort();},100);
    try { const result=await webAnswer(online,fetch,controller.signal); if(!run.cancelled)yield {text:clean(result.text)}; }
    finally {clearInterval(timer);}
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
      // A complete replacement must see the complete old file. Never silently truncate it.
      const prompt = `Write ${name}. ${request}\nReturn one complete code block.${file ? `\nCurrent file:\n${file.text}` : ''}`;
      const room = model.config.context - reserve(model, options);
      if (count(model, prompt) > room) throw new Error(`The whole request/file needs ${count(model, prompt)} tokens; this model has room for ${room}. Use a smaller file or a larger-context model. Nothing changed.`);
      yield { text: `Generating a proposal for ${clean(name)} from the loaded model…\n` };
      let reply = '';
      for await (const text of generate(model, prompt, options, run)) reply += text;
      if (run.cancelled) { yield { text: 'Cancelled; no proposal saved.' }; return; }
      const content = extractCode(reply);
      const syntax = await checkSyntax(name, content, options.python);
      // Recheck the source after inference, before storing the proposal.
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
