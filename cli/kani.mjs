#!/usr/bin/env node


import { readFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { emitKeypressEvents } from 'node:readline';
import { fileURLToPath } from 'node:url';
import { Model, enableKernels, prepareReply } from '../site/b1.js';
import { SEARCH_CALL, maybeSearch, searchedAnswer } from '../site/reading.js';
import { PLACES_CALL, maybePlaces, placesAnswer } from '../site/places.js';
import { CIDE_CALL, maybeCide, cideAnswer } from '../site/cide.js';
import { loadCide } from './cide.mjs';
import { webAnswer } from '../site/web.js';
import { Workspace, clean } from './workspace.mjs';
import { WORKSPACE_COMMANDS, isWorkspaceRequest, workspaceReply } from './workspace-chat.mjs';
import { Renderer, SPINNER, banner, box, fit, inputArea, theme, width } from './ui.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const BRAND = 'Kani';
const { version } = JSON.parse(await readFile(path.join(HERE, 'package.json'), 'utf8'));
const BUILDS = JSON.parse(await readFile(path.join(ROOT, 'kani-builds.json'), 'utf8'));
const NAME = BUILDS.find(b => b.id === 'latest')?.label.split(' · ')[0] || BRAND;
const SOURCES = { local_calculator: 'calculator', local_clock: 'clock', crisis_support: 'fixed safety message', sketch: 'Kani Sketch', workspace: 'workspace tools', web: 'online search', painter: 'Kani Painter', pixels: 'Kani Pixels', places: 'places search' };
const COMMANDS = [
  ...WORKSPACE_COMMANDS,
  ['/cide', 'ask Cide directly: /cide TASK'],
  ['/help', 'show commands and keys'],
  ['/new', 'start a new chat'],
  ['/clear', 'clear the screen and start over'],
  ['/info', 'show model details'],
  ['/model', 'list or switch builds, including kani-cide and cide'],
  ['/quit', 'exit'],
];
const SUGGESTIONS = ['analyze files', '/read README.md', '/search TODO', '/help'];
const HELP = `Usage: kani [options] [prompt]

  kani                        chat in the terminal
  kani "explain recursion"    answer one prompt and exit (also reads stdin)

Workspace commands: /files, /analyze, /read, /search, /ask, /draft, /do, /svg, /draw, /diff, /apply, /discard, /vision, /web
Coding specialist: /cide TASK${BUILDS.some(b => b.id === 'kani-cide') ? '; or kani --build kani-cide for learned delegation' : ''}
Example: kani --cwd ./project "/analyze"

Options:
  --cwd DIR          workspace directory (default: current directory)
  --vision-model FILE  scratch vision checkpoint (default: checkpoints/vision-ui.npz)
  --build NAME       latest (${NAME}, default), ${BUILDS.filter(b => b.id !== 'latest').map(b => b.id).join(', ')}
  --painter-dir DIR   exported Kani Painter model (default: training-runs/kani-painter-web)
  --python CMD       Python for optional /vision (default: python3)
  --model-dir DIR    exported model directory instead of a --build
  --max-tokens N     longest reply, 1-4096 (default 512, capped by model context)
  --temperature T    0 always picks the most likely token (default 0)
  -v, --version      print the version
  -h, --help         show this help

Export another checkpoint with:
  python3 -m kelus.export_web --checkpoint checkpoints/NAME.npz --out DIR`;

function parse(argv) {
  const options = { modelDir: path.join(ROOT, 'site', 'model'), maxTokens: 512, temperature: 0, cwd: process.cwd(), visionModel: path.join(ROOT, 'checkpoints/vision-ui.npz'), painterDir: path.join(ROOT, 'training-runs/kani-painter-web'), python: 'python3', engineRoot: ROOT, imageModel: 'painter', build: 'latest', prompt: [] };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const value = () => {
      if (i + 1 >= argv.length) fail(`${arg} needs a value.`);
      return argv[++i];
    };
    if (arg === '-h' || arg === '--help') options.help = true;
    else if (arg === '-v' || arg === '--version') options.version = true;
    else if (arg === '--vision-model') options.visionModel = path.resolve(value());
    else if (arg === '--build') { options.build = value(); if (!BUILDS.some(b => b.id === options.build)) fail('Unknown build. Choose ' + BUILDS.map(b => b.id).join(', ')); }
    else if (arg === '--painter-dir') options.painterDir = path.resolve(value());
    else if (arg === '--python') options.python = value();
    else if (arg === '--cwd') options.cwd = path.resolve(value());
    else if (arg === '--model-dir') { options.modelDir = path.resolve(value()); options.modelDirGiven = true; }
    else if (arg === '--max-tokens') options.cideMaxTokens = options.maxTokens = Number(value());
    else if (arg === '--temperature') options.temperature = Number(value());
    else if (arg.startsWith('-') && arg !== '-') fail(`Unknown option ${arg}. Try kani --help.`);
    else options.prompt.push(arg);
  }
  if (!(Number.isInteger(options.maxTokens) && options.maxTokens >= 1 && options.maxTokens <= 4096)) fail('--max-tokens must be a whole number from 1 to 4096.');
  if (!(options.temperature >= 0 && options.temperature <= 2)) fail('--temperature must be between 0 and 2.');
  return options;
}

function fail(message) {
  process.stderr.write(`kani: ${message}\n`);
  process.exit(1);
}

async function selectBuild(id, options) {
  const build = BUILDS.find(b => b.id === id);
  if (!build) throw new Error('Unknown build. Choose ' + BUILDS.map(b => b.id).join(', '));
  const directory = path.join(ROOT, build.export);
  {

    try { await readFile(path.join(directory, 'b1.json')); }
    catch {
      const { execFile } = await import('node:child_process');
      const { promisify } = await import('node:util');
      await promisify(execFile)(options.python, ['-m', 'kelus.export_web', '--checkpoint',
        path.join(ROOT, 'checkpoints', build.checkpoint + '.npz'), '--out', directory],
        { cwd: ROOT, timeout: 120000, maxBuffer: 1024 * 1024 });
    }
  }
  return { build: id, imageModel: build.imageModel, modelDir: directory, cide: Boolean(build.cide) };
}

async function load(directory) {
  try {
    const manifest = JSON.parse(await readFile(path.join(directory, 'b1.json'), 'utf8'));
    const weights = await readFile(path.join(directory, 'b1.bin'));
    await enableKernels();
    return new Model(manifest, weights.buffer.slice(weights.byteOffset, weights.byteOffset + weights.byteLength));
  } catch (error) {
    if (error.code === 'ENOENT') fail(`No exported model in ${directory}.\nDownload published weights: node "${path.join(ROOT, 'model-host', 'download.mjs')}" latest cide seal-3\nOr export your own: python3 -m kelus.export_web --checkpoint checkpoints/b1-v3.best.npz --out ${directory}`);
    fail(`Could not load the model in ${directory}: ${error.message}`);
  }
}


async function* reply(model, messages, options, run) {
  const latest = messages[messages.length - 1].content;
  if (/^\/cide(?:\s|$)/.test(latest)) {
    const task = latest.slice(5).trim();
    if (!task) throw new Error('Usage: /cide TASK');
    yield { source: 'cide', omitted: 0 };
    for await (const text of cideAnswer([...messages.slice(0, -1), { role: 'user', content: task }],
      { load: loadCide, ...options, maxTokens: options.cideMaxTokens ?? 4096, cancelled: () => run.cancelled, pause: () => new Promise(setImmediate) })) yield { text };
    return;
  }
  if (isWorkspaceRequest(latest)) {
    const coding = options.cide && /^\/(?:do|draft|ask)\s/.test(latest);
    for await (const event of workspaceReply(options.workspace, coding ? await loadCide() : model, latest, options, run))
      yield coding && event.source === 'workspace' ? { ...event, source: 'cide' } : event;
    return;
  }
  if (model.manifest.specialist === 'coding') {
    yield { source: 'cide', omitted: 0 };
    for await (const text of cideAnswer(messages, { load: async () => model, ...options, maxTokens: options.cideMaxTokens ?? 4096,
      cancelled: () => run.cancelled, pause: () => new Promise(setImmediate) })) yield { text };
    return;
  }
  const { retained, omitted, tool } = prepareReply(model, messages, options.maxTokens);
  yield { source: tool ? tool[1] : options.build === 'cide' ? 'cide' : 'model', omitted };
  if (tool) {
    yield { text: tool[0] };
    return;
  }

  let reply = '', held = true;
  for (const text of model.generate(retained, { maxTokens: options.maxTokens, temperature: options.temperature })) {
    if (run.cancelled) return;
    reply += text;
    if (held && !maybeSearch(reply) && !maybePlaces(reply) && !maybeCide(reply)) { held = false; yield { text: reply }; }
    else if (!held) yield { text };
    await new Promise(resolve => setImmediate(resolve));
  }
  if (held && reply.trim() === CIDE_CALL && options.build !== 'cide') {
    yield { source: 'cide', omitted: 0 };
    for await (const text of cideAnswer(messages, { load: loadCide, ...options, maxTokens: options.cideMaxTokens ?? 4096,
      cancelled: () => run.cancelled, pause: () => new Promise(setImmediate) })) yield { text };
    return;
  }
  const place = held && reply.trim().match(PLACES_CALL);
  if (place) {

    yield { source: 'places', omitted: 0 };
    for await (const text of placesAnswer(place, { locate: async () => null })) {
      if (run.cancelled) return;
      yield { text };
    }
    return;
  }
  const call = held && reply.trim().match(SEARCH_CALL);
  if (!call) { if (held && reply) yield { text: reply }; return; }
  const generate = async function* (messages) {
    for (const text of model.generate(messages, { maxTokens: 200, temperature: 0 })) {
      if (run.cancelled) return;
      yield text;
      await new Promise(resolve => setImmediate(resolve));
    }
  };
  for await (const text of searchedAnswer(call[1], latest, { generate, webAnswer })) {
    if (run.cancelled) return;
    yield { text };
  }
}


function conversation(messages) {
  const turns = [];
  for (const message of messages) {
    if (message.role === 'user') {
      if (turns.length && turns[turns.length - 1].role === 'user') turns.pop();
      turns.push({ role: 'user', content: message.content });
    } else if (message.content && turns.length && turns[turns.length - 1].role === 'user') {
      turns.push({ role: 'assistant', content: message.content });
    }
  }
  return turns.slice(-15);
}

function speaker(t, source) {
  const helper = source && source !== 'model'
    ? '  ' + t.color('gold', source === 'cide' ? 'Cide · coding specialist' : source === 'sketch' ? 'Kani Sketch · from-scratch doodle model (345 subjects)' : source === 'pixels' ? 'Kani Pixels v2 · experimental 64×64 diffusion model (174 subjects)'
      : source === 'painter' ? 'Kani Painter · separate from-scratch drawing model'
      : `${SOURCES[source] || source} · ${source==='web'?'retrieved sources':source==='places'?'retrieved places':'local helper'}, not the model`) : '';
  return t.gradient('●') + ' ' + t.bold(t.color('coral', BRAND)) + helper;
}

async function once(model, options) {
  const t = theme();
  const prompt = (options.prompt.length && options.prompt[0] !== '-' ? options.prompt.join(' ') : await readStdin()).trim();
  if (!prompt) fail('Nothing to ask. Pass a prompt or pipe one in.');
  if (prompt === '/help') { console.log(HELP + '\n\n' + COMMANDS.map(([n,d]) => n + '  ' + d).join('\n')); return; }
  const renderer = new Renderer(t, Math.min(process.stdout.columns || 80, 100) - 2, 0);
  try {
    for await (const event of reply(model, [{ role: 'user', content: prompt }], options, {})) {
      if (event.source && event.source !== 'model') process.stdout.write(t.color('gold', `[${SOURCES[event.source] || event.source}] `));
      if (event.text) process.stdout.write(renderer.push(clean(event.text)));
    }
    process.stdout.write(renderer.end() + '\n');
  } catch (error) {
    fail(error.message);
  }
}

async function readStdin() {
  let text = '';
  for await (const chunk of process.stdin) text += chunk;
  return text;
}

async function interactive(model, options) {
  const t = theme();
  const out = process.stdout;
  const write = text => out.write(text);
  const size = () => Math.max(30, Math.min((out.columns || 80) - 1, 100));
  const state = { text: '', cursor: 0, history: [], historyIndex: 0, messages: [], run: null, drawn: null };

  function info() {
    const m = model.manifest;
    const home = os.homedir();
    const root = options.workspace.root;
    const directory = root.startsWith(home + path.sep) ? '~' + root.slice(home.length) : root;
    const inner = size() - 4;
    const row = (label, value) => t.color('muted', label.padEnd(11)) + fit(value, inner - 11);
    return box(t, [
      row('model', `${m.name} · ${m.parameters.toLocaleString()} parameters`),
      row('trained', `${(m.training?.steps || 0).toLocaleString()} steps from random weights`),
      row('engine', 'local inference · /web sends only its query to Wikipedia'),
      row('directory', clean(directory)),
      row('context', `${m.config.context} tokens · /ask includes a bounded source excerpt`),
    ], size(), `>_ ${m.name || NAME}${m.codename ? ` (${m.codename})` : ''}  ${t.color('muted', 'v' + version)}`);
  }

  function welcome() {
    write('\n' + banner(t, out.columns || 80) + '\n\n' + info() + '\n');
    const tips = [];
    for (const tip of SUGGESTIONS) if (6 + [...tips, tip].join(' · ').length <= size()) tips.push(tip);
    write('  ' + t.color('muted', 'Try ') + tips.map(s => t.color('teal', s)).join(t.color('muted', ' · ')) + '\n\n');
  }

  function status() {
    return t.color('mint', '●') + ' ' + t.color('muted', `local · ${Math.round(model.parameterCount / 1000)}K`);
  }

  function suggestions() {
    if (!state.text.startsWith('/') || state.text.includes(' ')) return [];
    return COMMANDS.filter(([name]) => name.startsWith(state.text));
  }


  function clear() {
    if (state.drawn) write((state.drawn.row ? `\x1b[${state.drawn.row}A` : '') + '\r\x1b[J');
    state.drawn = null;
  }

  function draw() {
    clear();
    const area = inputArea(t, { text: state.text, cursor: state.cursor, size: size(), placeholder: `Ask ${BRAND} anything, or type / for commands`, status: status(), suggestions: suggestions() });
    const up = area.lines.length - 1 - area.row;
    write(area.lines.join('\n') + (up ? `\x1b[${up}A` : '') + '\r' + (area.column ? `\x1b[${area.column}C` : ''));
    state.drawn = { row: area.row };
  }

  function edit(text, cursor) {
    state.text = text;
    state.cursor = Math.max(0, Math.min(cursor, [...text].length));
    draw();
  }

  async function command(input) {
    const [name] = input.split(/\s+/);
    const match = COMMANDS.find(([command]) => command === name) || (suggestions().length === 1 && suggestions()[0]);
    const chosen = match ? match[0] : name;
    if (chosen === '/quit' || chosen === '/exit') return quit();
    if (chosen === '/help') {
      write(COMMANDS.map(([n, d]) => '  ' + t.color('coral', n.padEnd(10)) + t.color('muted', d)).join('\n') + '\n');
      write('  ' + t.color('muted', 'keys: ⏎ send · ↑/↓ history · ←/→ move · ctrl+a/e start/end · ctrl+u clear · tab complete · esc stop · ctrl+c quit') + '\n\n');
    } else if (chosen === '/new') {
      state.messages = [];
      state.omittedShown = false;
      write('  ' + t.color('muted', 'Started a new chat.') + '\n\n');
    } else if (chosen === '/clear') {
      state.messages = [];
      state.omittedShown = false;
      write('\x1b[2J\x1b[3J\x1b[H');
      welcome();
    } else if (chosen === '/model') {
      const selected = input.split(/\s+/)[1];
      if (!selected) write(BUILDS.map(b => `  ${b.id === options.build ? '●' : ' '} ${b.id}: ${b.label}`).join('\n') + '\n\n');
      else {
        try {
          const next = await selectBuild(selected, options);
          const loaded = await load(next.modelDir);
          Object.assign(options, next);
          model = loaded;
          state.messages = [];
          state.omittedShown = false;
          write(`  Switched to ${selected}. New conversation started.\n\n`);
        } catch (error) { write(`  Could not switch: ${error.message}\n\n`); }
      }
    } else if (chosen === '/info') {
      write(info() + '\n\n');
    } else {
      write('  ' + t.color('red', `Unknown command ${name}.`) + t.color('muted', ' /help lists them.') + '\n\n');
    }
  }

  async function submit() {
    const prompt = state.text.trim();
    state.text = '';
    state.cursor = 0;
    if (!prompt) return draw();
    if (state.history[state.history.length - 1] !== prompt) state.history.push(prompt);
    state.historyIndex = state.history.length;
    clear();
    write(t.color('mint', '›') + ' ' + t.bold(prompt) + '\n\n');
    if (prompt.startsWith('/') && !/^\/cide(?:\s|$)/.test(prompt) && !isWorkspaceRequest(prompt)) {
      state.run = { cancelled: false };
      try { await command(prompt); } finally { state.run = null; }
      return draw();
    }
    state.messages.push({ role: 'user', content: prompt });
    const turns = conversation(state.messages);
    const answer = { role: 'assistant', content: '' };
    state.messages.push(answer);
    const run = state.run = { cancelled: false };
    write('\x1b[?25l');
    let frame = 0;
    const spin = () => write('\r\x1b[2K' + t.color('teal', SPINNER[frame++ % SPINNER.length]) + ' ' + t.color('muted', isWorkspaceRequest(prompt) ? 'Reading workspace / preparing response…' : `${BRAND} is generating…`));
    spin();
    const spinner = setInterval(spin, 80);
    const renderer = new Renderer(t, size() - 2);
    let source = 'model', omitted = 0, started = false;
    const begin = () => {
      if (started) return;
      started = true;
      clearInterval(spinner);
      write('\r\x1b[2K' + speaker(t, source) + '\n' + renderer.begin());
    };
    try {
      for await (const event of reply(model, turns, options, run)) {
        if (event.source) {
          ({ source, omitted } = event);
          continue;
        }
        begin();
        answer.content += event.text;
        write(renderer.push(clean(event.text)));
      }
      begin();
      write(renderer.end() + '\n');

      const explain = omitted && !state.omittedShown;
      if (explain) state.omittedShown = true;
      const notes = [run.cancelled && 'stopped', explain && `Older turns were omitted to fit this model's context`].filter(Boolean);
      if (notes.length) write('  ' + t.color('muted', notes.join(' · ')) + '\n');
    } catch (error) {
      clearInterval(spinner);
      write('\r\x1b[2K  ' + t.color('red', error.message) + '\n');
      answer.content = '';
    }
    if (isWorkspaceRequest(prompt)) state.messages.splice(-2);
    state.run = null;
    write('\n\x1b[?25h');
    draw();
  }

  function quit() {
    clear();
    if (process.stdin.isTTY) process.stdin.setRawMode(false);
    write('\x1b[?25h  ' + t.color('muted', 'Bye from ') + t.gradient(BRAND) + '\n');
    process.exit(0);
  }

  function key(text, info = {}) {
    const chars = [...state.text];
    const at = state.cursor;
    if (info.ctrl && info.name === 'c') {
      if (state.run) state.run.cancelled = true;
      else if (state.text) edit('', 0);
      else quit();
      return;
    }
    if (state.run) {
      if (info.name === 'escape') state.run.cancelled = true;
      return;
    }
    if (info.ctrl && info.name === 'd' && !state.text) return quit();
    switch (info.ctrl ? 'ctrl+' + info.name : info.name) {
      case 'return': case 'enter': return void submit();
      case 'backspace': return at && edit([...chars.slice(0, at - 1), ...chars.slice(at)].join(''), at - 1);
      case 'delete': return edit([...chars.slice(0, at), ...chars.slice(at + 1)].join(''), at);
      case 'left': return edit(state.text, at - 1);
      case 'right': return edit(state.text, at + 1);
      case 'home': case 'ctrl+a': return edit(state.text, 0);
      case 'end': case 'ctrl+e': return edit(state.text, chars.length);
      case 'ctrl+u': return edit('', 0);
      case 'ctrl+w': {
        const start = chars.slice(0, at).join('').replace(/\S+\s*$/, '').length;
        return edit([...chars.slice(0, start), ...chars.slice(at)].join(''), start);
      }
      case 'ctrl+l': write('\x1b[2J\x1b[3J\x1b[H'); state.drawn = null; welcome(); return draw();
      case 'tab': {
        const [first] = suggestions();
        return first && edit(first[0] + ' ', first[0].length + 1);
      }
      case 'up': case 'down': {
        if (!state.history.length) return;
        state.historyIndex = Math.max(0, Math.min(state.history.length, state.historyIndex + (info.name === 'up' ? -1 : 1)));
        const recalled = state.history[state.historyIndex] || '';
        return edit(recalled, [...recalled].length);
      }
      case 'escape': return;
    }
    if (text && !info.ctrl && !info.meta) {
      const typed = [...text.replace(/[\r\n\t]+/g, ' ')].filter(c => c >= ' ');
      if (typed.length) edit([...chars.slice(0, at), ...typed, ...chars.slice(at)].join(''), at + typed.length);
    }
  }

  welcome();
  emitKeypressEvents(process.stdin);
  process.stdin.setRawMode(true);
  process.stdin.on('keypress', key);
  out.on('resize', () => { if (!state.run) draw(); });
  draw();
}

const options = parse(process.argv.slice(2));
if (options.help) {
  console.log(HELP);
} else if (options.version) {
  console.log(`kani ${version} (${NAME})`);
} else {

  if (!options.modelDirGiven) Object.assign(options, await selectBuild(options.build, options));
  try { options.workspace = await Workspace.open(options.cwd); } catch (error) { fail(error.message); }
  if (options.prompt.length || !process.stdin.isTTY || !process.stdout.isTTY) {
    const prompt = (options.prompt.length && options.prompt[0] !== '-' ? options.prompt.join(' ') : await readStdin()).trim();
    options.prompt = [prompt];
    const toolsOnly = isWorkspaceRequest(prompt) && !/^\/(ask|draft|do|svg)\b/.test(prompt);
    await once(toolsOnly || prompt === '/help' ? null : await load(options.modelDir), options);
  } else await interactive(await load(options.modelDir), options);
}
