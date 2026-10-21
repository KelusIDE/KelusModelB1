#!/usr/bin/env node
// kani: terminal chat with the B1 model, using the same JavaScript engine as the web app.
// No Python, NumPy, or npm dependencies; weights come from an export such as site/model/.
import { readFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { emitKeypressEvents } from 'node:readline';
import { fileURLToPath } from 'node:url';
import { Model, prepareReply } from '../site/b1.js';
import { Workspace, clean } from './workspace.mjs';
import { WORKSPACE_COMMANDS, isWorkspaceRequest, workspaceReply } from './workspace-chat.mjs';
import { Renderer, SPINNER, banner, box, fit, inputArea, theme, width } from './ui.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const NAME = 'Kani Seal 3 (Kani B2:10F)';
const BRAND = 'Kani';
const { version } = JSON.parse(await readFile(path.join(HERE, 'package.json'), 'utf8'));
const SOURCES = { local_calculator: 'calculator', local_clock: 'clock', workspace: 'workspace tools', web: 'online Wikipedia' };
const COMMANDS = [
  ...WORKSPACE_COMMANDS,
  ['/help', 'show commands and keys'],
  ['/new', 'start a new chat'],
  ['/clear', 'clear the screen and start over'],
  ['/info', 'show model details'],
  ['/quit', 'exit'],
];
const SUGGESTIONS = ['analyze files', '/read README.md', '/search TODO', '/help'];
const HELP = `Usage: kani [options] [prompt]

  kani                        chat in the terminal
  kani "explain recursion"    answer one prompt and exit (also reads stdin)

Workspace commands: /files, /analyze, /read, /search, /ask, /draft, /do, /svg, /diff, /apply, /discard, /vision, /web
Example: kani --cwd ./project "/analyze"

Options:
  --cwd DIR          workspace directory (default: current directory)
  --vision-model FILE  scratch vision checkpoint (default: checkpoints/vision-ui.npz)
  --python CMD       Python for optional /vision (default: python3)
  --model-dir DIR    exported model directory (default: site/model in this repository)
  --max-tokens N     longest reply, 1-2048 (default 512, capped by model context)
  --temperature T    0 always picks the most likely token (default 0)
  -v, --version      print the version
  -h, --help         show this help

Export another checkpoint with:
  python3 -m kelus.export_web --checkpoint checkpoints/NAME.npz --out DIR`;

function parse(argv) {
  const options = { modelDir: path.join(ROOT, 'site', 'model'), maxTokens: 512, temperature: 0, cwd: process.cwd(), visionModel: path.join(ROOT, 'checkpoints/vision-ui.npz'), python: 'python3', engineRoot: ROOT, prompt: [] };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const value = () => {
      if (i + 1 >= argv.length) fail(`${arg} needs a value.`);
      return argv[++i];
    };
    if (arg === '-h' || arg === '--help') options.help = true;
    else if (arg === '-v' || arg === '--version') options.version = true;
    else if (arg === '--vision-model') options.visionModel = path.resolve(value());
    else if (arg === '--python') options.python = value();
    else if (arg === '--cwd') options.cwd = path.resolve(value());
    else if (arg === '--model-dir') options.modelDir = path.resolve(value());
    else if (arg === '--max-tokens') options.maxTokens = Number(value());
    else if (arg === '--temperature') options.temperature = Number(value());
    else if (arg.startsWith('-') && arg !== '-') fail(`Unknown option ${arg}. Try kani --help.`);
    else options.prompt.push(arg);
  }
  if (!(Number.isInteger(options.maxTokens) && options.maxTokens >= 1 && options.maxTokens <= 2048)) fail('--max-tokens must be a whole number from 1 to 2048.');
  if (!(options.temperature >= 0 && options.temperature <= 2)) fail('--temperature must be between 0 and 2.');
  return options;
}

function fail(message) {
  process.stderr.write(`kani: ${message}\n`);
  process.exit(1);
}

async function load(directory) {
  try {
    const manifest = JSON.parse(await readFile(path.join(directory, 'b1.json'), 'utf8'));
    const weights = await readFile(path.join(directory, 'b1.bin'));
    return new Model(manifest, weights.buffer.slice(weights.byteOffset, weights.byteOffset + weights.byteLength));
  } catch (error) {
    if (error.code === 'ENOENT') fail(`No exported model in ${directory}.\nExport one: python3 -m kelus.export_web --checkpoint checkpoints/b1-v3.best.npz --out ${directory}`);
    fail(`Could not load the model in ${directory}: ${error.message}`);
  }
}

/** Yields { source, omitted } once, then text pieces; pauses between tokens so keys still register. */
async function* reply(model, messages, options, run) {
  const latest = messages[messages.length - 1].content;
  if (isWorkspaceRequest(latest)) {
    yield* workspaceReply(options.workspace, model, latest, options, run);
    return;
  }
  const { retained, omitted, tool } = prepareReply(model, messages, options.maxTokens);
  yield { source: tool ? tool[1] : 'model', omitted };
  if (tool) {
    yield { text: tool[0] };
    return;
  }
  for (const text of model.generate(retained, { maxTokens: options.maxTokens, temperature: options.temperature })) {
    if (run.cancelled) return;
    yield { text };
    await new Promise(resolve => setImmediate(resolve));
  }
}

/** Alternating turns for the model; a turn whose reply failed is dropped. */
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
    ? '  ' + t.color('gold', `${SOURCES[source] || source} · ${source==='web'?'retrieved sources':'local helper'}, not the model`) : '';
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
    ], size(), `>_ ${NAME}  ${t.color('muted', 'v' + version)}`);
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

  // The input area is redrawn in place: move up to its first line, clear below, draw again.
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

  function command(input) {
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
    if (prompt.startsWith('/') && !isWorkspaceRequest(prompt)) {
      command(prompt);
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
      // Report bounded history once per chat.
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
      return; // typing waits until the reply finishes
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
  try { options.workspace = await Workspace.open(options.cwd); } catch (error) { fail(error.message); }
  if (options.prompt.length || !process.stdin.isTTY || !process.stdout.isTTY) {
    const prompt = (options.prompt.length && options.prompt[0] !== '-' ? options.prompt.join(' ') : await readStdin()).trim();
    options.prompt = [prompt];
    const toolsOnly = isWorkspaceRequest(prompt) && !/^\/(ask|draft|do|svg)\b/.test(prompt);
    await once(toolsOnly || prompt === '/help' ? null : await load(options.modelDir), options);
  } else await interactive(await load(options.modelDir), options);
}
