'use strict';
// Runs `python -m kelus.cli stdio` and speaks its JSON-lines protocol.
// No VS Code dependency, so tests can drive it with plain Node.
const { spawn } = require('child_process');
const { EventEmitter } = require('events');
const readline = require('readline');

function explain(stderr, python) {
  if (/No module named ['"]?numpy/.test(stderr)) return `NumPy is not installed for ${python}. Run: ${python} -m pip install numpy`;
  if (/No module named ['"]?kelus/.test(stderr)) return 'The repository folder does not contain the kelus package.';
  const lines = stderr.trim().split('\n').filter(Boolean);
  return lines[lines.length - 1] || '';
}

class Bridge extends EventEmitter {
  constructor({ python, cwd, model, checkpoint }) {
    super();
    Object.assign(this, { python, cwd, model, checkpoint });
    this.state = 'stopped';
    this.info = null;
    this.error = null;
    this.child = null;
    this.inflight = new Set();
  }

  start() {
    this.stop();
    this.setStatus('starting');
    const args = ['-m', 'kelus.cli', 'stdio'];
    if (this.checkpoint) args.push('--checkpoint', this.checkpoint);
    else if (this.model) args.push('--model', this.model);
    const child = spawn(this.python, args, { cwd: this.cwd, env: { ...process.env, PYTHONUNBUFFERED: '1' } });
    this.child = child;
    let stderr = '';
    child.stderr.on('data', chunk => { stderr = (stderr + chunk).slice(-4000); });
    child.on('error', error => {
      if (this.child === child) this.fail(error.code === 'ENOENT' ? `Python was not found: ${this.python}` : error.message);
    });
    readline.createInterface({ input: child.stdout }).on('line', line => {
      if (this.child !== child) return;
      let event;
      try { event = JSON.parse(line); } catch { return; }
      if (event.type === 'ready') {
        this.info = event;
        this.setStatus('ready');
      } else if (event.type === 'error' && event.id === undefined) {
        this.fail(event.message);
      } else {
        if (event.type === 'done' || event.type === 'error') this.inflight.delete(event.id);
        this.emit('event', event);
      }
    });
    child.on('exit', (code, signal) => {
      if (this.child !== child) return;
      this.child = null;
      if (this.state !== 'error') this.fail(explain(stderr, this.python) || `The model process exited (${signal || code}).`);
    });
  }

  setStatus(state, error = null) {
    this.state = state;
    this.error = error;
    this.emit('status', { state, info: this.info, error });
  }

  fail(message) {
    for (const id of this.inflight) this.emit('event', { type: 'error', id, message });
    this.inflight.clear();
    this.setStatus('error', message);
  }

  chat(id, messages, options = {}) {
    if (!this.child || this.state !== 'ready') return false;
    this.inflight.add(id);
    this.write({ type: 'chat', id, messages, max_tokens: options.maxTokens ?? 240, temperature: options.temperature ?? 0 });
    return true;
  }

  cancel(id) {
    if (this.child && this.inflight.has(id)) this.write({ type: 'cancel', id });
  }

  write(request) {
    this.child.stdin.write(JSON.stringify(request) + '\n');
  }

  stop() {
    const child = this.child;
    this.child = null;
    this.inflight.clear();
    if (child) {
      child.stdin.end();
      child.kill();
    }
    this.state = 'stopped';
  }
}

module.exports = { Bridge, explain };
