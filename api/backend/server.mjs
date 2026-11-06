#!/usr/bin/env node






import crypto from 'node:crypto';
import { readFile } from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Model, conversation, enableKernels } from '../../site/b1.js';
import { answer } from './reply.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const MODELS = JSON.parse(await readFile(path.join(HERE, '../models.json'), 'utf8'));
export const LIMITS = { messages: 64, characters: 8000, total: 32000, maxTokens: 2048, body: 256 * 1024, waiting: 8 };
const DEFAULT_TOKENS = 512;

class ApiError extends Error {
  constructor(status, message, code = 'invalid_request_error') { super(message); this.status = status; this.code = code; }
}


export function parseRequest(body) {
  if (!body || typeof body !== 'object' || Array.isArray(body)) throw new ApiError(400, 'Send a JSON object.');
  const model = MODELS.find(m => m.id === (body.model ?? MODELS[0].id));
  if (!model) throw new ApiError(404, `Unknown model "${String(body.model).slice(0, 80)}". Models: ${MODELS.map(m => m.id).join(', ')}.`, 'model_not_found');
  if (body.tools !== undefined || body.functions !== undefined) throw new ApiError(400, 'Function calling is not supported. Kani runs its own lookups; send "kani_tools": false to get them back as text instead.');
  if (body.n !== undefined && body.n !== 1) throw new ApiError(400, 'Only n = 1 is supported.');
  const { messages } = body;
  if (!Array.isArray(messages) || !messages.length || messages.length > LIMITS.messages) throw new ApiError(400, `"messages" must be a list of 1 to ${LIMITS.messages} messages.`);
  const turns = [];
  let total = 0;
  for (const message of messages) {
    if (!message || typeof message !== 'object' || !['system', 'developer', 'user', 'assistant'].includes(message.role))
      throw new ApiError(400, 'Each message needs a role of "user", "assistant" or "system", and content.');
    let { content } = message;
    if (Array.isArray(content)) {
      if (content.some(part => part?.type !== 'text' || typeof part.text !== 'string')) throw new ApiError(400, 'Kani reads text only: send content as a string or as text parts.');
      content = content.map(part => part.text).join('\n');
    }
    if (typeof content !== 'string') throw new ApiError(400, 'Each message needs text content.');
    if (content.length > LIMITS.characters) throw new ApiError(400, `Each message can be at most ${LIMITS.characters.toLocaleString('en-US')} characters.`);
    total += content.length;
    if (message.role === 'user' || message.role === 'assistant') turns.push({ role: message.role, content });
  }
  if (total > LIMITS.total) throw new ApiError(400, `All messages together can be at most ${LIMITS.total.toLocaleString('en-US')} characters.`);

  const alternating = [];
  for (const turn of turns) {
    const last = alternating.at(-1);
    if (turn.role === 'user') { if (last?.role === 'user') alternating.pop(); alternating.push(turn); }
    else if (last?.role === 'user' && turn.content) alternating.push(turn);
  }
  if (alternating.at(-1)?.role !== 'user' || !alternating.at(-1).content.trim()) throw new ApiError(400, 'The last message must be a non-empty user message.');
  const maxTokens = body.max_completion_tokens ?? body.max_tokens ?? (model.coding ? 2048 : DEFAULT_TOKENS);
  if (!Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > LIMITS.maxTokens) throw new ApiError(400, `"max_tokens" must be a whole number from 1 to ${LIMITS.maxTokens}.`);
  const sampling = {
    temperature: setting(body, 'temperature', 0, 0, 2),
    topP: setting(body, 'top_p', 1, 0, 1, { above: true }),
    topK: setting(body, 'top_k', 30, 1, 200, { whole: true }),
    presencePenalty: setting(body, 'presence_penalty', 0, -2, 2),
    frequencyPenalty: setting(body, 'frequency_penalty', 0, -2, 2),
    repetitionPenalty: setting(body, 'repetition_penalty', 1, 1, 2),
  };
  const seed = body.seed ?? undefined;
  if (seed !== undefined && !Number.isSafeInteger(seed)) throw new ApiError(400, '"seed" must be a whole number.');
  let stop = body.stop ?? [];
  if (typeof stop === 'string') stop = [stop];
  if (!Array.isArray(stop) || stop.length > 4 || stop.some(s => typeof s !== 'string' || !s || s.length > 100)) throw new ApiError(400, '"stop" must be up to 4 non-empty strings.');
  for (const flag of ['stream', 'kani_tools']) if (body[flag] != null && typeof body[flag] !== 'boolean') throw new ApiError(400, `"${flag}" must be true or false.`);
  const usage = body.stream_options?.include_usage ?? false;
  if ((body.stream_options != null && typeof body.stream_options !== 'object') || typeof usage !== 'boolean') throw new ApiError(400, '"stream_options" can only hold "include_usage": true or false.');

  if (body.response_format != null && body.response_format.type !== 'text') throw new ApiError(400, 'Kani writes plain text only: "response_format" can only be {"type": "text"}.');
  if (body.logprobs) throw new ApiError(400, '"logprobs" is not supported.');
  if (body.logit_bias != null && Object.keys(body.logit_bias).length) throw new ApiError(400, '"logit_bias" is not supported.');
  if (body.user != null && (typeof body.user !== 'string' || body.user.length > 256)) throw new ApiError(400, '"user" must be a string of at most 256 characters.');
  return { model, messages: alternating.slice(-31), maxTokens, sampling, seed, stop, stream: Boolean(body.stream), usage, tools: body.kani_tools !== false };
}


function setting(body, name, fallback, low, high, { whole = false, above = false } = {}) {
  const value = body[name] ?? fallback;
  if (typeof value !== 'number' || (whole && !Number.isInteger(value)) || !(above ? value > low : value >= low) || !(value <= high))
    throw new ApiError(400, `"${name}" must be ${whole ? 'a whole number' : 'a number'} ${above ? `above ${low} and at most` : `from ${low} to`} ${high}.`);
  return value;
}


export function seeded(seed) {
  let state = crypto.createHash('sha256').update(String(seed)).digest().readUInt32LE(0);
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}


export class Stopper {
  constructor(stops) { this.stops = stops; this.pending = ''; this.done = false; }
  push(text) {
    if (this.done) return '';
    this.pending += text;
    const hits = this.stops.map(s => this.pending.indexOf(s)).filter(i => i >= 0);
    if (hits.length) { this.done = true; return this.pending.slice(0, Math.min(...hits)); }
    let keep = 0;
    for (const s of this.stops) for (let n = Math.min(s.length - 1, this.pending.length); n > keep; n--) if (this.pending.endsWith(s.slice(0, n))) { keep = n; break; }
    const out = this.pending.slice(0, this.pending.length - keep);
    this.pending = this.pending.slice(out.length);
    return out;
  }
  end() { const rest = this.done ? '' : this.pending; this.pending = ''; return rest; }
}


function queue() {
  let tail = Promise.resolve(), waiting = 0;
  return async () => {
    if (waiting >= LIMITS.waiting) throw new ApiError(503, 'Kani is busy with other requests. Try again in a few seconds.', 'server_busy');
    waiting++;
    let release;
    const turn = new Promise(resolve => { release = resolve; });
    const before = tail;
    tail = before.then(() => turn);
    await before;
    waiting--;
    return release;
  };
}

export function createServer({ secret, root = path.join(HERE, '../../site'), fetcher = fetch, log = console.log } = {}) {
  const loaded = new Map(), acquire = queue();
  const load = id => {
    if (!loaded.has(id)) {
      const entry = MODELS.find(m => m.id === id), directory = path.join(root, entry.directory);
      loaded.set(id, (async () => {
        const [manifest, weights] = await Promise.all([readFile(path.join(directory, 'b1.json'), 'utf8'), readFile(path.join(directory, 'b1.bin'))]);
        await enableKernels();
        return new Model(JSON.parse(manifest), weights.buffer.slice(weights.byteOffset, weights.byteOffset + weights.byteLength));
      })());
      loaded.get(id).catch(() => loaded.delete(id));
    }
    return loaded.get(id);
  };
  const cide = MODELS.find(m => m.coding)?.id;
  const expected = secret && crypto.createHash('sha256').update(secret).digest();
  const authorized = request => {
    if (!expected) return true;
    const given = crypto.createHash('sha256').update((request.headers.authorization || '').replace(/^Bearer /, '')).digest();
    return crypto.timingSafeEqual(given, expected);
  };

  async function complete(request, response) {
    const chunks = [];
    let size = 0;
    for await (const chunk of request) {
      size += chunk.length;
      if (size > LIMITS.body) throw new ApiError(413, 'The request is too large.');
      chunks.push(chunk);
    }
    let body;
    try { body = JSON.parse(Buffer.concat(chunks).toString('utf8')); } catch { throw new ApiError(400, 'The body is not valid JSON.'); }
    const options = parseRequest(body);
    const model = await load(options.model.id);
    const id = 'chatcmpl-' + crypto.randomUUID().replaceAll('-', ''), created = Math.floor(Date.now() / 1000);
    let closed = false;
    response.on('close', () => { closed = true; });
    const release = await acquire();
    const started = Date.now();
    try {
      const sampling = { ...options.sampling, random: options.seed === undefined ? Math.random : seeded(options.seed) };
      const events = answer(model, options.messages, { maxTokens: Math.min(options.maxTokens, model.config.context - 16), sampling,
        tools: options.tools, coding: Boolean(options.model.coding), loadCide: cide && options.model.id !== cide ? () => load(cide) : null,
        cancelled: () => closed, fetcher });
      const stopper = new Stopper(options.stop);
      const chunk = (delta, finish = null, extra = {}) => 'data: ' + JSON.stringify({ id, object: 'chat.completion.chunk', created, model: options.model.id,
        choices: [{ index: 0, delta, finish_reason: finish }], ...extra }) + '\n\n';
      let content = '', source = 'model', streaming = false;
      for await (const event of events) {
        if (closed) return;
        if (event.source) source = event.source;
        if (!event.text) continue;
        const text = stopper.push(event.text);
        content += text;
        if (options.stream) {
          if (!streaming) {
            response.writeHead(200, { 'Content-Type': 'text/event-stream; charset=utf-8', 'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no' });
            response.write(chunk({ role: 'assistant', content: '' }, null, { kani: { source } }));
            streaming = true;
          }
          if (text) response.write(chunk({ content: text }));
        }
        if (stopper.done) break;
      }
      const tail = stopper.end();
      content += tail;
      const completionTokens = model.tokenizer.encode(content).length;
      const promptTokens = conversation(options.messages, model.tokenizer).length;
      const usage = { prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens };
      const finish = !stopper.done && source === 'model' && completionTokens >= options.maxTokens ? 'length' : 'stop';
      log(JSON.stringify({ model: options.model.id, source, stream: options.stream, prompt_tokens: promptTokens, completion_tokens: completionTokens, ms: Date.now() - started }));
      if (options.stream) {
        if (!streaming) {
          response.writeHead(200, { 'Content-Type': 'text/event-stream; charset=utf-8', 'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no' });
          response.write(chunk({ role: 'assistant', content: '' }, null, { kani: { source } }));
        }
        if (tail) response.write(chunk({ content: tail }));

        const ending = options.usage ? chunk({}, finish) + 'data: ' + JSON.stringify({ id, object: 'chat.completion.chunk', created,
          model: options.model.id, choices: [], usage }) + '\n\n' : chunk({}, finish, { usage });
        response.end(ending + 'data: [DONE]\n\n');
      } else {
        send(response, 200, { id, object: 'chat.completion', created, model: options.model.id,
          choices: [{ index: 0, message: { role: 'assistant', content }, finish_reason: finish }], usage, kani: { source } });
      }
    } finally {
      release();
    }
  }

  return http.createServer(async (request, response) => {
    const url = new URL(request.url, 'http://backend');
    try {
      if (request.method === 'GET' && (url.pathname === '/' || url.pathname === '/health')) {
        send(response, 200, { name: 'Kani API backend', models: MODELS.map(m => m.id), docs: 'https://kelus-b1.pages.dev/developers/' });
      } else if (!authorized(request)) {
        throw new ApiError(401, 'This backend only answers the Kani API gateway. Use https://kani-api.chatgpt-ai-5-2o2.workers.dev with an hpk- key.', 'unauthorized');
      } else if (request.method === 'GET' && url.pathname === '/v1/models') {
        send(response, 200, { object: 'list', data: MODELS.map(m => ({ id: m.id, object: 'model', owned_by: 'kelus-corp', name: m.name, about: m.about })) });
      } else if (request.method === 'POST' && url.pathname === '/v1/chat/completions') {
        await complete(request, response);
      } else {
        throw new ApiError(404, 'Not found. Use POST /v1/chat/completions or GET /v1/models.', 'not_found');
      }
    } catch (caught) {

      const error = !(caught instanceof ApiError) && /exceeds B1's context|too long for Cide/.test(caught?.message)
        ? new ApiError(400, "The last message is too long for this model's context. Shorten it.", 'context_length_exceeded') : caught;
      const status = error instanceof ApiError ? error.status : 500;
      if (status === 500) log(JSON.stringify({ error: error.message }));
      const failure = { error: { message: status === 500 ? 'Kani could not answer that. Try again.' : error.message, type: status >= 500 ? 'server_error' : 'invalid_request_error', code: error.code || 'server_error' } };
      if (response.headersSent) response.end('data: ' + JSON.stringify(failure) + '\n\ndata: [DONE]\n\n');
      else send(response, status, failure);
    }
  });
}

function send(response, status, payload) {
  response.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8' });
  response.end(JSON.stringify(payload));
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const secret = process.env.KANI_BACKEND_SECRET;
  if (!secret && !process.argv.includes('--no-auth')) {
    console.error('Set KANI_BACKEND_SECRET (the gateway sends it), or pass --no-auth for local testing.');
    process.exit(1);
  }
  const server = createServer({ secret: secret || null });
  const port = Number(process.env.PORT) || 7860;
  server.listen(port, () => console.log(`Kani API backend on http://localhost:${port}${secret ? '' : ' (no auth: local testing only)'}`));
}
