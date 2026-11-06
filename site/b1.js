


import { webRequest } from './web.js';
import { supportFollowUp, supportReply } from './support.js';
import { KERNELS, KERNELS_FMA } from './kernels.js';
import { arithmeticExpression } from './arithmetic.js';

export const PAD = 256, BOS = 257, EOS = 258, USER = 259, ASSISTANT = 260;

const SPACE = '\\t\\n\\v\\f\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000';
const REST = ` ?[A-Za-z]+| ?[0-9]+|[${SPACE}]+|[^A-Za-z0-9${SPACE}]`;

const LATIN = 'A-Za-z\\u00c0-\\u00d6\\u00d8-\\u00f6\\u00f8-\\u024f\\u1e00-\\u1eff', HANGUL = '\\uac00-\\ud7a3\\u1100-\\u11ff\\u3130-\\u318f';
const KANA = '\\u3040-\\u309f\\u30a0-\\u30ff\\u31f0-\\u31ff', HAN = '\\u3400-\\u4dbf\\u4e00-\\u9fff\\uf900-\\ufaff';
const PATTERNS = { 1: new RegExp(REST, 'gu'), 2: new RegExp('```[A-Za-z]*|' + REST, 'gu'),
  3: new RegExp(`\`\`\`[A-Za-z]*| ?[${LATIN}]+| ?[${HANGUL}]+|[${KANA}]+|[${HAN}]+| ?[0-9]+|[${SPACE}]+|[^${SPACE}]`, 'gu') };
const utf8 = new TextEncoder();

export class Tokenizer {
  constructor(spec) {
    this.kind = spec.kind || 'bytes';
    this.pieces = Array.from({ length: 256 }, (_, i) => Uint8Array.of(i));
    for (let i = 0; i < 5; i++) this.pieces.push(new Uint8Array(0));
    this.ranks = new Map();
    this.cache = new Map();
    if (this.kind === 'bpe') {
      this.pattern = PATTERNS[spec.version || 1];
      if (!this.pattern) throw new Error('Unknown BPE pattern version');
      spec.merges.forEach(([a, b], rank) => {
        this.ranks.set(a * 65536 + b, [rank, this.pieces.length]);
        const piece = new Uint8Array(this.pieces[a].length + this.pieces[b].length);
        piece.set(this.pieces[a]); piece.set(this.pieces[b], this.pieces[a].length);
        this.pieces.push(piece);
      });
    } else if (this.kind !== 'bytes') {
      throw new Error('Unknown tokenizer');
    }
    this.vocabSize = this.pieces.length;
  }

  encodePiece(piece) {
    const cached = this.cache.get(piece);
    if (cached) return cached;
    let sequence = Array.from(utf8.encode(piece));
    while (sequence.length > 1) {
      let best = null;
      for (let i = 0; i + 1 < sequence.length; i++) {
        const entry = this.ranks.get(sequence[i] * 65536 + sequence[i + 1]);
        if (entry && (!best || entry[0] < best.rank)) best = { rank: entry[0], token: entry[1], a: sequence[i], b: sequence[i + 1] };
      }
      if (!best) break;
      const merged = [];
      for (let i = 0; i < sequence.length;) {
        if (i + 1 < sequence.length && sequence[i] === best.a && sequence[i + 1] === best.b) { merged.push(best.token); i += 2; }
        else merged.push(sequence[i++]);
      }
      sequence = merged;
    }
    if (this.cache.size < 16384) this.cache.set(piece, sequence);
    return sequence;
  }

  encode(text) {
    if (typeof text !== 'string') throw new Error('Message content must be text');
    if (this.kind === 'bytes') return Array.from(utf8.encode(text));
    const tokens = [];
    for (const match of text.matchAll(this.pattern)) tokens.push(...this.encodePiece(match[0]));
    return tokens;
  }

  tokenBytes(token) {
    if (!(token >= 0 && token < this.vocabSize)) throw new Error('Invalid token');
    return this.pieces[token];
  }
}






export function looping(generated, repeats = 4, least = 32, longest = 32) {
  for (let period = 1; period <= longest; period++) {
    const span = period * Math.max(repeats, Math.ceil(least / period));
    if (span > generated.length) break;
    let periodic = true;
    for (let i = generated.length - span + period; i < generated.length && periodic; i++) periodic = generated[i] === generated[i - period];
    if (periodic) return true;
  }
  return false;
}








export class Repeats {
  constructor(least = 4, start = 8) {
    this.least = least;
    this.start = start;
    this.said = new Set();
    this.starts = new Set();
    this.sentence = '';
    this.line = '';
    this.held = '';
    this.fenced = false;
    this.stopped = false;
  }

  static plain(text) {
    return text.toLowerCase().replace(/^\s*(?:\d+[.)]?|[-*•])(?:\s+|$)/, '').replace(/[^\p{L}\p{N}']+/gu, ' ').trim();
  }


  push(text) {
    let shown = '';
    for (const char of text) {
      if (this.stopped) break;
      this.held += char;
      this.sentence += char;
      this.line += char;
      const newline = char === '\n';
      if (this.fenced) {
        shown += this.held;
        this.held = this.sentence = '';
      } else {
        const plain = Repeats.plain(this.sentence), words = plain.split(' ');
        const ended = plain !== '' && (newline || /[.!?]["')\]]?\s$/.test(this.sentence));

        const opening = words.slice(0, this.start).join(' ');
        if ((ended && words.length >= this.least && this.said.has(plain)) || (words.length > this.start && this.starts.has(opening))) {
          this.stopped = true;
          this.held = '';
          break;
        }
        if (ended && words.length >= this.least) {
          this.said.add(plain);
          if (words.length >= this.start) this.starts.add(opening);
        }
        if (ended || newline || ![...this.said].some(said => said.startsWith(plain))) {

          const [, text, space] = this.held.match(/^([\s\S]*?)(\s*)$/);
          shown += text;
          this.held = space;
        }
        if (ended || newline) this.sentence = '';
      }
      if (newline) {
        if (/^\s*```/.test(this.line)) this.fenced = !this.fenced;
        this.line = '';
      }
    }
    return shown;
  }


  finish() {
    const rest = this.stopped ? '' : this.held;
    this.held = '';
    return rest;
  }
}





export function pickToken(logits, temperature = 0, topK = 30, random = Math.random, topP = 1) {
  for (const blocked of [PAD, BOS, USER, ASSISTANT]) logits[blocked] = -Infinity;
  let token = 0;
  if (temperature === 0) {
    for (let i = 1; i < logits.length; i++) if (logits[i] > logits[token]) token = i;
    return token;
  }
  const top = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]).slice(0, topK);
  const highest = logits[top[0]] / temperature;
  const weights = top.map(i => Math.exp(logits[i] / temperature - highest));
  let total = weights.reduce((a, b) => a + b, 0);
  if (topP < 1) {
    let kept = 0, held = 0;
    while (kept < weights.length && held < topP * total) held += weights[kept++];
    top.length = weights.length = kept;
    total = held;
  }
  let pick = random() * total;
  token = top[top.length - 1];
  for (let i = 0; i < top.length; i++) { pick -= weights[i]; if (pick <= 0) { token = top[i]; break; } }
  return token;
}





export function penalize(logits, counts, { presencePenalty = 0, frequencyPenalty = 0, repetitionPenalty = 1 } = {}) {
  for (const [token, count] of counts) {
    let logit = logits[token];
    if (repetitionPenalty !== 1) logit = logit > 0 ? logit / repetitionPenalty : logit * repetitionPenalty;
    logits[token] = logit - presencePenalty - frequencyPenalty * count;
  }
}

export function conversation(messages, tokenizer) {
  if (!Array.isArray(messages) || !messages.length) throw new Error('Expected a nonempty messages list');
  const tokens = [BOS];
  let expected = 'user';
  for (const message of messages) {
    if (!message || message.role !== expected) throw new Error('Messages must alternate user and assistant, beginning with user');
    tokens.push(expected === 'user' ? USER : ASSISTANT, ...tokenizer.encode(message.content));
    if (expected === 'assistant') tokens.push(EOS);
    expected = expected === 'user' ? 'assistant' : 'user';
  }
  if (expected !== 'assistant') throw new Error('A generation prompt must end with a user message');
  tokens.push(ASSISTANT);
  return tokens;
}



let kernels;




export async function enableKernels({ fused = true } = {}) {
  if (kernels === undefined) {
    kernels = null;
    for (const build of fused ? [KERNELS_FMA, KERNELS] : [KERNELS]) {
      try { kernels = await Promise.all(build.map(source => WebAssembly.compile(Uint8Array.from(atob(source), c => c.charCodeAt(0))))); break; }
      catch {  }
    }
  }
  return Boolean(kernels);
}

export function disableKernels() { kernels = undefined; }

const BATCH = 8;

export class Model {
  constructor(manifest, buffer) {
    if (manifest.format !== 'kelus-b1-web' || manifest.version !== 1) throw new Error('Unsupported model files');
    this.manifest = manifest;
    this.config = { positions: 'learned', tie_embeddings: false, ...manifest.config };
    this.tokenizer = new Tokenizer(manifest.tokenizer);
    const { context, width, heads, vocab_size: vocab } = this.config;
    if (this.tokenizer.vocabSize !== vocab || width % heads) throw new Error('Model and tokenizer do not match');
    if (kernels) {



      const vector = (Math.max(4 * width, vocab) * 4 + 63) & ~63, batch = Math.max(vector, BATCH * 4 * width * 4), layers = this.config.layers;
      const input = buffer.byteLength + (-buffer.byteLength & 63), output = input + batch, query = output + batch;
      const scores = query + vector, mixed = scores + ((context * 4 + 63) & ~63), keys = mixed + vector;
      const values = keys + layers * context * width * 4, end = values + layers * context * width * 4;
      const memory = new WebAssembly.Memory({ initial: Math.ceil(end / 65536) });
      new Uint8Array(memory.buffer).set(new Uint8Array(buffer));
      const run = Object.assign({}, ...kernels.map(module => new WebAssembly.Instance(module, { env: { memory } }).exports));
      this.kernel = { run, f32: new Float32Array(memory.buffer),
        input, output, query, scores, mixed, keys, values, owner: null };
      buffer = memory.buffer;
    }
    const data = new Float32Array(buffer, 0, Math.floor(buffer.byteLength / 4));
    this.weights = {};
    for (const tensor of manifest.tensors) {
      const size = tensor.shape.reduce((a, b) => a * b, 1);
      if (tensor.dtype === 'int8') {

        const byRow = tensor.scale_axis === 1, scales = tensor.shape[byRow ? 0 : 1];
        if (tensor.byte_offset + size > buffer.byteLength || tensor.scale + scales > data.length) throw new Error('Model weights are incomplete');
        this.weights[tensor.name] = { int8: new Int8Array(buffer, tensor.byte_offset, size), scale: data.subarray(tensor.scale, tensor.scale + scales), byRow };
        continue;
      }
      if (tensor.offset + size > data.length) throw new Error('Model weights are incomplete');
      this.weights[tensor.name] = data.subarray(tensor.offset, tensor.offset + size);
    }
    this.headWidth = width / heads;
    if (this.config.positions === 'rope') {
      const half = this.headWidth / 2;
      this.cos = new Float64Array(context * this.headWidth);
      this.sin = new Float64Array(context * this.headWidth);
      for (let position = 0; position < context; position++) {
        for (let i = 0; i < half; i++) {
          const angle = position * 10000 ** (-i / half), base = position * this.headWidth;
          this.cos[base + i] = this.cos[base + i + half] = Math.fround(Math.cos(angle));
          this.sin[base + i] = this.sin[base + i + half] = Math.fround(Math.sin(angle));
        }
      }
    }
  }

  get parameterCount() { return this.manifest.parameters; }

  norm(x, name) {
    const gain = this.weights[name + '.gain'], bias = this.weights[name + '.bias'], n = x.length;
    let mean = 0, variance = 0;
    for (let i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (let i = 0; i < n; i++) variance += (x[i] - mean) ** 2;
    const scale = (variance / n + 1e-5) ** -0.5, out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * scale * gain[i] + bias[i];
    return out;
  }


  multiply(x, matrix, columns) {
    const kernel = this.kernel;
    if (!kernel || columns % 32) return Model.multiply(x, matrix, columns);
    kernel.f32.set(x, kernel.input / 4);
    if (matrix.int8) kernel.run.matvec_i8(kernel.input, matrix.int8.byteOffset, matrix.scale.byteOffset, kernel.output, x.length, columns);
    else kernel.run.matvec_f32(kernel.input, matrix.byteOffset, kernel.output, x.length, columns);
    return kernel.f32.slice(kernel.output / 4, kernel.output / 4 + columns);
  }





  multiplyAll(xs, matrix, columns) {
    const kernel = this.kernel, rows = xs[0].length;
    if (!kernel || xs.length !== BATCH || columns % 32) return xs.map(x => this.multiply(x, matrix, columns));
    xs.forEach((x, t) => kernel.f32.set(x, kernel.input / 4 + t * rows));
    if (matrix.int8) kernel.run.matmul8_i8(kernel.input, matrix.int8.byteOffset, matrix.scale.byteOffset, kernel.output, rows, columns);
    else kernel.run.matmul8_f32(kernel.input, matrix.byteOffset, kernel.output, rows, columns);
    return xs.map((_, t) => kernel.f32.slice(kernel.output / 4 + t * columns, kernel.output / 4 + (t + 1) * columns));
  }


  static multiply(x, matrix, columns) {
    const out = new Float64Array(columns);
    if (matrix.int8) {
      const values = matrix.int8, scale = matrix.scale;
      for (let i = 0; i < x.length; i++) {
        const value = x[i], row = i * columns;
        if (value) for (let j = 0; j < columns; j++) out[j] += value * values[row + j];
      }
      for (let j = 0; j < columns; j++) out[j] *= scale[j];
      return out;
    }
    for (let i = 0; i < x.length; i++) {
      const value = x[i], row = i * columns;
      if (value) for (let j = 0; j < columns; j++) out[j] += value * matrix[row + j];
    }
    return out;
  }


  static row(table, index, width) {
    if (!table.int8) return Float64Array.from(table.subarray(index * width, index * width + width));
    const out = new Float64Array(width), scale = table.scale[index];
    for (let i = 0, start = index * width; i < width; i++) out[i] = table.int8[start + i] * scale;
    return out;
  }

  rotate(vector, position) {
    const width = this.headWidth, half = width / 2, base = position * width;
    for (let head = 0; head < vector.length; head += width) {
      const original = vector.slice(head, head + width);
      for (let i = 0; i < width; i++) {
        const turned = i < half ? -original[i + half] : original[i - half];
        vector[head + i] = original[i] * this.cos[base + i] + turned * this.sin[base + i];
      }
    }
  }


  attend(q, keys, values, layer, position, kernel) {
    const { context, width: d, heads } = this.config, hw = this.headWidth;
    if (kernel) {
      const f32 = kernel.f32, count = position + 1, weights = new Float64Array(count), layerBytes = layer * context * d * 4;
      f32.set(q, kernel.query / 4);
      for (let head = 0; head < heads; head++) {
        const offset = head * hw * 4;
        kernel.run.head_scores(kernel.query + offset, kernel.keys + layerBytes + offset, kernel.scores, count, hw, d);
        let highest = -Infinity, total = 0;
        for (let p = 0; p < count; p++) { weights[p] = f32[kernel.scores / 4 + p] * hw ** -0.5; if (weights[p] > highest) highest = weights[p]; }
        for (let p = 0; p < count; p++) { weights[p] = Math.exp(weights[p] - highest); total += weights[p]; }
        for (let p = 0; p < count; p++) f32[kernel.scores / 4 + p] = weights[p] / total;
        kernel.run.head_mix(kernel.scores, kernel.values + layerBytes + offset, kernel.mixed + offset, count, hw, d);
      }
      return f32.slice(kernel.mixed / 4, kernel.mixed / 4 + d);
    }
    const mixed = new Float64Array(d), scores = new Float64Array(position + 1);
    for (let head = 0; head < heads; head++) {
      const offset = head * hw;
      let highest = -Infinity;
      for (let p = 0; p <= position; p++) {
        let score = 0;
        for (let i = 0; i < hw; i++) score += q[offset + i] * keys[p * d + offset + i];
        scores[p] = score * hw ** -0.5;
        if (scores[p] > highest) highest = scores[p];
      }
      let total = 0;
      for (let p = 0; p <= position; p++) { scores[p] = Math.exp(scores[p] - highest); total += scores[p]; }
      for (let p = 0; p <= position; p++) {
        const weight = scores[p] / total;
        for (let i = 0; i < hw; i++) mixed[offset + i] += weight * values[p * d + offset + i];
      }
    }
    return mixed;
  }


  nextLogits(tokens, cache) {
    const { context, width: d, layers, vocab_size: vocab } = this.config, w = this.weights, hw = this.headWidth;
    const start = cache.length || 0;
    if (!(tokens.length > start && tokens.length <= context)) throw new Error('Expected new tokens within the context limit');
    const kernel = this.kernel && hw % 4 === 0 ? this.kernel : null;
    if (kernel && kernel.owner !== cache) {

      if (start) { cache.length = 0; delete cache.keys; return this.nextLogits(tokens, cache); }
      kernel.owner = cache;
    }
    if (!cache.keys) {

      const layer = context * d * 4;
      cache.keys = Array.from({ length: layers }, (_, i) => kernel ? new Float32Array(kernel.f32.buffer, kernel.keys + i * layer, context * d) : new Float32Array(context * d));
      cache.values = Array.from({ length: layers }, (_, i) => kernel ? new Float32Array(kernel.f32.buffer, kernel.values + i * layer, context * d) : new Float32Array(context * d));
    }
    let x;
    for (let position = start; position < tokens.length;) {

      const count = kernel && tokens.length - position >= BATCH ? BATCH : 1;
      const positions = Array.from({ length: count }, (_, t) => position + t);
      const xs = positions.map(p => {
        const token = tokens[p];
        if (!(token >= 0 && token < vocab)) throw new Error('Invalid token ID');
        const row = Model.row(w.token, token, d);
        if (this.config.positions === 'learned') { const learned = Model.row(w.position, p, d); for (let i = 0; i < d; i++) row[i] += learned[i]; }
        return row;
      });
      for (let layer = 0; layer < layers; layer++) {
        const prefix = `block${layer}.`, normalized = xs.map(x => this.norm(x, prefix + 'norm1'));
        const q = this.multiplyAll(normalized, w[prefix + 'q'], d);
        const k = this.multiplyAll(normalized, w[prefix + 'k'], d);
        const v = this.multiplyAll(normalized, w[prefix + 'v'], d);
        const keys = cache.keys[layer], values = cache.values[layer];
        positions.forEach((p, t) => {
          if (this.config.positions === 'rope') { this.rotate(q[t], p); this.rotate(k[t], p); }
          keys.set(k[t], p * d); values.set(v[t], p * d);
        });

        const projected = this.multiplyAll(positions.map((p, t) => this.attend(q[t], keys, values, layer, p, kernel)), w[prefix + 'projection'], d);
        xs.forEach((x, t) => { for (let i = 0; i < d; i++) x[i] += projected[t][i]; });
        const hidden = this.multiplyAll(xs.map(x => this.norm(x, prefix + 'norm2')), w[prefix + 'up'], 4 * d);
        for (const h of hidden) {
          for (let i = 0; i < h.length; i++) {
            const value = h[i];
            h[i] = 0.5 * value * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (value + 0.044715 * value ** 3)));
          }
        }
        const down = this.multiplyAll(hidden, w[prefix + 'down'], d);
        xs.forEach((x, t) => { for (let i = 0; i < d; i++) x[i] += down[t][i]; });
      }
      x = xs[count - 1];
      position += count;
    }
    cache.length = tokens.length;
    const final = this.norm(x, 'final');
    if (!this.config.tie_embeddings) return this.multiply(final, w.output, vocab);
    if (this.kernel && d % 16 === 0) {
      const kernel = this.kernel, table = w.token;
      kernel.f32.set(final, kernel.input / 4);
      if (table.int8) kernel.run.rows_dot_i8(kernel.input, table.int8.byteOffset, table.scale.byteOffset, kernel.output, vocab, d);
      else kernel.run.rows_dot_f32(kernel.input, table.byteOffset, kernel.output, vocab, d);
      return kernel.f32.slice(kernel.output / 4, kernel.output / 4 + vocab);
    }
    const logits = new Float64Array(vocab), table = w.token.int8 || w.token;
    for (let token = 0; token < vocab; token++) {
      let total = 0;
      for (let i = 0, row = token * d; i < d; i++) total += final[i] * table[row + i];
      logits[token] = w.token.int8 ? total * w.token.scale[token] : total;
    }
    return logits;
  }


  preparePrompt(messages, reserve = 64) {
    const context = this.config.context;
    let retained = [...messages], tokens = conversation(retained, this.tokenizer);
    reserve = Math.min(reserve, Math.max(1, Math.floor(context / 3)));
    while (tokens.length + reserve > context && retained.length > 1) {
      retained = retained.slice(2);
      tokens = conversation(retained, this.tokenizer);
    }
    if (tokens.length > context - Math.max(2, Math.min(16, Math.floor(context / 4)))) {
      throw new Error("This message exceeds B1's context. Shorten it or start with a smaller code example.");
    }
    return { tokens, omitted: messages.length - retained.length };
  }








  *generate(messages, { maxTokens = 240, temperature = 0, topK = 30, topP = 1, random = Math.random, prefix = '',
    presencePenalty = 0, frequencyPenalty = 0, repetitionPenalty = 1, repeats = true } = {}) {
    if (!(maxTokens >= 1 && maxTokens <= 4096 && temperature >= 0 && temperature <= 2 && topK >= 1 && topP > 0 && topP <= 1 && repetitionPenalty > 0))
      throw new Error('Invalid generation settings');
    const penalties = presencePenalty || frequencyPenalty || repetitionPenalty !== 1 ? { presencePenalty, frequencyPenalty, repetitionPenalty } : null;
    const counts = new Map();

    const said = repeats ? new Repeats() : null, show = text => said ? said.push(text) : text;
    const { tokens } = this.preparePrompt(messages, maxTokens);
    const decoder = new TextDecoder('utf-8'), cache = {};
    for (const token of prefix ? this.tokenizer.encode(prefix).slice(0, -1) : []) {
      if (tokens.length >= this.config.context - 1) break;
      tokens.push(token);
      const text = show(decoder.decode(this.tokenizer.tokenBytes(token), { stream: true }));
      if (text) yield text;
    }
    const steps = Math.min(maxTokens, this.config.context - tokens.length), generated = [];
    for (let step = 0; step < steps; step++) {
      const logits = this.nextLogits(tokens, cache);
      if (penalties) penalize(logits, counts, penalties);
      const token = pickToken(logits, temperature, topK, random, topP);
      if (token === EOS) break;
      tokens.push(token);
      generated.push(token);
      counts.set(token, (counts.get(token) ?? 0) + 1);
      const text = show(decoder.decode(this.tokenizer.tokenBytes(token), { stream: true }));
      if (text) yield text;
      if (said?.stopped || looping(generated)) break;
    }
    const tail = show(decoder.decode()) + (said ? said.finish() : '');
    if (tail) yield tail;
  }
}



class CalculatorError extends Error {}
const SYNTAX = 'syntax', ZERO = 'zero';
const OPERATORS = { '+': (a, b) => a + b, '-': (a, b) => a - b, '*': (a, b) => a * b };

function parseExpression(text) {
  const tokens = [];
  const lexer = new RegExp(`([${SPACE}]+)|(\\d+\\.?\\d*|\\.\\d+)|(\\/\\/|\\*\\*|[-+*/%()])`, 'gy');
  let match;
  while (lexer.lastIndex < text.length) {
    if (!(match = lexer.exec(text))) throw SYNTAX;
    if (match[2]) {
      if (/^0\d*[1-9]/.test(match[2]) && !match[2].includes('.')) throw SYNTAX;
      tokens.push({ number: Number(match[2]), integer: !match[2].includes('.') });
    } else if (match[3]) tokens.push({ op: match[3] });
  }
  let index = 0;
  const peek = () => tokens[index], take = () => tokens[index++];
  const expression = () => {
    let node = term();
    while (peek()?.op === '+' || peek()?.op === '-') node = { op: take().op, left: node, right: term() };
    return node;
  };
  const term = () => {
    let node = factor();
    while (['*', '/', '//', '%'].includes(peek()?.op)) node = { op: take().op, left: node, right: factor() };
    return node;
  };
  const factor = () => {
    if (peek()?.op === '+' || peek()?.op === '-') return { unary: take().op, operand: factor() };
    const base = atom();
    if (peek()?.op === '**') { take(); return { op: '**', left: base, right: factor() }; }
    return base;
  };
  const atom = () => {
    const token = take();
    let node;
    if (token?.number !== undefined) node = { value: token.number, integer: token.integer };
    else if (token?.op === '(') {
      if (peek()?.op === ')') throw SYNTAX;
      node = expression();
      if (take()?.op !== ')') throw SYNTAX;
    } else throw SYNTAX;
    while (peek()?.op === '(') {
      take(); const args = peek()?.op === ')' ? [] : [expression()];
      if (take()?.op !== ')') throw SYNTAX;
      node = { call: node, args };
    }
    return node;
  };
  const tree = expression();
  if (index !== tokens.length) throw SYNTAX;
  return tree;
}

function countNodes(node) {
  if (node.value !== undefined) return 1;
  if (node.unary) return 2 + countNodes(node.operand);
  if (node.call) return 1 + countNodes(node.call) + node.args.reduce((total, arg) => total + countNodes(arg), 0);
  return 2 + countNodes(node.left) + countNodes(node.right);
}

function evaluate(node) {
  let result;
  if (node.value !== undefined) result = { value: node.value, integer: node.integer };
  else if (node.unary) {
    const operand = evaluate(node.operand);
    result = { value: (node.unary === '-' ? -1 : 1) * operand.value, integer: operand.integer };
  } else if (node.op && node.op !== '**') {
    const left = evaluate(node.left), right = evaluate(node.right), integer = left.integer && right.integer;
    if (['/', '//', '%'].includes(node.op) && right.value === 0) throw ZERO;
    if (node.op === '/') result = { value: left.value / right.value, integer: false };
    else if (node.op === '//') result = { value: Math.floor(left.value / right.value), integer };
    else if (node.op === '%') {
      let remainder = left.value % right.value;
      if (remainder !== 0 && (remainder < 0) !== (right.value < 0)) remainder += right.value;
      result = { value: remainder, integer };
    } else result = { value: OPERATORS[node.op](left.value, right.value), integer };
  } else throw new CalculatorError('That operator is not supported by the local calculator.');
  if (!Number.isFinite(result.value) || Math.abs(result.value) > 1e12) throw new CalculatorError('Keep numbers and results within one trillion.');
  return result;
}

function formatNumber(value) {
  if (value === Math.trunc(value)) return String(Math.trunc(value) + 0);
  const [mantissa, exponentText] = value.toExponential(11).split('e'), exponent = Number(exponentText);
  const strip = text => text.includes('.') ? text.replace(/0+$/, '').replace(/\.$/, '') : text;
  if (exponent < -4 || exponent >= 12) return `${strip(mantissa)}e${exponent < 0 ? '-' : '+'}${String(Math.abs(exponent)).padStart(2, '0')}`;
  return strip(value.toFixed(11 - exponent));
}

export function calculate(expression) {
  if (expression.length > 100 || !new RegExp(`^[0-9.${SPACE}+*/%()\\-]+$`).test(expression)) {
    throw new CalculatorError('Use numbers and +, -, *, /, //, %, or parentheses.');
  }
  const tree = parseExpression(expression.trim());
  if (1 + countNodes(tree) > 40) throw new CalculatorError('That calculation is too large.');
  return formatNumber(evaluate(tree).value);
}

const DATE_QUERIES = new Set(["what date is today", "what is the date today", "what's the date today", "what is today's date",
  "what's today's date", "today's date", "todays date", "what is the date", "what's the date", "current date", "today date",
  "date today", "what day is it", "what day is today", "tell me today's date", "tell me the date", "what is the current date"]);
const TIME_QUERIES = new Set(["what time is it", "what is the time", "what's the time", "current time", "tell me the time",
  "what is the current time"]);


export function wordMath(query) {
  const number = String.raw`(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?`;
  let text=query.replace(/^(?:what is|what's|calculate|compute|how much is)\s+/, '').replace(/^(?:a|an)\s+/, '').replace(/\s+is$/, '');
  const match=text.match(new RegExp(`^(${number})\\s*(?:%|percent)\\s*(?:(tip|tax|discount)\\s+)?(?:of|on|off)\\s+(?:a\\s+)?([$€£₩]?)(${number})(?:\\s+bill)?$`));
  if(!match)return null;
  let [,rate,kind,currency,amount]=match;
  if(text.includes(' off '))kind='discount';
  rate=rate.replaceAll(',','');amount=amount.replaceAll(',','');
  try {
    if(kind==='discount' && Number(rate)>100)return 'A discount cannot exceed 100% of the original price.';
    const portion=calculate(`${amount} * ${rate} / 100`), equation=`${amount} × ${rate} / 100 = ${portion}`;
    if(['tip','tax','discount'].includes(kind)) {
      const total=calculate(`${amount} ${kind==='discount'?'-':'+'} ${portion}`);
      return `${kind[0].toUpperCase()+kind.slice(1)}: ${currency}${portion}. Total: ${currency}${total}.\n${equation}`;
    }
    return `${rate}% of ${currency}${amount} is ${currency}${portion}.\n${equation}`;
  } catch { return 'I could not calculate that. Keep numbers and results within one trillion.'; }
}


export function localAnswer(prompt, now = new Date()) {
  let query = prompt.toLowerCase().replaceAll('’', "'").split(new RegExp(`[${SPACE}]+`)).filter(Boolean).join(' ').replace(/[?! .]+$/, '');
  query = query.replace(/^please\s+|\s+please$/g, '');
  const support = supportReply(query);
  if (support) return [support, 'crisis_support'];
  const wordAnswer = wordMath(query);
  if (wordAnswer !== null) return [wordAnswer, 'local_calculator'];
  if (DATE_QUERIES.has(query) || TIME_QUERIES.has(query)) {
    const zone = new Intl.DateTimeFormat('en-US', { timeZoneName: 'short' }).formatToParts(now).find(part => part.type === 'timeZoneName')?.value || 'local time';
    if (DATE_QUERIES.has(query)) {
      const weekday = now.toLocaleDateString('en-US', { weekday: 'long' }), month = now.toLocaleDateString('en-US', { month: 'long' });
      return [`Today is ${weekday}, ${month} ${now.getDate()}, ${now.getFullYear()} (${zone}, this device's clock).`, 'local_clock'];
    }
    const pad = value => String(value).padStart(2, '0');
    return [`It is ${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())} (${zone}, this device's clock).`, 'local_clock'];
  }
  const expression = arithmeticExpression(query);
  if (expression === null) return null;
  try {
    return [`${expression} = ${calculate(expression)}`, 'local_calculator'];
  } catch (error) {
    if (error === ZERO) return ['Division by zero is undefined.', 'local_calculator'];
    const detail = error instanceof CalculatorError ? error.message : 'Check the numbers and parentheses.';
    return [`I could not calculate that. ${detail}`, 'local_calculator'];
  }
}


export function prepareReply(model, messages, maxTokens = 240) {
  conversation(messages, model.tokenizer);
  const followUp = supportFollowUp(messages);
  const tool = localAnswer(messages[messages.length - 1].content) || (followUp ? [followUp, 'crisis_support'] : null);
  const recent = messages.slice(model.config.context >= 512 ? -15 : -3);
  let retained = [];
  for (let i = 0; i < recent.length - 1; i += 2) {
    if (localAnswer(recent[i].content) === null && !webRequest(recent[i].content)) retained.push(...recent.slice(i, i + 2));
  }
  retained.push(recent.at(-1));
  let omitted = messages.length - retained.length;
  if (!tool) {
    const extra = model.preparePrompt(retained, maxTokens).omitted;
    omitted += extra;
    retained = retained.slice(extra);
  }
  return { retained, omitted, tool };
}

const WEIGHT_CACHE = 'kani-weights';
const SHA_HEADER = 'x-weights-sha256';

export async function loadWeights(base, manifest, fetcher = fetch) {
  const sha = manifest.weights_sha256, key = base + 'b1.bin';
  const verified = async weights => {
    if (manifest.bytes && weights.byteLength !== manifest.bytes) throw new Error('The model download was incomplete. Try again.');
    if (sha && globalThis.crypto?.subtle) {
      const actual = [...new Uint8Array(await crypto.subtle.digest('SHA-256', weights))].map(n => n.toString(16).padStart(2, '0')).join('');
      if (actual !== sha) throw new Error('Model weights failed verification. Try loading again.');
    }
    return weights;
  };
  let cache = null;
  try { if (sha && typeof caches !== 'undefined' && typeof location !== 'undefined') cache = await caches.open(WEIGHT_CACHE); } catch { cache = null; }
  try {
    const saved = cache && await cache.match(key);
    if (saved?.headers.get(SHA_HEADER) === sha) return await verified(await saved.arrayBuffer());
  } catch { try { await cache?.delete(key); } catch {} }
  const response = await fetcher(key);
  if (!response.ok) throw new Error('Could not load the model weights.');
  const weights = await verified(await response.arrayBuffer());
  if (cache && (!manifest.bytes || weights.byteLength === manifest.bytes)) {
    try { await cache.put(key, new Response(weights, { headers: { 'Content-Type': 'application/octet-stream', [SHA_HEADER]: sha } })); } catch { /* full or blocked storage only costs a later download */ }
  }
  return weights;
}

export async function loadModel(base = 'model/', fetcher = fetch) {
  const response = await fetcher(base + 'b1.json');
  if (!response.ok) throw new Error('Could not load the model description.');
  const manifest = await response.json();
  const weights = await loadWeights(base, manifest, fetcher);
  await enableKernels();
  return new Model(manifest, weights);
}
