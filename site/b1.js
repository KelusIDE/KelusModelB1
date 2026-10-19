// Kelus Model B1 browser engine. A port of kelus/tokenizer.py, the cached
// decoder in kelus/model.py, kelus/runtime.py and kelus/local_tools.py.
// Inference only: the page downloads the model files and nothing is sent back.

export const PAD = 256, BOS = 257, EOS = 258, USER = 259, ASSISTANT = 260;
// Python's `\s` for str, so text splits into exactly the trainer's pieces.
const SPACE = '\\t\\n\\v\\f\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000';
const REST = ` ?[A-Za-z]+| ?[0-9]+|[${SPACE}]+|[^A-Za-z0-9${SPACE}]`;
const PATTERNS = { 1: new RegExp(REST, 'gu'), 2: new RegExp('```[A-Za-z]*|' + REST, 'gu') };
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

export class Model {
  constructor(manifest, buffer) {
    if (manifest.format !== 'kelus-b1-web' || manifest.version !== 1) throw new Error('Unsupported model files');
    this.manifest = manifest;
    this.config = { positions: 'learned', tie_embeddings: false, ...manifest.config };
    this.tokenizer = new Tokenizer(manifest.tokenizer);
    const { context, width, heads, vocab_size: vocab } = this.config;
    if (this.tokenizer.vocabSize !== vocab || width % heads) throw new Error('Model and tokenizer do not match');
    const data = new Float32Array(buffer);
    this.weights = {};
    for (const tensor of manifest.tensors) {
      const size = tensor.shape.reduce((a, b) => a * b, 1);
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

  // x [rows] times row-major matrix [rows, columns].
  static multiply(x, matrix, columns) {
    const out = new Float64Array(columns);
    for (let i = 0; i < x.length; i++) {
      const value = x[i], row = i * columns;
      if (value) for (let j = 0; j < columns; j++) out[j] += value * matrix[row + j];
    }
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

  /** Logits after `tokens`, processing only positions not already in `cache`. */
  nextLogits(tokens, cache) {
    const { context, width: d, layers, heads, vocab_size: vocab } = this.config, w = this.weights, hw = this.headWidth;
    const start = cache.length || 0;
    if (!(tokens.length > start && tokens.length <= context)) throw new Error('Expected new tokens within the context limit');
    if (!cache.keys) {
      cache.keys = Array.from({ length: layers }, () => new Float64Array(context * d));
      cache.values = Array.from({ length: layers }, () => new Float64Array(context * d));
    }
    let x;
    for (let position = start; position < tokens.length; position++) {
      const token = tokens[position];
      if (!(token >= 0 && token < vocab)) throw new Error('Invalid token ID');
      x = Float64Array.from(w.token.subarray(token * d, token * d + d));
      if (this.config.positions === 'learned') for (let i = 0; i < d; i++) x[i] += w.position[position * d + i];
      for (let layer = 0; layer < layers; layer++) {
        const prefix = `block${layer}.`, normalized = this.norm(x, prefix + 'norm1');
        const q = Model.multiply(normalized, w[prefix + 'q'], d);
        const k = Model.multiply(normalized, w[prefix + 'k'], d);
        const v = Model.multiply(normalized, w[prefix + 'v'], d);
        if (this.config.positions === 'rope') { this.rotate(q, position); this.rotate(k, position); }
        const keys = cache.keys[layer], values = cache.values[layer];
        keys.set(k, position * d); values.set(v, position * d);
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
        const projected = Model.multiply(mixed, w[prefix + 'projection'], d);
        for (let i = 0; i < d; i++) x[i] += projected[i];
        const hidden = Model.multiply(this.norm(x, prefix + 'norm2'), w[prefix + 'up'], 4 * d);
        for (let i = 0; i < hidden.length; i++) {
          const h = hidden[i];
          hidden[i] = 0.5 * h * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (h + 0.044715 * h ** 3)));
        }
        const down = Model.multiply(hidden, w[prefix + 'down'], d);
        for (let i = 0; i < d; i++) x[i] += down[i];
      }
    }
    cache.length = tokens.length;
    const final = this.norm(x, 'final');
    if (!this.config.tie_embeddings) return Model.multiply(final, w.output, vocab);
    const logits = new Float64Array(vocab);
    for (let token = 0; token < vocab; token++) {
      let total = 0;
      for (let i = 0, row = token * d; i < d; i++) total += final[i] * w.token[row + i];
      logits[token] = total;
    }
    return logits;
  }

  /** Keep complete recent turns and leave room for the reply. */
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

  /** Yields decoded text pieces. Temperature 0 is greedy and repeatable. */
  *generate(messages, { maxTokens = 240, temperature = 0, topK = 30, random = Math.random } = {}) {
    if (!(maxTokens >= 1 && maxTokens <= 2048 && temperature >= 0 && temperature <= 2)) throw new Error('Invalid generation settings');
    const { tokens } = this.preparePrompt(messages, maxTokens);
    const decoder = new TextDecoder('utf-8'), cache = {};
    const steps = Math.min(maxTokens, this.config.context - tokens.length);
    for (let step = 0; step < steps; step++) {
      const logits = this.nextLogits(tokens, cache);
      for (const blocked of [PAD, BOS, USER, ASSISTANT]) logits[blocked] = -Infinity;
      let token = 0;
      if (temperature === 0) {
        for (let i = 1; i < logits.length; i++) if (logits[i] > logits[token]) token = i;
      } else {
        const top = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]).slice(0, topK);
        const highest = logits[top[0]] / temperature;
        const weights = top.map(i => Math.exp(logits[i] / temperature - highest));
        let pick = random() * weights.reduce((a, b) => a + b, 0);
        token = top[top.length - 1];
        for (let i = 0; i < top.length; i++) { pick -= weights[i]; if (pick <= 0) { token = top[i]; break; } }
      }
      if (token === EOS) break;
      tokens.push(token);
      const text = decoder.decode(this.tokenizer.tokenBytes(token), { stream: true });
      if (text) yield text;
    }
    const tail = decoder.decode();
    if (tail) yield tail;
  }
}

// ---- Local helpers (kelus/local_tools.py). Labeled in the UI, never model text.

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

/** Returns [text, source] or null. The source must be shown in the UI. */
export function localAnswer(prompt, now = new Date()) {
  let query = prompt.toLowerCase().replaceAll('’', "'").split(new RegExp(`[${SPACE}]+`)).filter(Boolean).join(' ').replace(/[?! .]+$/, '');
  query = query.replace(/^please\s+|\s+please$/g, '');
  if (DATE_QUERIES.has(query) || TIME_QUERIES.has(query)) {
    const zone = new Intl.DateTimeFormat('en-US', { timeZoneName: 'short' }).formatToParts(now).find(part => part.type === 'timeZoneName')?.value || 'local time';
    if (DATE_QUERIES.has(query)) {
      const weekday = now.toLocaleDateString('en-US', { weekday: 'long' }), month = now.toLocaleDateString('en-US', { month: 'long' });
      return [`Today is ${weekday}, ${month} ${now.getDate()}, ${now.getFullYear()} (${zone}, this device's clock).`, 'local_clock'];
    }
    const pad = value => String(value).padStart(2, '0');
    return [`It is ${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())} (${zone}, this device's clock).`, 'local_clock'];
  }
  let expression = query.replace(/^(?:what is|what's|calculate|compute|solve|evaluate)\s+/, '').replaceAll('×', '*').replaceAll('÷', '/').replaceAll('−', '-');
  if (expression.endsWith('=')) expression = expression.slice(0, -1).replace(new RegExp(`[${SPACE}]+$`), '');
  if (!new RegExp(`^[0-9.${SPACE}+*/%()\\-]+$`).test(expression) || !/[+*/%-]/.test(expression)) return null;
  try {
    return [`${expression} = ${calculate(expression)}`, 'local_calculator'];
  } catch (error) {
    if (error === ZERO) return ['Division by zero is undefined.', 'local_calculator'];
    const detail = error instanceof CalculatorError ? error.message : 'Check the numbers and parentheses.';
    return [`I could not calculate that. ${detail}`, 'local_calculator'];
  }
}

/** kelus/runtime.py: one preceding exchange, and helper replies are not fed back. */
export function prepareReply(model, messages, maxTokens = 240) {
  conversation(messages, model.tokenizer);
  const tool = localAnswer(messages[messages.length - 1].content);
  let retained = messages.slice(-3);
  if (retained.length === 3 && localAnswer(retained[0].content) !== null) retained = retained.slice(-1);
  let omitted = messages.length - retained.length;
  if (!tool) {
    const extra = model.preparePrompt(retained, maxTokens).omitted;
    omitted += extra;
    retained = retained.slice(extra);
  }
  return { retained, omitted, tool };
}

export async function loadModel(base = 'model/', fetcher = fetch) {
  const [manifest, weights] = await Promise.all([
    fetcher(base + 'b1.json').then(response => { if (!response.ok) throw new Error('Could not load the model description.'); return response.json(); }),
    fetcher(base + 'b1.bin').then(response => { if (!response.ok) throw new Error('Could not load the model weights.'); return response.arrayBuffer(); }),
  ]);
  return new Model(manifest, weights);
}
