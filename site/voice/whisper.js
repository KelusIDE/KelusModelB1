// Whisper base.en (OpenAI, MIT) as ONNX: speech to text on this device, with how sure it is of every word, and how likely
// the words a learner was asked to say are given what they actually said (to find the sounds that need work).

export const SAMPLE_RATE = 16000;
const N_FFT = 400, HOP = 160, MELS = 80, FRAMES = 3000, SAMPLES = 30 * SAMPLE_RATE, BINS = N_FFT / 2 + 1;
const SOT = 50257, EOT = 50256, NO_TIMESTAMPS = 50362, LAYERS = 6, HEADS = 8, HEAD_SIZE = 64;
// generation_config.json's suppress_tokens: symbols Whisper should not write (music notes, brackets and so on).
const SUPPRESS = [1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 357, 366, 438, 532, 685, 705, 796,
  930, 1058, 1220, 1267, 1279, 1303, 1343, 1377, 1391, 1635, 1782, 1875, 2162, 2361, 2488, 3467, 4008, 4211, 4600, 4808, 5299, 5855, 6329,
  7203, 9609, 9959, 10563, 10786, 11420, 11709, 11907, 13163, 13697, 13700, 14808, 15306, 16410, 16791, 17992, 19203, 19510, 20724, 22305,
  22935, 27007, 30109, 30420, 33409, 34949, 40283, 40493, 40549, 47282, 49146];

let tables = null;
function melTables() {
  if (tables) return tables;
  // Slaney mel filters (librosa's default, as Whisper's feature extractor uses).
  const toMel = f => f < 1000 ? 3 * f / 200 : 15 + 27 * Math.log(f / 1000) / Math.log(6.4);
  const toHz = m => m < 15 ? 200 * m / 3 : 1000 * Math.exp((m - 15) * Math.log(6.4) / 27);
  const top = toMel(SAMPLE_RATE / 2), points = Array.from({ length: MELS + 2 }, (_, i) => toHz(top * i / (MELS + 1)));
  const filters = new Float32Array(MELS * BINS);
  for (let m = 0; m < MELS; m++) {
    const [low, centre, high] = [points[m], points[m + 1], points[m + 2]], norm = 2 / (high - low);
    for (let k = 0; k < BINS; k++) {
      const f = k * SAMPLE_RATE / N_FFT;
      filters[m * BINS + k] = Math.max(0, Math.min((f - low) / (centre - low), (high - f) / (high - centre))) * norm;
    }
  }
  const cos = new Float32Array(BINS * N_FFT), sin = new Float32Array(BINS * N_FFT), window = new Float32Array(N_FFT);
  for (let n = 0; n < N_FFT; n++) window[n] = 0.5 - 0.5 * Math.cos(2 * Math.PI * n / N_FFT);
  for (let k = 0; k < BINS; k++) {
    for (let n = 0; n < N_FFT; n++) { cos[k * N_FFT + n] = Math.cos(2 * Math.PI * k * n / N_FFT); sin[k * N_FFT + n] = Math.sin(2 * Math.PI * k * n / N_FFT); }
  }
  return (tables = { filters, cos, sin, window });
}

// Whisper's log-mel spectrogram of up to 30 seconds of 16 kHz audio: 80 x 3000, mel-major.
export function logMel(audio) {
  const { filters, cos, sin, window } = melTables();
  const length = Math.min(audio.length, SAMPLES), half = N_FFT / 2;
  const padded = new Float32Array(SAMPLES + N_FFT);
  padded.set(audio.subarray(0, length), half);
  for (let i = 1; i <= half; i++) { padded[half - i] = padded[half + i]; padded[half + SAMPLES - 1 + i] = padded[half + SAMPLES - 1 - i]; }
  const mel = new Float32Array(MELS * FRAMES), frame = new Float32Array(N_FFT), power = new Float32Array(BINS);
  const used = Math.min(FRAMES, Math.ceil((length + half) / HOP) + 1);
  let peak = -Infinity;
  for (let t = 0; t < FRAMES; t++) {
    if (t < used) {
      for (let n = 0; n < N_FFT; n++) frame[n] = padded[t * HOP + n] * window[n];
      for (let k = 0; k < BINS; k++) {
        let re = 0, im = 0;
        const row = k * N_FFT;
        for (let n = 0; n < N_FFT; n++) { re += frame[n] * cos[row + n]; im -= frame[n] * sin[row + n]; }
        power[k] = re * re + im * im;
      }
    } else power.fill(0);
    for (let m = 0; m < MELS; m++) {
      let sum = 0;
      const row = m * BINS;
      for (let k = 0; k < BINS; k++) sum += filters[row + k] * power[k];
      const value = Math.log10(Math.max(sum, 1e-10));
      mel[m * FRAMES + t] = value;
      if (value > peak) peak = value;
    }
  }
  for (let i = 0; i < mel.length; i++) mel[i] = (Math.max(mel[i], peak - 8) + 4) / 4;
  return mel;
}

// GPT-2 style byte-level BPE (Whisper's tokenizer): site/voice/whisper-tokens.json.
const BYTES = (() => {
  const keep = [...Array(94).keys()].map(i => i + 33).concat([...Array(12).keys()].map(i => i + 161), [...Array(82).keys()].map(i => i + 174));
  const map = new Array(256);
  let extra = 0;
  for (let b = 0; b < 256; b++) map[b] = String.fromCharCode(keep.includes(b) ? b : 256 + extra++);
  return map;
})();
const UNBYTES = new Map(BYTES.map((c, b) => [c, b]));
const PIECES = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

export class WhisperTokens {
  constructor({ tokens, merges }) {
    this.tokens = tokens;
    this.ids = new Map(tokens.map((token, id) => [token, id]));
    this.ranks = new Map(merges.map((merge, rank) => [merge, rank]));
  }

  encode(text) {
    const ids = [];
    for (const piece of text.match(PIECES) || []) {
      let parts = [...new TextEncoder().encode(piece)].map(b => BYTES[b]);
      while (parts.length > 1) {
        let best = -1, rank = Infinity;
        for (let i = 0; i < parts.length - 1; i++) {
          const r = this.ranks.get(parts[i] + ' ' + parts[i + 1]);
          if (r !== undefined && r < rank) { rank = r; best = i; }
        }
        if (best < 0) break;
        parts = [...parts.slice(0, best), parts[best] + parts[best + 1], ...parts.slice(best + 2)];
      }
      for (const part of parts) ids.push(this.ids.get(part));
    }
    return ids;
  }

  text(id) {
    if (id >= this.tokens.length) return '';
    return new TextDecoder().decode(Uint8Array.from([...this.tokens[id]].map(c => UNBYTES.get(c))));
  }

  decode(ids) {
    return new TextDecoder().decode(Uint8Array.from(ids.filter(id => id < this.tokens.length).flatMap(id => [...this.tokens[id]].map(c => UNBYTES.get(c)))));
  }
}

// Tokens grouped into words (a new word starts at a token with a leading space); punctuation does not count toward how
// sure Whisper was.
function words(tokens, pieces, logprobs) {
  const out = [];
  pieces.forEach((piece, i) => {
    const starts = piece.startsWith(' ') || !out.length;
    const spoken = /[\p{L}\p{N}]/u.test(piece);
    if (starts && spoken) out.push({ text: piece.trim(), logprob: logprobs[i], tokens: [tokens[i]] });
    else if (out.length) { const last = out[out.length - 1]; last.text += piece.trimEnd(); if (spoken) last.logprob += logprobs[i]; last.tokens.push(tokens[i]); }
    else out.push({ text: piece.trim(), logprob: 0, tokens: [tokens[i]] });
  });
  return out.map(({ text, logprob }) => ({ text, probability: Math.exp(logprob) }));
}

const logSoftmaxAt = (data, offset, size, id) => {
  let max = -Infinity;
  for (let i = 0; i < size; i++) if (data[offset + i] > max) max = data[offset + i];
  let sum = 0;
  for (let i = 0; i < size; i++) sum += Math.exp(data[offset + i] - max);
  return data[offset + id] - max - Math.log(sum);
};

export class Whisper {
  constructor(ort, encoder, decoder, tokens) { this.ort = ort; this.encoder = encoder; this.decoder = decoder; this.tokens = tokens; }

  static async create(ort, { encoder, decoder, tokens }) {
    const options = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
    const [e, d] = await Promise.all([ort.InferenceSession.create(encoder, options), ort.InferenceSession.create(decoder, options)]);
    return new Whisper(ort, e, d, new WhisperTokens(tokens));
  }

  async encode(audio) {
    const { Tensor } = this.ort;
    const result = await this.encoder.run({ input_features: new Tensor('float32', logMel(audio), [1, MELS, FRAMES]) });
    return result.last_hidden_state;
  }

  async #step(ids, states, past) {
    const { Tensor } = this.ort;
    const empty = () => new Tensor('float32', new Float32Array(0), [1, HEADS, 0, HEAD_SIZE]);
    const feeds = {
      input_ids: new Tensor('int64', BigInt64Array.from(ids, BigInt), [1, ids.length]),
      encoder_hidden_states: states,
      use_cache_branch: new Tensor('bool', [Boolean(past)], [1]),
    };
    for (let i = 0; i < LAYERS; i++) {
      for (const part of ['decoder', 'encoder']) {
        for (const kind of ['key', 'value']) feeds[`past_key_values.${i}.${part}.${kind}`] = past?.[`present.${i}.${part}.${kind}`] || empty();
      }
    }
    const out = await this.decoder.run(feeds);
    if (past) for (let i = 0; i < LAYERS; i++) for (const kind of ['key', 'value']) out[`present.${i}.encoder.${kind}`] = past[`present.${i}.encoder.${kind}`];
    return out;
  }

  // What was said, greedily, with each word's probability. states: from encode().
  async transcribe(states, { maxTokens = 120 } = {}) {
    const vocab = 51864, chosen = [], logprobs = [];
    let past = null, input = [SOT, NO_TIMESTAMPS];
    for (let step = 0; step < maxTokens; step++) {
      const out = await this.#step(input, states, past);
      const logits = out.logits.data, offset = (input.length - 1) * vocab;
      const row = logits.slice(offset, offset + vocab);
      for (const id of SUPPRESS) row[id] = -Infinity;
      for (let id = EOT + 1; id < vocab; id++) row[id] = -Infinity;
      if (step === 0) { row[220] = -Infinity; row[EOT] = -Infinity; }
      let best = 0;
      for (let id = 1; id < vocab; id++) if (row[id] > row[best]) best = id;
      if (best === EOT) break;
      chosen.push(best); logprobs.push(logSoftmaxAt(row, 0, vocab, best));
      // Stop a loop ("the the the ...").
      const n = chosen.length;
      if (n >= 12 && chosen.slice(n - 4).join() === chosen.slice(n - 8, n - 4).join() && chosen.slice(n - 4).join() === chosen.slice(n - 12, n - 8).join()) break;
      past = out; input = [best];
    }
    const pieces = chosen.map(id => this.tokens.text(id));
    return { text: this.tokens.decode(chosen).trim(), words: words(chosen, pieces, logprobs) };
  }

  // How likely Whisper finds `text` as what was said (teacher forcing), word by word: low for words that sounded different.
  async score(states, text) {
    const ids = this.tokens.encode(' ' + text.trim());
    const out = await this.#step([SOT, NO_TIMESTAMPS, ...ids], states, null);
    const vocab = out.logits.dims[2], logits = out.logits.data;
    const logprobs = ids.map((id, i) => logSoftmaxAt(logits, (i + 1) * vocab, vocab, id));
    return words(ids, ids.map(id => this.tokens.text(id)), logprobs);
  }
}

// Linear-phase resampling with a windowed-sinc low-pass (for the microphone's 44.1/48 kHz and Kokoro's 24 kHz).
export function resample(samples, from, to) {
  if (from === to) return samples;
  const ratio = from / to, length = Math.floor(samples.length / ratio), out = new Float32Array(length);
  const cutoff = Math.min(1, to / from) * 0.95, taps = 16;
  for (let i = 0; i < length; i++) {
    const centre = i * ratio, start = Math.ceil(centre - taps / cutoff), end = Math.floor(centre + taps / cutoff);
    let sum = 0, weight = 0;
    for (let j = Math.max(0, start); j <= Math.min(samples.length - 1, end); j++) {
      const x = (j - centre) * cutoff, w = 0.5 + 0.5 * Math.cos(Math.PI * (j - centre) * cutoff / taps);
      const s = (x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x)) * w;
      sum += samples[j] * s; weight += s;
    }
    out[i] = weight ? sum / weight : 0;
  }
  return out;
}
