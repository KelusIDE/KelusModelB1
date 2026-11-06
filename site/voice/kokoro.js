// Kokoro-82M (hexgrad, Apache-2.0) as ONNX: text to 24 kHz speech on this device. Phonemes come from site/voice/pronounce.js
// (the CMU dictionary), not espeak.
import { toKokoro } from './pronounce.js';

export const KOKORO_RATE = 24000;
const VOCAB = { ';': 1, ':': 2, ',': 3, '.': 4, '!': 5, '?': 6, '—': 9, '…': 10, '"': 11, '(': 12, ')': 13, '“': 14, '”': 15, ' ': 16,
  A: 24, I: 25, O: 31, Q: 33, S: 35, T: 36, W: 39, Y: 41, 'ᵊ': 42, a: 43, b: 44, c: 45, d: 46, e: 47, f: 48, h: 50, i: 51, j: 52, k: 53,
  l: 54, m: 55, n: 56, o: 57, p: 58, q: 59, r: 60, s: 61, t: 62, u: 63, v: 64, w: 65, x: 66, y: 67, z: 68, 'ɑ': 69, 'ɐ': 70, 'ɒ': 71,
  'æ': 72, 'ɔ': 76, 'ð': 81, 'ʤ': 82, 'ə': 83, 'ɚ': 85, 'ɛ': 86, 'ɜ': 87, 'ɡ': 92, 'ɪ': 102, 'ŋ': 112, 'θ': 119, 'ɹ': 123, 'ɾ': 125,
  'ʃ': 131, 'ʧ': 133, 'ʊ': 135, 'ʌ': 138, 'ʒ': 147, 'ʔ': 148, 'ˈ': 156, 'ˌ': 157, 'ː': 158, 'ᵻ': 177 };
const MAX_TOKENS = 510, STYLE = 256;

export const kokoroIds = phonemes => [...phonemes].map(symbol => VOCAB[symbol]).filter(id => id !== undefined);

// Phoneme strings short enough for one Kokoro call, split after sentences (or commas, for very long sentences).
export function kokoroChunks(text, words, limit = 400) {
  const chunks = [];
  for (const sentence of text.match(/[^.!?…\n]+[.!?…]*|\n+/g) || []) {
    const phonemes = toKokoro(sentence, words);
    if (!phonemes) continue;
    const pieces = phonemes.length <= limit ? [phonemes] : phonemes.split(/(?<=[,;:]) /);
    for (const piece of pieces) {
      for (let start = 0; start < piece.length; start += limit) {
        const part = piece.slice(start, start + limit).trim();
        const last = chunks.length - 1;
        if (last >= 0 && chunks[last].length + part.length + 1 <= limit && part.length < 60) chunks[last] += ' ' + part;
        else if (part) chunks.push(part);
      }
    }
  }
  return chunks;
}

export class Kokoro {
  constructor(ort, session, voices) { this.ort = ort; this.session = session; this.voices = voices; }

  // model: the ONNX file's bytes; voices: { name: bytes of voices/<name>.bin } (510 styles of 256 numbers, one per length).
  static async create(ort, model, voices) {
    const session = await ort.InferenceSession.create(model, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    const styles = Object.fromEntries(Object.entries(voices).map(([name, bytes]) => [name, new Float32Array(bytes.buffer ? bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) : bytes)]));
    return new Kokoro(ort, session, styles);
  }

  // One chunk of phonemes (see kokoroChunks) to samples at KOKORO_RATE.
  async speak(phonemes, { voice = Object.keys(this.voices)[0], speed = 1 } = {}) {
    const ids = kokoroIds(phonemes).slice(0, MAX_TOKENS - 2);
    if (!ids.length) return new Float32Array(0);
    const styles = this.voices[voice];
    if (!styles) throw new Error(`Unknown voice ${voice}`);
    const row = Math.min(ids.length, styles.length / STYLE - 1);
    const { Tensor } = this.ort;
    const result = await this.session.run({
      input_ids: new Tensor('int64', BigInt64Array.from([0, ...ids, 0], BigInt), [1, ids.length + 2]),
      style: new Tensor('float32', styles.slice(row * STYLE, (row + 1) * STYLE), [1, STYLE]),
      speed: new Tensor('float32', Float32Array.of(speed), [1]),
    });
    return result.waveform.data;
  }
}
