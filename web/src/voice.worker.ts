// Runs the voice models off the main thread: Whisper (listen) and Kokoro (speak), with ONNX Runtime's WebAssembly backend.
import * as ort from 'onnxruntime-web/wasm';
import wasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import { parseDictionary } from '../../site/voice/pronounce.js';
import { Kokoro, kokoroChunks } from '../../site/voice/kokoro.js';
import { Whisper } from '../../site/voice/whisper.js';
import { LISTEN_FILES, SPEAK_FILES, totalSize, voiceFile } from '../../site/voice/models.js';

ort.env.wasm.wasmPaths = { wasm: new URL(wasmUrl, self.location.href).href };
ort.env.wasm.numThreads = 1;

type Part = 'listen' | 'speak';
type Request =
  | { id: number; type: 'load'; part: Part; base: string }
  | { id: number; type: 'listen'; audio: Float32Array; target?: string }
  | { id: number; type: 'speak'; text: string; voice: string; speed: number };

let words: Map<string, string[]> | null = null;
let whisper: Promise<Whisper> | null = null;
let kokoro: Promise<Kokoro> | null = null;
const post = (message: unknown, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(message, transfer);

async function dictionary(base: string) {
  if (!words) words = parseDictionary(await (await fetch(new URL('voice/cmudict.txt', base))).text());
  return words;
}

async function files(id: number, names: string[]) {
  const total = totalSize(names), done: Record<string, number> = {};
  const report = () => post({ id, type: 'progress', loaded: Object.values(done).reduce((a, b) => a + b, 0), total });
  const bytes = await Promise.all(names.map(name => voiceFile(name, { onProgress: n => { done[name] = n; report(); } })));
  return Object.fromEntries(names.map((name, i) => [name, bytes[i]]));
}

function load(id: number, part: Part, base: string) {
  if (part === 'listen') {
    whisper ||= (async () => {
      const [got, tokens] = await Promise.all([files(id, LISTEN_FILES), fetch(new URL('voice/whisper-tokens.json', base)).then(r => r.json())]);
      return Whisper.create(ort, { encoder: got.encoder, decoder: got.decoder, tokens });
    })().catch(error => { whisper = null; throw error; });
    return whisper;
  }
  kokoro ||= (async () => {
    const [got] = await Promise.all([files(id, SPEAK_FILES), dictionary(base)]);
    return Kokoro.create(ort, got.kokoro, { af_heart: got.af_heart, am_michael: got.am_michael });
  })().catch(error => { kokoro = null; throw error; });
  return kokoro;
}

self.onmessage = async ({ data }: MessageEvent<Request>) => {
  const { id } = data;
  try {
    if (data.type === 'load') {
      await load(id, data.part, data.base);
      post({ id, type: 'done' });
    } else if (data.type === 'listen') {
      const model = await whisper;
      if (!model) throw new Error('The listening model is not loaded.');
      const states = await model.encode(data.audio);
      const heard = await model.transcribe(states);
      const scored = data.target ? await model.score(states, data.target) : await model.score(states, heard.text || ' ');
      post({ id, type: 'done', heard, scored });
    } else if (data.type === 'speak') {
      const model = await kokoro;
      if (!model || !words) throw new Error('The speaking model is not loaded.');
      for (const chunk of kokoroChunks(data.text, words)) {
        const samples = Float32Array.from(await model.speak(chunk, { voice: data.voice, speed: data.speed }));
        post({ id, type: 'audio', samples }, [samples.buffer]);
      }
      post({ id, type: 'done' });
    }
  } catch (error) {
    post({ id, type: 'error', message: (error as Error)?.message || String(error) });
  }
};

export {};
