// The microphone, the speaker, and the voice worker (Whisper and Kokoro). Audio never leaves this device.
import { SAMPLE_RATE } from '../../site/voice/whisper.js';
import { KOKORO_RATE } from '../../site/voice/kokoro.js';
import { LISTEN_FILES, SPEAK_FILES, totalSize } from '../../site/voice/models.js';

export interface SpokenWord { text: string; probability: number }
export interface Heard { text: string; words: SpokenWord[] }
export type Part = 'listen' | 'speak';
export const PART_SIZE: Record<Part, number> = { listen: totalSize(LISTEN_FILES), speak: totalSize(SPEAK_FILES) };

type Reply = { type: 'progress'; loaded: number; total: number } | { type: 'audio'; samples: Float32Array }
  | { type: 'done'; heard?: Heard; scored?: SpokenWord[] } | { type: 'error'; message: string };

let worker: Worker | null = null, nextId = 1;
const waiting = new Map<number, (reply: Reply) => void>();
function call(message: Record<string, unknown>, onReply: (reply: Reply) => void, transfer: Transferable[] = []) {
  if (!worker) {
    worker = new Worker(new URL('./voice.worker.ts', import.meta.url), { type: 'module' });
    worker.onmessage = ({ data }) => waiting.get(data.id)?.(data);
  }
  const id = nextId++;
  return new Promise<Reply & { type: 'done' }>((resolve, reject) => {
    waiting.set(id, reply => {
      if (reply.type === 'done') { waiting.delete(id); resolve(reply); }
      else if (reply.type === 'error') { waiting.delete(id); reject(new Error(reply.message)); }
      else onReply(reply);
    });
    worker!.postMessage({ ...message, id }, transfer);
  });
}

export const loadPart = (part: Part, onProgress: (loaded: number, total: number) => void) =>
  call({ type: 'load', part, base: document.baseURI }, reply => { if (reply.type === 'progress') onProgress(reply.loaded, reply.total); });

// What was said (with how sure Whisper is of each word) and, for read-aloud practice, how likely the target words are.
export async function transcribe(audio: Float32Array, target?: string) {
  const done = await call({ type: 'listen', audio, target }, () => {}, [audio.buffer]);
  return { heard: done.heard!, scored: done.scored! };
}

// Speech, one sentence group at a time, through one shared AudioContext (made on a tap, as browsers require).
let context: AudioContext | null = null, playing = new Set<AudioBufferSourceNode>(), endsAt = 0, generation = 0;
export function primeAudio() {
  context ||= new AudioContext();
  if (context.state === 'suspended') void context.resume();
  return context;
}

export function stopSpeaking() {
  generation++;
  for (const source of playing) { try { source.stop(); } catch { /* already stopped */ } }
  playing.clear(); endsAt = 0;
}

let queue: Promise<void> = Promise.resolve();
// Queued after anything already speaking; resolves when this text has finished playing.
export function speak(text: string, { voice = 'af_heart', speed = 1 } = {}) {
  const mine = generation;
  queue = queue.then(async () => {
    if (mine !== generation || !text.trim()) return;
    const audio = primeAudio();
    let last: Promise<void> = Promise.resolve();
    await call({ type: 'speak', text, voice, speed }, reply => {
      if (reply.type !== 'audio' || mine !== generation) return;
      const buffer = audio.createBuffer(1, reply.samples.length, KOKORO_RATE);
      buffer.copyToChannel(reply.samples as Float32Array<ArrayBuffer>, 0);
      const source = audio.createBufferSource();
      source.buffer = buffer; source.connect(audio.destination);
      const start = Math.max(audio.currentTime + 0.05, endsAt);
      source.start(start); endsAt = start + buffer.duration;
      playing.add(source);
      last = new Promise(resolve => { source.onended = () => { playing.delete(source); resolve(); }; });
    });
    await last;
  }).catch(() => {});
  return queue;
}

// Records until the speaker stops (about a second of quiet after speech), then returns 16 kHz mono samples.
export async function recordSpeech({ onLevel, signal, maxSeconds = 20 }: { onLevel: (level: number) => void; signal: AbortSignal; maxSeconds?: number }) {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 } });
  // The context made on the tap (one made after an await starts suspended and hears only silence).
  const audio = primeAudio(), analyser = audio.createAnalyser(), source = audio.createMediaStreamSource(stream);
  await audio.resume().catch(() => {});
  analyser.fftSize = 2048;
  source.connect(analyser);
  const recorder = new MediaRecorder(stream), chunks: Blob[] = [];
  recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
  const stopped = new Promise<void>(resolve => { recorder.onstop = () => resolve(); });
  recorder.start(250);
  const frame = new Float32Array(analyser.fftSize), started = performance.now();
  let floor = 0.005, heard = false, lastLoud = started;
  await new Promise<void>(resolve => {
    const timer = setInterval(() => {
      analyser.getFloatTimeDomainData(frame);
      const level = Math.sqrt(frame.reduce((sum, x) => sum + x * x, 0) / frame.length), now = performance.now();
      if (now - started < 400) floor = Math.min(0.02, Math.max(floor, level));
      onLevel(Math.min(1, level * 8));
      if (level > Math.max(0.012, floor * 2.5)) { heard = true; lastLoud = now; }
      const quiet = heard ? now - lastLoud > 1300 : now - started > 8000;
      if (quiet || now - started > maxSeconds * 1000 || signal.aborted) { clearInterval(timer); resolve(); }
    }, 80);
  });
  recorder.stop();
  await stopped;
  for (const track of stream.getTracks()) track.stop();
  source.disconnect();
  onLevel(0);
  if (!chunks.length) return null;
  const encoded = await new Blob(chunks, { type: recorder.mimeType }).arrayBuffer();
  const decoded = (await new OfflineAudioContext(1, 1, SAMPLE_RATE).decodeAudioData(encoded)).getChannelData(0).slice();
  // Whether anything was said is decided from the recording itself (the level meter is only for stopping on time).
  let loudest = 0;
  for (let i = 0; i + 800 <= decoded.length; i += 800) {
    let sum = 0;
    for (let j = i; j < i + 800; j++) sum += decoded[j] * decoded[j];
    loudest = Math.max(loudest, Math.sqrt(sum / 800));
  }
  return loudest > 0.01 || heard ? decoded : null;
}

export const canRecord = () => 'mediaDevices' in navigator && typeof MediaRecorder !== 'undefined';
