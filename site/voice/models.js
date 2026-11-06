// The voice models: ONNX files from Hugging Face at fixed revisions, checked against their SHA-256 before use and kept in
// the browser's Cache Storage, so they download once. They are not Kani's weights: Whisper base.en is OpenAI's (MIT) and
// Kokoro-82M is hexgrad's (Apache-2.0).

const WHISPER = 'https://huggingface.co/onnx-community/whisper-base.en/resolve/51eefc0af78b103839eda9e7e4f4186acc6517fe/onnx/';
const KOKORO = 'https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/1939ad2a8e416c0acfeecc08a694d14ef25f2231/';
export const VOICE_FILES = {
  encoder: { url: WHISPER + 'encoder_model_quantized.onnx', size: 23201320, sha256: '6e8001198c490bbae018c0044f630c2915efb826bad957006ce36152d0ab2a10' },
  decoder: { url: WHISPER + 'decoder_model_merged_quantized.onnx', size: 53692803, sha256: 'dd4761a3f7add26afda3512abff4706920404c2517e85a9f2ff090b0c0987909' },
  kokoro: { url: KOKORO + 'onnx/model_quantized.onnx', size: 92361116, sha256: 'fbae9257e1e05ffc727e951ef9b9c98418e6d79f1c9b6b13bd59f5c9028a1478' },
  af_heart: { url: KOKORO + 'voices/af_heart.bin', size: 522240, sha256: 'd583ccff3cdca2f7fae535cb998ac07e9fcb90f09737b9a41fa2734ec44a8f0b' },
  am_michael: { url: KOKORO + 'voices/am_michael.bin', size: 522240, sha256: '1d1f21dd8da39c30705cd4c75d039d265e9bc4a2a93ed09bc9e1b1225eb95ba1' },
};
export const LISTEN_FILES = ['encoder', 'decoder'];
export const SPEAK_FILES = ['kokoro', 'af_heart', 'am_michael'];
const CACHE = 'kani-voice-v1';

const hex = buffer => [...new Uint8Array(buffer)].map(b => b.toString(16).padStart(2, '0')).join('');

// One file's bytes, from the cache or the network; onProgress(bytesSoFar) while downloading.
/** @param {string} name @param {{ onProgress?: (bytes: number) => void, fetcher?: typeof fetch, caches?: CacheStorage }} [options] */
export async function voiceFile(name, { onProgress = () => {}, fetcher = fetch, caches: store = globalThis.caches } = {}) {
  const file = VOICE_FILES[/** @type {keyof typeof VOICE_FILES} */ (name)];
  let cache = null;
  try { cache = await store?.open(CACHE); } catch { cache = null; }
  const cached = await cache?.match(file.url).catch(() => null);
  let bytes = cached ? new Uint8Array(await cached.arrayBuffer()) : null;
  if (bytes && bytes.length === file.size && hex(await crypto.subtle.digest('SHA-256', bytes)) === file.sha256) {
    onProgress(file.size);
    return bytes;
  }
  const response = await fetcher(file.url, { credentials: 'omit', referrerPolicy: 'no-referrer' });
  if (!response.ok || !response.body) throw new Error(`Could not download the voice model (${response.status}).`);
  const reader = response.body.getReader(), chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); received += value.length;
    if (received > file.size) throw new Error('The voice model download was larger than expected.');
    onProgress(received);
  }
  bytes = new Uint8Array(received);
  let at = 0;
  for (const chunk of chunks) { bytes.set(chunk, at); at += chunk.length; }
  if (received !== file.size || hex(await crypto.subtle.digest('SHA-256', bytes)) !== file.sha256) {
    throw new Error('The voice model download did not match its checksum, so it was not used.');
  }
  await cache?.put(file.url, new Response(bytes, { headers: { 'Content-Type': 'application/octet-stream' } })).catch(() => {});
  return bytes;
}

export const totalSize = names => names.reduce((sum, name) => sum + VOICE_FILES[name].size, 0);
