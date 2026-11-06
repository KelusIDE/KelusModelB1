import { modelBytes, validateManifest } from './model-validation.mjs';



export async function loadWeights(base, progress, signal) {
  progress(0, 'Opening Kani…');
  const response = await fetch(base + 'b1.json', { signal });
  if (!response.ok) throw new Error('The Kani model is missing from this app. Rebuild with npm run model.');
  const manifest = await response.json(), expected = modelBytes(manifest);
  validateManifest(manifest, expected);
  const weights = await fetch(base + 'b1.bin', { signal });
  if (!weights.ok) throw new Error('Could not open the model weights.');
  if (!weights.body) {
    const buffer = await weights.arrayBuffer(); validateManifest(manifest, buffer.byteLength);
    return { manifest, buffer };
  }
  const buffer = new Uint8Array(expected), reader = weights.body.getReader();
  let offset = 0, lastPercent = -1;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (offset + value.length > expected) throw new Error('Incorrect model file size.');
      buffer.set(value, offset); offset += value.length;
      const percent = Math.floor(offset / expected * 90);
      if (percent !== lastPercent) { progress(percent, 'Loading on your device…'); lastPercent = percent; }
    }
  } finally { await reader.cancel(); }
  validateManifest(manifest, offset);
  return { manifest, buffer: buffer.buffer };
}
