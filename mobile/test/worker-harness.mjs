
import { parentPort } from 'node:worker_threads';
import { readFile } from 'node:fs/promises';
import { loadWeights } from '../src/load-weights.mjs';
globalThis.self = globalThis;
self.postMessage = event => parentPort.postMessage(event);


const EINSTEIN = { query: { pages: [{ ns: 0, index: 1, title: 'Albert Einstein',
  extract: 'Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist. He was born in Ulm, in the Kingdom of Württemberg.' }] } };

export const placesRequests = [];
const PLACES = { source: 'Google Maps', places: [
  { name: 'Joe’s Pizza', address: '1435 Broadway, New York', rating: 4.6, ratings: 18234, open: true, price: '$', url: 'https://maps.google.com/?cid=1' },
  { name: 'Quiet Slice', address: '2 Main St, New York', rating: 4.2, ratings: 310, open: false, price: '$$', url: 'https://maps.google.com/?cid=2' }] };
globalThis.fetch = async (url, options) => {
  if (url === 'https://somewhat.chatgpt-ai-5-2o2.workers.dev/search') return new Response('Forbidden origin', { status: 403 });
  if (url === 'https://somewhat.chatgpt-ai-5-2o2.workers.dev/places') {
    parentPort.postMessage({ type: 'places-request', body: JSON.parse(options.body) });
    return new Response(JSON.stringify(PLACES));
  }
  if (String(url).startsWith('https://en.wikipedia.org/w/api.php?')) return new Response(JSON.stringify(EINSTEIN));
  const bundled = String(url).match(/^https:\/\/kani\.test\/models\/(otter-4-12|seal-3)\/(b1\.json|b1\.bin)$/);
  if (!bundled) throw new Error('Unexpected network request: ' + url);
  return new Response(await readFile(new URL(`../public/models/${bundled[1]}/${bundled[2]}`, import.meta.url)));
};
await import('../src/inference.worker.ts');
parentPort.on('message', async data => {
  if (data.type === 'load') {
    const { manifest, buffer } = await loadWeights(data.base, () => {}, undefined);
    self.onmessage({ data: { type: 'load', manifest, buffer, gpu: true } });
  } else self.onmessage({ data });
});
