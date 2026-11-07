import { Model, enableKernels, prepareReply } from '../../site/b1.js';
import { GpuModel, gpuModel } from '../../site/gpu.js';
import { SEARCH_CALL, maybeSearch, searchedAnswer } from '../../site/reading.js';
import { webAnswer, webRequest } from '../../site/web.js';
import { PLACES_CALL, maybePlaces, placesAnswer, placesSummary } from '../../site/places.js';
import { maybeCide } from '../../site/cide.js';
import { validateManifest } from './model-validation.mjs';
import { sha256 } from '@noble/hashes/sha2.js';
import type { Message, WorkerReply } from './types';

const emit = (event: WorkerReply) => self.postMessage(event);
let model: InstanceType<typeof Model> | GpuModel | null = null;
let active: string | null = null;
let stopped = false;

let located: ((near: unknown) => void) | null = null;
async function load(manifest: any, buffer: ArrayBuffer, useGpu: boolean) {
  validateManifest(manifest, buffer.byteLength);
  emit({ type: 'loading', progress: 95, label: 'Checking model integrity…' });
  const hash = Array.from(sha256(new Uint8Array(buffer)), b => b.toString(16).padStart(2, '0')).join('');
  if (hash !== manifest.weights_sha256) throw new Error('Model integrity check failed. Reinstall the app.');
  await enableKernels();
  const cpu = new Model(manifest, buffer);


  let gpu: GpuModel | null = null, note = '';
  if (useGpu) {
    emit({ type: 'loading', progress: 97, label: 'Checking the GPU…' });
    try { gpu = await gpuModel(cpu); if (!gpu) note = 'This phone offers no GPU to apps yet, so Kani uses the CPU.'; }
    catch (error: any) { note = `Kani uses the CPU: ${error.message || error}.`; }
  }
  const loaded = model = gpu || cpu;
  if (gpu) void gpu.lost.then((info: { message?: string }) => {
    if (model !== loaded) return;
    model = null;
    emit({ type: 'gpu-lost', error: `The GPU stopped${info.message ? ` (${info.message})` : ''}, so Kani switched to the CPU.` });
  });
  emit({ type: 'ready', model: { name: manifest.name, codename: manifest.codename, parameters: manifest.parameters, sha256: hash,
    context: manifest.config.context, device: gpu ? 'gpu' : 'cpu', gpu: gpu?.name, note } });
}

const pause = () => new Promise(resolve => setTimeout(resolve, 0));

async function generate(id: string, messages: Message[], search: boolean) {
  if (!model || active) throw new Error('Kani is not ready yet.');
  const loaded = model;
  active = id; stopped = false;
  let text = '', source = 'model', saved: string | undefined;
  try {
    const question = messages[messages.length - 1].content;
    // Same limits as the website, so the app gives the same answers.
    const maxTokens = loaded.config.context >= 1024 ? 768 : 240;

    const online = search ? webRequest(question) : null;
    if (online) {
      emit({ type: 'token', id, text: online.mode === 'read' ? 'Reading Wikipedia…' : 'Searching the web…' });
      const found = await webAnswer(online, fetch);
      text = found.text; source = found.engine === 'wikipedia' ? 'web' : 'web_search';
      emit({ type: 'token', id, text });
      emit({ type: 'done', id, stopped, source });
      return;
    }
    const { retained, tool } = prepareReply(loaded, messages.map(({ role, content }) => ({ role, content })), maxTokens);
    if (tool) { [text, source] = tool; emit({ type: 'token', id, text }); }
    else {


      let held = true;
      for await (const piece of loaded.generate(retained, { maxTokens, temperature: 0 })) {
        if (stopped) break;
        text += piece;
        held = held && (maybeSearch(text) || maybePlaces(text) || maybeCide(text));
        if (!held) emit({ type: 'token', id, text });
        await pause();
      }
      const call = held && !stopped ? text.trim().match(SEARCH_CALL) : null;
      const place = held && !stopped ? text.trim().match(PLACES_CALL) : null;
      if (place && search) {

        text = ''; source = 'places';
        const locate = () => new Promise(resolve => { located = resolve; emit({ type: 'locate', id }); });
        for await (const piece of placesAnswer(place, { locate, signal: undefined })) {
          if (stopped) break;
          text += piece;
          emit({ type: 'token', id, text });
          await pause();
        }
        saved = placesSummary(place[1].trim(), place[2].trim());
      } else if (call && search) {
        text = ''; source = 'model_web';
        const generate = (turns: { role: string; content: string }[]) => loaded.generate(turns, { maxTokens: 200, temperature: 0 });


        const fits = (prompt: string) => loaded.tokenizer.encode(prompt).length + 4 + 200 <= loaded.config.context;

        for await (const piece of searchedAnswer(call[1], question, { generate, webAnswer, fits, signal: undefined })) {
          if (stopped) break;
          text += piece;
          emit({ type: 'token', id, text });
          await pause();
        }
      } else if (place || (held && /^PLACES:\s*/i.test(text.trim()))) {
        text = 'I can find places once web search is on. Turn it on in Settings, then ask again.';
        source = 'search_unavailable'; emit({ type: 'token', id, text });
      } else if (call || (held && /^SEARCH:\s*/i.test(text.trim()))) {
        text = 'I need to look that up. Turn on web search in Settings to let me search for it.';
        source = 'search_unavailable'; emit({ type: 'token', id, text });
      } else if (held && /^CIDE\s*$/.test(text)) {
        text = 'That is a job for Cide 2, Kani’s coding model. It runs on the website: open kani.kelus.org and choose Cide 2.';
        source = 'cide_unavailable'; emit({ type: 'token', id, text });
      } else if (held && text) emit({ type: 'token', id, text });
      if (!text && !stopped) emit({ type: 'token', id, text: 'Kani ended this reply without text. Try another wording.' });
    }
    emit({ type: 'done', id, stopped, source, saved });
  } finally { active = null; located = null; }
}
self.onmessage = event => {
  const request = event.data;
  if (request.type === 'stop') { stopped = true; return; }
  if (request.type === 'location') { located?.(request.near ?? null); located = null; return; }
  const task = request.type === 'load' ? load(request.manifest, request.buffer, Boolean(request.gpu)) : generate(request.id, request.messages, Boolean(request.search));
  task.catch(error => emit({ type: 'error', id: request.id, error: error.message || String(error) }));
};
