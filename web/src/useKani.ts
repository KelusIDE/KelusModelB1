import { useCallback, useEffect, useRef, useState } from 'react';
import {
  PLACES_CALL, SEARCH_CALL, browserLocation, extractSvg, findPlaces, gpuModel, loadAccount, loadModel, maybePlaces, maybeSearch, placesPhrase, placesReply, placesSummary,
  prepareAttachments, prepareReply, searchedAnswer, spokenAnswer,
  svgPrompt, svgRequest, webAnswer, webRequest,
  type Account, type Attachment, type GpuModel, type Model, type PlacesResult, type PreparedFiles, type Role, type SavedChat, type StoredTurn, type Turn, type User,
} from './engine';
import type { TutorNote } from './English';
import type { SpokenWord } from './voice';

export const LABELS: Record<string, string> = {
  local_clock: 'LOCAL CLOCK', local_calculator: 'LOCAL CALCULATOR', web: 'WEB · WIKIPEDIA', web_search: 'WEB SEARCH',
  file_analysis: 'LOCAL FILE INSPECTION', document: 'FROM YOUR PDF', crisis_support: 'SUPPORT · FIXED SAFETY MESSAGE', places: 'PLACES SEARCH',
};

export const labelFor = (source: string, model: string) =>
  LABELS[source] || (source === 'model_web' ? `${model} · searched the web` : model).toUpperCase();
export const HINT = 'Enter to send · Shift + Enter for a new line · Temperature 0 gives repeatable answers';



export const MODELS = [
  { id: 'otter-4-12', name: 'Kani Otter 4.12', base: 'models/otter-4-12/', about: 'Newest and smartest · 206 MB' },
  { id: 'otter-4-3', name: 'Kani Otter 4.3', base: 'models/otter-4-3/', about: 'Faster to load · 89 MB' },
  { id: 'big-v2', name: 'Big Kani v2 (experimental)', base: 'models/big-v2/', about: 'Experimental, 205.6M parameters · 206 MB' },
  { id: 'cide-4-5', name: 'Cide 4.5', base: 'models/cide-4-5/', about: 'Coding: better Python, 19 languages, big websites, follows the chat. Kel programs are often wrong · 206 MB' },
  { id: 'cide-4-3', name: 'Cide 4.3', base: 'models/cide-4-3/', about: 'Previous coder · 206 MB' },
  { id: 'cide-4-2', name: 'Cide 4.2', base: 'models/cide-4-2/', about: 'Older coder · 206 MB' },
  { id: 'cide-4-1', name: 'Cide 4.1', base: 'models/cide-4-1/', about: 'Older coder · 206 MB' },
  { id: 'cide-3', name: 'Cide 3', base: 'models/cide-3/', about: 'Older coder · 206 MB' },
  { id: 'cide-2', name: 'Cide 2', base: 'models/cide-2/', about: 'Shorter code, faster to load · 89 MB' },
  { id: 'seal-3', name: 'Kani Seal 3', base: 'model/', about: 'Smaller and quicker · 35 MB' },
] as const;
export type ModelId = typeof MODELS[number]['id'];
const MODEL_KEY = 'kani-model';
const savedModel = (): ModelId => {
  try {
    const id = localStorage.getItem(MODEL_KEY);
    return MODELS.find(m => m.id === id)?.id ?? MODELS[0].id;
  } catch { return MODELS[0].id; }
};


export interface Item { id: string; role: Role; content: string; source: string; receipt?: string; done: boolean; by?: string; places?: PlacesResult;
  note?: TutorNote; heard?: SpokenWord[] }
// English practice: the tutor's note and the spoken words go on the user's message; onReply gets the finished reply (to read it aloud).
export interface SendExtras { note?: TutorNote; heard?: SpokenWord[]; onReply?: (reply: string, source: string) => void }
interface TranscriptTurn extends StoredTurn { excludeFromContext?: boolean }



const channel = new MessageChannel();
const nextFrame = () => new Promise<void>(resolve => {
  if (!document.hidden) { requestAnimationFrame(() => resolve()); return; }
  channel.port1.onmessage = () => resolve();
  channel.port2.postMessage(null);
});
const plain = (messages: StoredTurn[]): TranscriptTurn[] =>
  messages.map(({ role, content, source }) => source ? { role, content, source } : { role, content });
const toItems = (turns: TranscriptTurn[]): Item[] =>
  turns.map(turn => ({ id: crypto.randomUUID(), role: turn.role, content: turn.content, source: turn.source || 'model', done: true }));
export const hasSvg = (reply: string) => { try { extractSvg(reply); return true; } catch { return false; } };
const message = (error: unknown) => (error as Error)?.message || String(error);


export function useKani(onNewChat: () => void) {
  const [modelId, setModelId] = useState<ModelId>(savedModel);
  const [model, setModel] = useState<Model | null>(null);
  const [status, setStatus] = useState<'loading' | 'ready' | 'failed'>('loading');
  const [engine, setEngine] = useState('');
  const [items, setItems] = useState<Item[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [info, setInfo] = useState(HINT);
  const [updateAvailable, setUpdateAvailable] = useState(false);
  const [account, setAccount] = useState<Account | null>(null);
  const [accountState, setAccountState] = useState<'loading' | 'ready' | 'unavailable'>('loading');
  const [user, setUser] = useState<User | null>(null);
  const [savedChats, setSavedChats] = useState<SavedChat[]>([]);
  const [chatId, setChatId] = useState<string>(() => crypto.randomUUID());


  const transcript = useRef<TranscriptTurn[]>([]);
  const refs = useRef({ model: null as Model | null, coding: false, lastAskOnly: false, busy: false, stopped: false, aborter: null as AbortController | null,
    chatId, saved: false, private: false, account: null as Account | null, user: null as User | null,
    stopWatchingChats: null as null | (() => void), onNewChat });
  refs.current.onNewChat = onNewChat;

  const startChat = useCallback((messages: StoredTurn[] = [], id: string = crypto.randomUUID(), saved = false) => {
    const r = refs.current;
    if (r.busy) return;
    r.onNewChat(); r.private = false;
    transcript.current = plain(messages); r.chatId = id; r.saved = saved;
    setChatId(id); setItems(toItems(transcript.current)); setError('');
  }, []);

  const persist = useCallback(async () => {
    const r = refs.current;
    if (r.private || !r.account || !r.user || !transcript.current.length) return;

    const created = r.saved; r.saved = true;
    try {
      await r.account.saveChat(r.user.uid, r.chatId, transcript.current, { created, model: r.model?.manifest.name });
    } catch (failure) {
      r.saved = created;
      setError('This chat was not saved: ' + r.account.friendlyError(failure));
    }
  }, []);

  const send = useCallback(async (raw: string, attached: Attachment[], temperatureSetting: number, extras: SendExtras = {}): Promise<boolean> => {
    const r = refs.current, model = r.model;
    if (r.busy || !model || (!raw.trim() && !attached.length)) return false;
    const withFiles = attached.length > 0;
    const prompt = raw.trim() || 'Analyze these files.';
    let fileTurn: PreparedFiles | undefined;
    try { if (withFiles) fileTurn = prepareAttachments(prompt, attached, model.tokenizer, model.config.context, model.config.context >= 1024 ? 768 : 240); }
    catch (failure) { setError(message(failure)); return false; }
    if (withFiles) r.private = true;
    r.busy = true; r.stopped = false; setBusy(true); setError('');
    const shown = prompt + (withFiles ? '\nAttached: ' + attached.map(f => f.name).join(', ') : '');
    const replyId = crypto.randomUUID();
    setItems(previous => [...previous,
      { id: crypto.randomUUID(), role: 'user', content: shown, source: 'model', done: true, note: extras.note, heard: extras.heard },
      { id: replyId, role: 'assistant', content: '', source: 'model', done: false, by: model.manifest.name }]);
    const update = (patch: Partial<Item>) => setItems(previous => previous.map(item => item.id === replyId ? { ...item, ...patch } : item));
    transcript.current.push({ role: 'user', content: shown, excludeFromContext: withFiles });
    let reply = '', source = 'model', saved = '';
    const started = performance.now();
    try {
      const temperature = Math.min(2, Math.max(0, temperatureSetting || 0));
      // A searched answer goes back to the model as the answer alone, without the search line and the sources.
      let context: Turn[] = transcript.current.filter(turn => !turn.excludeFromContext).slice(-15)
        .map(({ role, content, source }) => ({ role, content: source === 'model_web' ? spokenAnswer(content) : content }));
      if (fileTurn) context = [{ role: 'user', content: fileTurn.text }];
      // Cide 4.2 and older learned from single requests: an earlier page in the chat derails the next ask, so they see only the latest.
      else if (r.lastAskOnly) context = context.slice(-1);
      if (context[0]?.role !== 'user') context = context.slice(1);
      const drawing = withFiles ? null : svgRequest(context[context.length - 1].content);
      if (drawing) context = [{ role: 'user', content: svgPrompt(drawing) }];
      const maxTokens = r.coding ? Math.min(4096, model.config.context) : model.config.context >= 1024 ? 768 : drawing ? Math.min(512, model.config.context) : 240;
      const online = withFiles ? null : webRequest(prompt);
      const { retained, omitted, tool } = online || fileTurn?.helper ? { retained: [], omitted: 0, tool: null } : prepareReply(model, context, maxTokens);
      if (fileTurn?.helper) {
        reply = fileTurn.text; source = fileTurn.source || 'file_analysis';
      } else if (online) {
        r.aborter = new AbortController();
        update({ content: online.mode === 'read' ? 'Reading Wikipedia…' : 'Searching the web…', source: online.mode === 'read' ? 'web' : 'web_search' });
        const found = await webAnswer(online, fetch, r.aborter.signal);
        reply = found.text; source = found.engine === 'wikipedia' ? 'web' : 'web_search';
      } else if (tool) {
        [reply, source] = tool;
      } else {


        let painted = performance.now(), held = !drawing && !withFiles;
        for await (const text of model.generate(retained, { maxTokens, temperature })) {
          reply += text;
          held = held && (maybeSearch(reply) || maybePlaces(reply));
          if (performance.now() - painted > 30) { update({ content: held ? 'Thinking…' : reply }); await nextFrame(); painted = performance.now(); }
          if (r.stopped) break;
        }
        const call = held && !r.stopped ? reply.trim().match(SEARCH_CALL) : null;
        const place = held && !r.stopped ? reply.trim().match(PLACES_CALL) : null;
        if (place) {

          reply = ''; source = 'places'; r.aborter = new AbortController();
          update({ content: `Looking for ${placesPhrase(place[1].trim(), place[2].trim())}…`, source });
          const found = await findPlaces(place, { locate: browserLocation, signal: r.aborter.signal });
          reply = placesReply(found);
          update({ content: reply, places: found.status === 'found' ? found : undefined });
          saved = placesSummary(place[1].trim(), place[2].trim());
        } else if (call) {
          reply = ''; source = 'model_web'; r.aborter = new AbortController();
          update({ content: '', source });
          const generate = (messages: Turn[]) => model.generate(messages, { maxTokens: 200, temperature: 0 });

          const fits = (text: string) => model.tokenizer.encode(text).length + 4 + 200 <= model.config.context;
          for await (const piece of searchedAnswer(call[1], prompt, { generate, webAnswer, signal: r.aborter.signal, fits })) {
            reply += piece; update({ content: reply });
            await nextFrame();
            if (r.stopped) break;
          }
        }
      }
      update({ content: reply || '(The model ended its reply without text. Try another wording.)', source, done: true,
        receipt: fileTurn && !fileTurn.helper ? fileTurn.receipt : undefined });
      if (drawing && !(source === 'model' && hasSvg(reply)) && !r.stopped) setError('The model did not produce a complete, supported SVG. Its response is shown above.');
      setInfo(`${source.startsWith('web') ? 'Retrieved online' : source === 'model_web' ? 'Searched the web, then answered on this device' : source === 'places' ? 'Looked up on a map service' : source === 'document' ? 'Read your PDF on this device' : 'Generated on this device'} in ${((performance.now() - started) / 1000).toFixed(2)}s`
        + (omitted ? ' · Older turns omitted to fit this model' : '') + (r.private ? ' · Attachment chat stays local; not saved to your account.' : ' · Review and test generated code.'));
      transcript.current.push({ role: 'assistant', content: saved || reply, source, excludeFromContext: withFiles });
      void persist();
      if (!r.stopped) extras.onReply?.(reply, source);
    } catch (failure) {
      setError(message(failure));
      transcript.current.pop();
      update({ done: true, ...(reply ? {} : { content: '(No response generated.)' }) });
    } finally {
      r.aborter = null; r.busy = false; setBusy(false);
    }
    return true;
  }, [persist]);

  const stop = useCallback(() => { refs.current.stopped = true; refs.current.aborter?.abort(); }, []);



  const onChats = useCallback((chats: SavedChat[]) => {
    const r = refs.current;
    setSavedChats(chats);
    const current = chats.find(chat => chat.id === r.chatId);

    if (current && !current.pending && !r.busy && !r.private && JSON.stringify(plain(current.messages || [])) !== JSON.stringify(transcript.current)) {
      transcript.current = plain(current.messages || []); r.saved = true; setItems(toItems(transcript.current));
    }
  }, []);

  const onUser = useCallback((next: User | null) => {
    const r = refs.current, wasSignedIn = Boolean(r.user);
    r.user = next; setUser(next);
    r.stopWatchingChats?.(); r.stopWatchingChats = null; setSavedChats([]);
    if (next && r.account) {
      const account = r.account;
      r.stopWatchingChats = account.watchChats(next.uid, onChats, failure => setError('Saved chats are unavailable: ' + account.friendlyError(failure)));
      if (transcript.current.length && !r.saved) void persist();
    } else if (wasSignedIn) {
      startChat();
    }
  }, [onChats, persist, startChat]);

  const deleteChat = useCallback(async (chat: SavedChat) => {
    const r = refs.current;
    if (!r.account || !r.user || !confirm('Delete this chat from your account? This cannot be undone.')) return;
    try {
      await r.account.deleteChat(r.user.uid, chat.id);
      if (chat.id === r.chatId) startChat();
    } catch (failure) { setError('Could not delete the chat: ' + r.account.friendlyError(failure)); }
  }, [startChat]);




  const base = MODELS.find(m => m.id === modelId)!.base;
  useEffect(() => {
    let live = true, gpu: GpuModel | null = null;
    refs.current.model = null; setModel(null); setStatus('loading'); setUpdateAvailable(false); setEngine('');
    const use = (chosen: Model, name: string) => { refs.current.model = chosen; setModel(chosen); setEngine(name); };
    loadModel(base).then(async loaded => {
      if (!live) return;
      // WebGPU where the browser has it; the WebAssembly engine without it, or once the GPU is lost.
      gpu = await gpuModel(loaded).catch(() => null);
      if (!live) { gpu?.destroy(); return; }
      refs.current.coding = modelId.startsWith('cide');
      refs.current.lastAskOnly = /^cide-(2|3|4-1|4-2)$/.test(modelId);
      const onGpu = gpu;
      if (onGpu) void onGpu.lost.then(() => { if (live && refs.current.model === onGpu) use(loaded, ''); });
      use(onGpu || loaded, onGpu?.name || ''); setStatus('ready');
    }).catch(failure => { if (live) { setStatus('failed'); setError(message(failure)); } });
    return () => { live = false; gpu?.destroy(); };
  }, [base, modelId]);

  const chooseModel = useCallback((id: ModelId) => {
    if (refs.current.busy) return;
    try { localStorage.setItem(MODEL_KEY, id); } catch {  }
    setError(''); setModelId(id);
  }, []);

  useEffect(() => {
    let live = true;
    loadAccount().then(module => {
      if (!live) return;
      refs.current.account = module; setAccount(module); setAccountState('ready');
      module.watchUser(onUser);
    }).catch(() => { if (live) setAccountState('unavailable'); });
    return () => { live = false; };
  }, [onUser]);


  useEffect(() => {
    const check = async () => {
      const current = refs.current.model;
      if (!current) return;
      try {
        const latest = await (await fetch(base + 'b1.json', { cache: 'no-store' })).json();
        if (refs.current.model === current && latest.weights_sha256 !== current.manifest.weights_sha256) setUpdateAvailable(true);
      } catch {  }
    };
    const timer = setInterval(check, 10 * 60 * 1000);
    const visible = () => { if (document.visibilityState === 'visible') void check(); };
    document.addEventListener('visibilitychange', visible);
    return () => { clearInterval(timer); document.removeEventListener('visibilitychange', visible); };
  }, [base]);

  return { modelId, chooseModel, model, engine, status, items, busy, error, setError, info, updateAvailable, account, accountState, user, savedChats, chatId,
    send, stop, startChat, deleteChat, privateChat: () => refs.current.private };
}
