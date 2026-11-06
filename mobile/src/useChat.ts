import { useCallback, useEffect, useRef, useState } from 'react';
import { Preferences } from '@capacitor/preferences';
import { App } from '@capacitor/app';
import { completedContext, restoreChats, storable, updateHistory } from './history.mjs';
import { browserLocation } from '../../site/places.js';
import type { Chat, Message, ModelInfo, WorkerReply } from './types';
import InferenceWorker from './inference.worker?worker&inline';
import { loadWeights } from './load-weights.mjs';
import { id as newId } from './id';
import { translationAsk, translationReply } from './learn/ask';
import { canTranslate, translate } from './learn/translate';
import { mergeChats } from './sync.mjs';
import type { Person, RemoteChat } from './account';
type AccountModule = typeof import('./account');
export type Sync = 'off' | 'syncing' | 'synced' | 'offline';

const HISTORY_KEY = 'kani.mobile.chats.v1';
const SEARCH_KEY = 'kani.mobile.search';
const MODEL_KEY = 'kani.mobile.model';
const GPU_KEY = 'kani.mobile.gpu';
export interface ModelChoice { id: string; name: string; parameters: number; bytes: number }
const blank = (): Chat => ({ id: newId(), title: 'New conversation', updated: Date.now(), messages: [], model: '' });
export function useChat() {
  const [chat, setChat] = useState<Chat>(blank);
  const [history, setHistory] = useState<Chat[]>([]);
  const [ready, setReady] = useState(false);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState('Opening Kani…');
  const [progress, setProgress] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const [search, setSearchState] = useState(true);

  const [models, setModels] = useState<ModelChoice[]>([]);
  const [modelId, setModelId] = useState('otter-4-12');

  const [gpu, setGpuState] = useState(true);
  const state = useRef({ chat, history, ready: false, model: null as ModelInfo | null, worker: null as Worker | null,
    request: '', writes: Promise.resolve(), disposed: false, loadAbort: null as AbortController | null, search: true, modelId: 'otter-4-12',
    gpu: true, gpuFailed: false,

    account: null as AccountModule | null, person: null as Person | null, stopChats: null as (() => void) | null,
    known: new Set<string>(), created: new Set<string>(), pendingRemote: null as RemoteChat[] | null });
  const [person, setPerson] = useState<Person | null>(null);
  const [accountReady, setAccountReady] = useState(false);
  const [sync, setSync] = useState<Sync>('off');

  const show = useCallback((next: Chat) => { state.current.chat = next; setChat(next); }, []);
  const saveHistory = useCallback((next: Chat[]) => {
    const r = state.current;
    r.history = next; setHistory(next);
    if (!r.ready) return;
    r.writes = r.writes.then(() => Preferences.set({ key: HISTORY_KEY, value: JSON.stringify(storable(next)) }))
      .catch(() => setError('Your latest changes could not be saved. Free some storage and try again.'));
  }, []);
  const remember = useCallback(() => {
    const r = state.current;
    if (r.person) void Preferences.set({ key: `kani.mobile.synced.${r.person.uid}`, value: JSON.stringify([...r.known]) }).catch(() => {});
  }, []);
  const upload = useCallback((chat: Chat) => {
    const r = state.current, who = r.person;
    if (!who?.verified || !r.account || !chat.messages.length) return;
    const created = r.created.has(chat.id);
    r.created.add(chat.id); r.known.add(chat.id); remember();
    r.account.saveChat(who.uid, chat, created).catch(() => { if (!created) r.created.delete(chat.id); setSync('offline'); });
  }, [remember]);
  const persist = useCallback(() => {
    const r = state.current;
    saveHistory(updateHistory(r.history, r.chat));
    upload(r.chat);
  }, [saveHistory, upload]);
  const applyRemote = useCallback((list: RemoteChat[]) => {
    const r = state.current;
    if (!r.ready || !r.person) { r.pendingRemote = list; return; }
    r.pendingRemote = null;
    const { merged, upload: newer, dropped } = mergeChats(r.history, list, r.known);
    for (const chat of list) { r.known.add(chat.id); r.created.add(chat.id); }
    for (const id of dropped) { r.known.delete(id); r.created.delete(id); }
    remember();
    saveHistory(merged);
    if (!r.request) {
      const current = merged.find(c => c.id === r.chat.id);
      if (current && current.updated > r.chat.updated) show(current);
      else if (dropped.includes(r.chat.id)) show(blank());
    }
    for (const chat of newer) upload(chat);
    setSync('synced');
  }, [remember, saveHistory, show, upload]);
  const follow = useCallback(async (next: Person | null) => {
    const r = state.current;
    r.stopChats?.(); r.stopChats = null; r.pendingRemote = null; r.created = new Set(); r.known = new Set();
    r.person = next; setPerson(next);
    if (!next?.verified || !r.account) { setSync('off'); return; }
    setSync('syncing');
    const { value } = await Preferences.get({ key: `kani.mobile.synced.${next.uid}` }).catch(() => ({ value: null }));
    if (r.person !== next) return;
    r.known = new Set(JSON.parse(value || '[]'));
    r.stopChats = r.account.watchChats(next.uid, applyRemote, () => setSync('offline'));
  }, [applyRemote]);
  const finish = useCallback((status: Message['status'], source = 'model', saved?: string) => {
    const r = state.current, id = r.request;
    if (!id) return;
    show({ ...r.chat, updated: Date.now(), messages: r.chat.messages.map(m => m.id === id ? { ...m, status, source, ...(saved && { saved }) } : m) });
    r.request = ''; setBusy(false); persist();
  }, [show, persist]);
  const load = useCallback(() => {
    const r = state.current;
    r.loadAbort?.abort(); r.loadAbort = new AbortController();
    r.worker?.terminate(); r.model = null; setModel(null); setLoading('Opening Kani…'); setProgress(0);

    const worker = new InferenceWorker();
    r.worker = worker;
    worker.onmessage = ({ data }: MessageEvent<WorkerReply>) => {
      if (r.worker !== worker || r.disposed) return;
      if (data.type === 'loading') { setLoading(data.label); setProgress(data.progress); }
      if (data.type === 'ready') { r.model = data.model; setModel(data.model); setLoading(''); setProgress(100); }
      if (data.type === 'token' && data.id === r.request) {
        show({ ...r.chat, messages: r.chat.messages.map(m => m.id === data.id ? { ...m, content: data.text } : m) });
      }
      if (data.type === 'done' && data.id === r.request) finish(data.stopped ? 'stopped' : 'complete', data.source, data.saved);

      if (data.type === 'locate' && data.id === r.request) void browserLocation().then(near => { if (r.worker === worker) worker.postMessage({ type: 'location', near }); });
      if (data.type === 'error' && (!data.id || data.id === r.request)) {
        setError(data.error); setLoading(''); finish('failed');
      }
      if (data.type === 'gpu-lost') { r.gpuFailed = true; setError(data.error); finish('failed'); load(); }
    };
    worker.onerror = () => {
      if (r.worker !== worker) return;
      r.model = null; setModel(null); setLoading(''); setError('Kani stopped unexpectedly. Tap Reload model to try again.'); finish('failed');
    };
    void loadWeights(new URL(`models/${r.modelId}/`, document.baseURI).href, (progress: number, label: string) => {
      if (r.worker === worker && !r.disposed) { setLoading(label); setProgress(progress); }
    }, r.loadAbort.signal).then(({ manifest, buffer }) => {
      if (r.worker === worker && !r.disposed) worker.postMessage({ type: 'load', manifest, buffer, gpu: r.gpu && !r.gpuFailed }, [buffer]);
    }).catch(error => {
      if (r.worker === worker && !r.disposed && error.name !== 'AbortError') { setLoading(''); setError(error.message); }
    });
  }, [show, finish]);
  const stop = useCallback((background = false) => {
    const r = state.current;
    if (!r.request) return;
    r.worker?.terminate(); r.worker = null; r.model = null; setModel(null);
    finish('stopped');
    if (!background) load(); else setLoading('Paused while the app is in the background.');
  }, [finish, load]);
  const setSearch = useCallback((on: boolean) => {
    state.current.search = on; setSearchState(on);
    void Preferences.set({ key: SEARCH_KEY, value: on ? 'on' : 'off' }).catch(() => setError('Could not save the web search setting.'));
  }, []);
  const setGpu = useCallback((on: boolean) => {
    const r = state.current;
    if (r.request || on === r.gpu) return;
    r.gpu = on; r.gpuFailed = false; setGpuState(on);
    void Preferences.set({ key: GPU_KEY, value: on ? 'on' : 'off' }).catch(() => setError('Could not save the GPU setting.'));
    load();
  }, [load]);
  const chooseModel = useCallback((id: string) => {
    const r = state.current;
    if (r.request || id === r.modelId) return;
    r.modelId = id; setModelId(id);
    void Preferences.set({ key: MODEL_KEY, value: id }).catch(() => setError('Could not save your model choice.'));
    load();
  }, [load]);
  useEffect(() => {
    const r = state.current; r.disposed = false;
    void Preferences.get({ key: SEARCH_KEY }).then(({ value }) => { if (value === 'off') { r.search = false; setSearchState(false); } }).catch(() => {});
    Preferences.get({ key: HISTORY_KEY }).then(({ value }) => {
      if (r.disposed) return;
      const saved = restoreChats(value) as Chat[];
      r.history = saved; setHistory(saved); r.ready = true; setReady(true);
      if (saved[0]) show(saved[0]);
      if (r.pendingRemote) applyRemote(r.pendingRemote);
    }).catch(() => setError('Saved chats could not be read. Restart the app to retry. Your stored copy has been left intact.'));
    void Promise.all([
      fetch(new URL('models/index.json', document.baseURI).href).then(response => response.json()).catch(() => ({})),
      Preferences.get({ key: MODEL_KEY }).then(({ value }) => value).catch(() => null),
      Preferences.get({ key: GPU_KEY }).then(({ value }) => value).catch(() => null),
    ]).then(([index, saved, gpuSetting]) => {
      if (r.disposed) return;
      if (gpuSetting === 'off') { r.gpu = false; setGpuState(false); }
      const available = Object.entries(index as Record<string, Omit<ModelChoice, 'id'>>).map(([id, info]) => ({ id, ...info }));
      setModels(available);
      const preferred = [saved, 'otter-4-12', available[0]?.id].find(id => id && available.some(m => m.id === id));
      if (preferred) { r.modelId = preferred; setModelId(preferred); }
      load();
    });
    const subscription = App.addListener('appStateChange', ({ isActive }) => {
      if (!isActive) stop(true);
      else if (!r.worker) load();
    });
    return () => { r.disposed = true; r.loadAbort?.abort(); r.worker?.terminate(); void subscription.then(handle => handle.remove()); };
  }, [load, show, stop, applyRemote]);
  useEffect(() => {
    let live = true, unwatch: (() => void) | undefined;
    import('./account').then(account => {
      if (!live) return;
      state.current.account = account; setAccountReady(true);
      unwatch = account.watchPerson(next => { if (live) void follow(next); });
    }).catch(() => {});
    return () => { live = false; unwatch?.(); state.current.stopChats?.(); };
  }, [follow]);
  // The new user message's id, or false when nothing was sent.
  const send = (text: string): string | false => {
    const r = state.current;
    if (!text.trim() || !r.ready || !r.model || r.request || !r.worker) return false;
    if (text.length > 6000) { setError('Please keep each message under 6,000 characters.'); return false; }
    if (r.chat.messages.length >= 100) { setError('This conversation is full. Start a new chat to continue.'); return false; }
    const user: Message = { id: newId(), role: 'user', content: text.trim(), status: 'complete' };
    const id = newId(), messages = [...completedContext(r.chat.messages), user];
    show({ ...r.chat, model: r.model.name, title: r.chat.messages.length ? r.chat.title : text.trim().slice(0, 48),
      updated: Date.now(), messages: [...r.chat.messages, user, { id, role: 'assistant', content: '' }] });
    r.request = id; setBusy(true); setError('');

    const ask = canTranslate() ? translationAsk(text) : null;
    if (ask) {
      void translate(ask.text, ask.from, ask.to).then(out => translationReply(ask, out), (error: Error) => error.message).then(content => {
        if (r.request !== id) return;
        show({ ...r.chat, messages: r.chat.messages.map(m => m.id === id ? { ...m, content } : m) });
        finish('complete', 'local_translator');
      });
      return user.id;
    }
    r.worker.postMessage({ type: 'generate', id, messages, search: r.search });
    return user.id;
  };
  const open = (next = blank()) => { if (!state.current.request) { show(next); setError(''); } };
  const remove = (id: string) => {
    const r = state.current;
    if (r.request) return;
    saveHistory(r.history.filter(c => c.id !== id));
    if (r.chat.id === id) open();
    if (r.person?.verified && r.account && r.known.has(id)) {
      r.known.delete(id); r.created.delete(id); remember();
      r.account.deleteChat(r.person.uid, id).catch(() => setSync('offline'));
    }
  };

  const attempt = async (action: (account: AccountModule) => Promise<unknown>) => {
    const account = state.current.account;
    if (!account) return 'Accounts are still loading. Try again in a moment.';
    try { await action(account); return ''; } catch (error) { return account.friendlyError(error); }
  };
  const accountActions = {
    google: () => attempt(a => a.signInGoogle()),
    signIn: (email: string, password: string) => attempt(a => a.signInEmail(email, password)),
    create: (email: string, password: string) => attempt(a => a.createAccount(email, password)),
    resend: () => attempt(a => a.resendVerification()),
    reset: (email: string) => attempt(a => a.resetPassword(email)),
    verified: () => attempt(async a => {
      const next = await a.checkVerified();
      if (!next?.verified) throw new Error('Not verified yet. Open the link in the email we sent, then try again.');
      await follow(next);
    }),
    signOut: () => attempt(async a => { await a.signOutEverywhere(); await follow(null); }),
    canUseGoogle: () => state.current.account?.canUseGoogle() ?? false,
  };
  return { chat, history, ready, model, loading, progress, busy, error, setError, send, open, remove, stop, reload: load, search, setSearch, models, modelId, chooseModel, gpu, setGpu,
    account: { ready: accountReady, person, sync, ...accountActions } };
}
