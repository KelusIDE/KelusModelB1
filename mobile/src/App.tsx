import { useEffect, useRef, useState, type CSSProperties, type FormEvent } from 'react';
import { App as NativeApp } from '@capacitor/app';
import { Capacitor } from '@capacitor/core';
import { Preferences } from '@capacitor/preferences';
import { Share } from '@capacitor/share';
import { useChat } from './useChat';
import { Icon } from './Icon';
import { Copy, Reply } from './Reply';
import { Learn } from './Learn';
import { AccountCard } from './AccountCard';
import { useUpdate } from './useUpdate';
import { megabytes } from './update-manifest.mjs';
import { version as appVersion } from '../package.json';
import otter from '../../brand/kani-otter-4/kani-otter-4.svg';
import { EnglishWelcome, ModeSwitch, SpokenText, TutorCard, UNCLEAR, VoiceStatus, useMode, useVoice, useWords, type TutorNote } from '../../web/src/English';
import { canRecord, primeAudio, recordSpeech, speak, stopSpeaking, transcribe, type SpokenWord } from '../../web/src/voice';
import { spokenFeedback, tutor } from '../../site/english.js';
import { spokenAnswer } from '../../site/reading.js';
import './english-app.css';

type Tab = 'chat' | 'learn' | 'history' | 'settings';
type Theme = 'system' | 'light' | 'dark';
const starters = [
  { icon: 'chat' as const, title: 'Talk it through', description: 'A question, an idea, a little curiosity.', prompt: 'Hello! What can you help me with?' },
  { icon: 'write' as const, title: 'Find the words', description: 'Start with a thought. Make it yours.', prompt: 'Write a short friendly thank-you note.' },
  { icon: 'code' as const, title: 'Make something', description: 'Explore a small piece of code.', prompt: 'Write a Python function that checks whether a number is even.' },
];

export default function App() {
  const kani = useChat();
  const updater = useUpdate();
  const offer = updater.update && !updater.hidden && ['available', 'downloading', 'permission', 'installing', 'error'].includes(updater.phase) ? updater.update : null;
  const [tab, setTab] = useState<Tab>('chat');
  const [draft, setDraft] = useState('');
  const [theme, setTheme] = useState<Theme>('system');
  const [themeReady, setThemeReady] = useState(false);
  const [search, setSearch] = useState('');
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const scroll = useRef<HTMLDivElement>(null), textarea = useRef<HTMLTextAreaElement>(null);
  const follow = useRef(true);
  const navigation = useRef({ tab, deleteId }); navigation.current = { tab, deleteId };
  const learnBack = useRef<(() => boolean) | null>(null);
  const modelName = kani.model?.name || 'Kani';
  const active = kani.chat.messages.length > 0;
  // English practice (the website's Train English / Speak English): the coach's notes are kept by message id.
  const [mode, setMode] = useMode();
  const english = mode !== 'chat';
  const words = useWords(english);
  const voice = useVoice();
  const [notes, setNotes] = useState<Record<string, { note?: TutorNote; heard?: SpokenWord[] }>>({});
  const [talking, setTalking] = useState<'idle' | 'recording' | 'checking'>('idle');
  const [level, setLevel] = useState(0);
  const stopper = useRef<AbortController | null>(null), turn = useRef(0), sayReplyTo = useRef<string | null>(null);
  const coach = (feedback: string) => { if (feedback) void voice.ensure('speak').then(() => speak(feedback)).catch(() => {}); };
  const sendText = (text: string, heard?: SpokenWord[]) => {
    const note = english && words && text.trim() ? tutor(text.trim(), words, turn.current++) : undefined;
    const id = kani.send(text);
    if (!id) return null;
    if (note || heard) setNotes(all => ({ ...all, [id]: { note, heard } }));
    if (mode === 'speak') sayReplyTo.current = id;
    return { id, note };
  };
  // Speak mode: record until you stop talking, write it down, coach it out loud, then send it to Kani.
  const talk = async () => {
    if (talking === 'recording') { stopper.current?.abort(); return; }
    if (!words || kani.busy || !kani.model) return;
    primeAudio(); stopSpeaking(); kani.setError('');
    try {
      await voice.ensure('listen');
      void voice.ensure('speak').catch(() => {});
      stopper.current = new AbortController(); setTalking('recording');
      const audio = await recordSpeech({ onLevel: setLevel, signal: stopper.current.signal });
      if (!audio) { kani.setError("I didn't hear anything. Tap Speak and talk."); return; }
      setTalking('checking');
      const { heard, scored } = await transcribe(audio);
      const text = heard.text.trim();
      if (!text) { kani.setError("I couldn't make out any words. Try again a little closer to the phone."); return; }
      follow.current = true;
      const sent = sendText(text, scored);
      const unclear = scored.find((w, i) => w.probability < UNCLEAR && /[a-z]/i.test(w.text) && !(i && /^[A-Z]/.test(w.text)))?.text.replace(/[^\w']/g, '');
      if (sent?.note) coach(spokenFeedback(sent.note) || (unclear ? `I wasn't sure I heard "${unclear}". Say it like this: ${unclear}.` : ''));
    } catch (failure) {
      kani.setError((failure as Error).name === 'NotAllowedError' ? 'Kani needs the microphone to listen. Allow it in the phone settings and try again.' : (failure as Error).message);
    } finally { setTalking('idle'); stopper.current = null; }
  };
  // In Speak mode Kani's finished reply is read aloud: the answer without its sources, a few sentences at most.
  useEffect(() => {
    const id = sayReplyTo.current;
    if (kani.busy || !id) return;
    const at = kani.chat.messages.findIndex(m => m.id === id), reply = kani.chat.messages[at + 1];
    if (at < 0 || reply?.role !== 'assistant') return;
    sayReplyTo.current = null;
    if (reply.status !== 'complete' || !reply.content) return;
    const text = reply.source === 'model_web' ? spokenAnswer(reply.content) : reply.source === 'places' ? reply.content.split('\n')[0] : reply.content;
    void speak(text.length > 450 ? (text.slice(0, 450).match(/^[\s\S]*[.!?](?=\s)/)?.[0] || text.slice(0, 450)) : text);
  }, [kani.busy, kani.chat.messages]);

  useEffect(() => {
    void Preferences.get({ key: 'kani.mobile.theme' }).then(({ value }) => {
      if (['system', 'light', 'dark'].includes(value || '')) setTheme(value as Theme);
      setThemeReady(true);
    }).catch(() => setThemeReady(true));
    const listener = NativeApp.addListener('backButton', () => {
      if (navigation.current.deleteId) setDeleteId(null);
      else if (navigation.current.tab === 'learn' && learnBack.current?.()) return;
      else if (navigation.current.tab !== 'chat') setTab('chat');
      else void NativeApp.minimizeApp();
    });
    return () => { void listener.then(handle => handle.remove()); };
  }, []);
  useEffect(() => {
    const media = matchMedia('(prefers-color-scheme: dark)');
    const apply = () => { document.documentElement.dataset.theme = theme === 'system' ? media.matches ? 'dark' : 'light' : theme; };
    apply(); media.addEventListener('change', apply);
    if (themeReady) void Preferences.set({ key: 'kani.mobile.theme', value: theme }).catch(() => kani.setError('Could not save your appearance setting.'));
    return () => media.removeEventListener('change', apply);
  }, [theme, themeReady]);
  useEffect(() => {
    const element = textarea.current;
    if (element) { element.style.height = 'auto'; element.style.height = `${Math.min(element.scrollHeight, 140)}px`; }
  }, [draft, tab]);
  useEffect(() => {
    if (scroll.current) scroll.current.scrollTop = active && follow.current ? scroll.current.scrollHeight : 0;
  }, [kani.chat.messages, tab, active]);
  const newChat = () => { if (kani.busy) return; kani.open(); setDraft(''); setTab('chat'); follow.current = true; };
  const submit = (event: FormEvent) => {
    event.preventDefault(); follow.current = true;
    if (mode === 'speak') { primeAudio(); stopSpeaking(); }
    const sent = sendText(draft);
    if (sent) { setDraft(''); textarea.current?.focus(); if (mode === 'speak' && sent.note) coach(spokenFeedback(sent.note)); }
  };
  const share = async () => {
    const text = kani.chat.messages.map(m => `${m.role === 'user' ? 'You' : kani.chat.model}:\n${m.content}`).join('\n\n');
    try { await Share.share({ title: kani.chat.title, text, dialogTitle: 'Share this conversation' }); }
    catch (error) {
      if (!/cancel|abort/i.test(String(error))) kani.setError('Sharing is unavailable. You can still copy individual replies.');
    }
  };

  return <div className="app-shell">
    <header className="topbar">
      <button className="brand" onClick={() => setTab('chat')} aria-label="Kani home"><img src={otter} alt="" /><span>kani<span className="brand-dot">.</span></span></button>
      <div className="top-actions"><span className={`local-badge ${kani.model ? 'ready' : ''}`}><span />{kani.model ? kani.model.device === 'gpu' ? 'ON DEVICE · GPU' : 'ON DEVICE' : 'GETTING READY'}</span>
        <button className="icon-button" onClick={newChat} disabled={kani.busy || !kani.ready} aria-label="New conversation"><Icon name="plus" /></button></div>
    </header>

    {offer && <div className="update-banner" role="status">
      <span className="update-icon"><Icon name="download" size={18} /></span>
      <div><strong>Kani {offer.versionName} is ready</strong><small>{
        updater.phase === 'downloading' ? `Downloading… ${Math.round(updater.progress * 100)}%`
        : updater.phase === 'permission' ? 'Turn on “Allow from this source” for Kani in the screen that opened, then come back.'
        : updater.phase === 'installing' ? 'Tap Update in the installer. Your chats and settings stay.'
        : updater.phase === 'error' ? updater.problem
        : `${offer.notes[0] ?? 'Improvements and fixes'} · ${megabytes(offer.size)}`}</small>
        {updater.phase === 'downloading' && <progress max={1} value={updater.progress} aria-label="Update download" />}</div>
      {updater.phase === 'available' || updater.phase === 'error' ? <><button className="update-later" onClick={updater.hide}>Later</button><button className="update-go" onClick={() => void updater.start()}>Update</button></>
        : updater.phase === 'permission' || updater.phase === 'installing' ? <button className="update-go" onClick={() => void updater.install()}>{updater.phase === 'permission' ? 'Allow' : 'Install'}</button> : null}
    </div>}
    {kani.error && <div className="error" role="alert"><span>{kani.error}</span><button className="icon-button" aria-label="Dismiss error" onClick={() => kani.setError('')}><Icon name="close" size={18} /></button></div>}

    {tab === 'chat' && <>
      <main className="chat-scroll" ref={scroll} onScroll={() => {
        const node = scroll.current; if (node) follow.current = node.scrollHeight - node.scrollTop - node.clientHeight < 100;
      }}>
        {!active && english ? <EnglishWelcome mode={mode} composer={null} disabled={!kani.model || kani.busy} words={words} voice={voice}
          onPick={prompt => { setDraft(prompt); textarea.current?.focus(); }} />
        : !active ? <div className="welcome">
          <div className="welcome-mark"><span className="orbit orbit-one" /><span className="orbit orbit-two" /><img src={otter} alt="Kani otter mascot" /><span className="hello-note">oh, hey there.</span></div>
          <p className="eyebrow">YOUR POCKET COMPANION</p>
          <h1>A little curious.<br /><span>A lot of Kani.</span></h1>
          <p className="welcome-copy">A space for your questions,<br />half-formed ideas, and next little project.</p>
          <div className="starters">{starters.map(s => <button className="starter" key={s.title} onClick={() => { setDraft(s.prompt); textarea.current?.focus(); }}>
            <span className="starter-icon"><Icon name={s.icon} /></span><span><strong>{s.title}</strong><small>{s.description}</small></span><Icon name="chevron" size={18} />
          </button>)}</div>
          <div className="private-note"><Icon name="shield" size={15} /><span>{kani.account.person?.verified ? 'Chats stay on this device and in your Kelus account.' : 'Just you and Kani. Chats stay on this device.'}</span></div>
        </div> : <div className="conversation">
          <div className="conversation-heading"><p>{kani.chat.model || modelName}</p><button className="copy" disabled={kani.busy} onClick={() => void share()}><Icon name="share" size={16} />Share</button></div>
          {kani.chat.messages.map(item => <article className={`message ${item.role}`} key={item.id}>
            {item.role === 'assistant' && <div className="speaker"><img src={otter} alt="" /><span>Kani</span></div>}
            <div className="message-body">{item.role === 'user' ? notes[item.id]?.heard ? <SpokenText heard={notes[item.id].heard!} words={words} voice={voice} /> : <p>{item.content}</p> : item.content ? <Reply text={item.content} /> : item.status ? <p className="muted">{item.status === 'failed' ? 'No reply generated.' : 'Reply stopped.'}</p> : <div className="thinking" role="status"><span /><span /><span /><span className="sr-only">Kani is thinking</span></div>}</div>
            {item.role === 'user' && notes[item.id]?.note && <TutorCard note={notes[item.id].note!} voice={voice} />}
            {item.role === 'assistant' && item.status && <div className="message-footer">{item.content && <Copy text={item.content} />}
              <span>{item.status === 'stopped' ? 'Stopped' : item.status === 'failed' ? 'Reply failed' : item.source === 'local_calculator' ? 'Local calculator' : item.source === 'local_clock' ? 'Local clock' : item.source === 'search_unavailable' ? 'Web search is off' : item.source === 'cide_unavailable' ? 'Coding model is on the web' : item.source === 'model_web' ? 'Searched the web' : item.source === 'web' ? 'Wikipedia' : item.source === 'web_search' ? 'Web results' : item.source === 'places' ? 'Places search' : item.source === 'local_translator' ? 'Translated on this phone' : ''}</span></div>}
          </article>)}
        </div>}
      </main>
      <section className="composer-area" aria-label="Message Kani">
        {kani.loading && <div className="load-status" role="status"><div><span>{kani.loading}</span><span>{Math.round(kani.progress)}%</span></div><progress max="100" value={kani.progress} aria-label="Model loading" /></div>}
        {!kani.model && !kani.loading && <button className="reload" onClick={kani.reload}>Reload model</button>}
        <div className="composer-modes">
          <ModeSwitch mode={mode} disabled={kani.busy || talking !== 'idle'} onChange={next => { stopSpeaking(); setMode(next); }} />
          {english && <VoiceStatus voice={voice} />}
        </div>
        <form className="composer" onSubmit={submit}>
          <textarea ref={textarea} value={draft} onChange={e => setDraft(e.target.value)} aria-label="Message" rows={1} maxLength={6000}
            placeholder={mode === 'train' ? 'Write in English. Kani checks it…' : mode === 'speak' ? 'Tap Speak, or type…' : "What's on your mind?"}
            onKeyDown={e => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey) && !e.nativeEvent.isComposing) { e.preventDefault(); if (sendText(draft)) setDraft(''); } }} />
          {mode === 'speak' && !kani.busy && <button type="button" className={`mic-button${talking === 'recording' ? ' live' : ''}`} style={{ '--level': level } as CSSProperties}
            disabled={!canRecord() || !words || !kani.model || !kani.ready || talking === 'checking'} onClick={() => void talk()} aria-label={talking === 'recording' ? 'Stop recording' : 'Speak'}>
            {talking === 'recording' ? '■ Done' : talking === 'checking' ? 'Listening…' : voice.state.listen === 'loading' ? 'Loading…' : '● Speak'}</button>}
          {kani.busy ? <button className="send-button" type="button" onClick={() => kani.stop()} aria-label="Stop generating"><Icon name="stop" size={20} /></button>
            : <button className="send-button" type="submit" disabled={!draft.trim() || !kani.model || !kani.ready} aria-label="Send message"><Icon name="arrow" /></button>}
        </form>
        <p className="composer-note">{kani.busy ? 'Thinking on your device…' : `${modelName} · Experimental. Check important answers.`}</p>
      </section>
    </>}

    {tab === 'history' && <main className="page">
      <p className="eyebrow">PICK UP WHERE YOU LEFT OFF</p><h1>Your conversations<span className="brand-dot">.</span></h1>
      <p className="page-subtitle">A few good thoughts, kept close.</p>
      {kani.account.person?.verified ? <p className="sync-line on"><span />{kani.account.sync === 'offline' ? 'Offline · syncs when you are back online' : `Synced with ${kani.account.person.email}`}</p>
        : <p className="sync-line"><span />On this phone only · <button onClick={() => setTab('settings')}>Sign in to sync</button></p>}
      <input className="search-input" value={search} onChange={e => setSearch(e.target.value)} placeholder="Find a conversation" aria-label="Find a conversation" type="search" />
      {!kani.history.length && <div className="empty-state"><Icon name="history" size={40} /><h2>A fresh start.</h2><p>Your conversations will appear here.</p><button className="primary" onClick={newChat}>Start a conversation</button></div>}
      {kani.history.length > 0 && !kani.history.some(c => c.title.toLowerCase().includes(search.toLowerCase())) && <p className="muted">No conversations match that search.</p>}
      <div className="history-list">{kani.history.filter(c => c.title.toLowerCase().includes(search.toLowerCase())).map(c => <div className="history-row" key={c.id}>
        <button onClick={() => { kani.open(c); setDraft(''); setTab('chat'); follow.current = true; }} disabled={kani.busy}><strong>{c.title}</strong><small>{c.model} · {new Date(c.updated).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}</small></button>
        <button className="icon-button" disabled={kani.busy} aria-label={`Delete ${c.title}`} onClick={() => setDeleteId(c.id)}><Icon name="trash" size={18} /></button>
      </div>)}</div>
      {kani.history.length > 0 && <p className="storage-note">{kani.account.person?.verified ? 'Your conversations are saved on this device and in your Kelus account, where kelus-b1.pages.dev shows them too.' : 'Your latest 30 conversations are saved on this device. Sign in to a Kelus account to keep them there too.'}</p>}
    </main>}

    {tab === 'learn' && <Learn backRef={learnBack} ask={prompt => { if (kani.busy) return; kani.open(); setDraft(prompt); setTab('chat'); follow.current = true; }} />}

    {tab === 'settings' && <main className="page settings">
      <p className="eyebrow">MAKE YOURSELF AT HOME</p><h1>A little more you<span className="brand-dot">.</span></h1>
      <div className="model-card"><img src={otter} alt="" /><div><small>MODEL ON THIS DEVICE</small><h2>{modelName}</h2><p>{kani.model ? `${(kani.model.parameters / 1e6).toFixed(1)}M parameters · Runs offline${kani.model.device === 'gpu' ? ' on the GPU' : ''}` : kani.loading || 'Model unavailable'}</p></div></div>
      <AccountCard account={kani.account} />
      {updater.supported && <section className="settings-section"><h2><Icon name="download" size={19} />Updates</h2>
        <p>Version {updater.version.name || appVersion} ({updater.version.build}). Kani checks GitHub when it opens. Tap Update to download a new version, then confirm in Android’s installer. Your chats and settings stay.</p>
        {updater.update && updater.phase === 'available' ? <button className="auth-button main" onClick={() => void updater.start()}>Update to {updater.update.versionName} · {megabytes(updater.update.size)}</button>
          : ['permission', 'installing'].includes(updater.phase) ? <button className="auth-button main" onClick={() => void updater.install()}>{updater.phase === 'permission' ? 'Allow updates in Android settings' : 'Open installer again'}</button>
          : <button className="auth-button quiet" disabled={['checking', 'downloading'].includes(updater.phase)} onClick={() => void updater.check()}>
            {updater.phase === 'checking' ? 'Checking…' : updater.phase === 'downloading' ? `Downloading… ${Math.round(updater.progress * 100)}%` : 'Check for updates'}</button>}
        {updater.phase === 'current' && <p className="auth-note good">You have the latest version.</p>}
        {updater.phase === 'error' && <p className="auth-note bad">{updater.problem}</p>}
      </section>}
      <section className="settings-section"><h2><Icon name="moon" size={19} />Appearance</h2><div className="segmented" role="group" aria-label="Appearance">{(['system', 'light', 'dark'] as Theme[]).map(t => <button key={t} aria-pressed={theme === t} onClick={() => setTheme(t)}>{t[0].toUpperCase() + t.slice(1)}</button>)}</div></section>
      <section className="settings-section"><h2><Icon name="spark" size={19} />Model</h2>
        <div className="model-options" role="radiogroup" aria-label="Model">{kani.models.map(m => <label key={m.id} className={`model-option ${kani.modelId === m.id ? 'selected' : ''}`}>
          <input type="radio" name="model" checked={kani.modelId === m.id} disabled={kani.busy} onChange={() => kani.chooseModel(m.id)} />
          <span><strong>{m.name}</strong><small>{m.parameters > 20e6 ? 'Smarter answers, slower replies' : 'Faster replies, simpler answers'} · {(m.parameters / 1e6).toFixed(m.parameters > 20e6 ? 0 : 1)}M parameters · {Math.round(m.bytes / 1e6)} MB</small></span>
        </label>)}</div></section>
      <section className="settings-section"><h2><Icon name="bolt" size={19} />Speed</h2>
        <label className="switch-row"><span><strong>Use the GPU</strong><small>{!kani.gpu ? 'Off: Kani runs on the CPU.'
          : !kani.model ? 'Kani looks for a GPU when the model opens.'
          : kani.model.device === 'gpu' ? `Running on the GPU (${kani.model.gpu}): prompts and replies are several times faster.` : kani.model.note || 'Kani runs on the CPU.'}</small></span>
          <input type="checkbox" role="switch" checked={kani.gpu} disabled={kani.busy} onChange={e => kani.setGpu(e.target.checked)} /></label></section>
      <section className="settings-section"><h2><Icon name="search" size={19} />Web search</h2>
        <label className="switch-row"><span><strong>Search the web for facts</strong><small>When a question needs facts, Kani looks it up and answers with numbered sources; when you ask where to go, it finds places on a map service. Only the search words are sent, to Kani's search service or Wikipedia, never your conversation. For places near you, Kani asks for your location and sends it rounded to about a kilometre.</small></span>
          <input type="checkbox" role="switch" checked={kani.search} onChange={e => kani.setSearch(e.target.checked)} /></label></section>
      <section className="settings-section"><h2><Icon name="shield" size={19} />A quiet corner of the internet</h2><p>Messages and replies stay on this device. Signing in is optional: then your chats are also kept in your Kelus account, visible only to you. No ads or analytics.</p><p>{kani.search ? 'Kani runs offline; web search needs a connection.' : 'Web search is off. Kani generates replies on your device.'} Android checks GitHub for app updates without sending your chats. Image generation is not included.</p><p>In Learn, photos are read and translated on this phone. Looking a word up sends only that word, to Wiktionary or Tatoeba.</p><p>Train English and Speak English check your English with fixed grammar rules on this phone. Speak English downloads two open voice models once from Hugging Face (Whisper by OpenAI to listen, Kokoro by hexgrad to talk); they are not Kani’s own weights, and your voice never leaves the phone.</p></section>
      <section className="settings-section"><h2><Icon name="spark" size={19} />Made from scratch. Still learning.</h2><p>Kani uses Kelus’s own trained model. It’s experimental: answers and generated code can be wrong.</p><p>Kani Otter 4.12 is the newest and smartest; Kani Seal 3 is smaller and quicker. Both run entirely on this phone. The model shown above is the one answering. Kani Otter 4.12 also learned from <a href="https://huggingface.co/datasets/allenai/soda" target="_blank" rel="noreferrer">SODA</a> by Kim et al. (2023), <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noreferrer">CC BY 4.0</a>. The dialogues were filtered and reformatted for training.</p></section>
      <footer className="about"><span>kani<span className="brand-dot">.</span></span><p>Made with curiosity by Kelus<br />Version {updater.version.name || appVersion} · {Capacitor.getPlatform()}</p>
        <small>© {new Date().getFullYear()} Kelus Corp. All rights reserved.</small></footer>
    </main>}

    <nav className="tabbar" aria-label="Main navigation">{([{ id: 'chat', icon: 'chat', label: 'Chat' }, { id: 'learn', icon: 'book', label: 'Learn' }, { id: 'history', icon: 'history', label: 'Saved' }, { id: 'settings', icon: 'settings', label: 'Settings' }] as const).map(t =>
      <button key={t.id} className={tab === t.id ? 'selected' : ''} aria-current={tab === t.id ? 'page' : undefined} onClick={() => setTab(t.id)}><span><Icon name={t.icon} /></span>{t.label}</button>)}</nav>
    {deleteId && <div className="modal-backdrop" onClick={() => setDeleteId(null)}><section className="modal" role="alertdialog" aria-modal="true" aria-labelledby="delete-title" onClick={e => e.stopPropagation()}>
      <h2 id="delete-title">Delete this conversation?</h2><p>This removes its saved copy from this device.</p><div><button className="secondary" autoFocus onClick={() => setDeleteId(null)}>Keep it</button><button className="danger" onClick={() => { kani.remove(deleteId); setDeleteId(null); }}>Delete</button></div>
    </section></div>}
  </div>;
}
