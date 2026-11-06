import { useEffect, useRef, useState, type CSSProperties, type FormEvent, type KeyboardEvent } from 'react';
import AnimatedContent from '@/components/AnimatedContent';
import ClickSpark from '@/components/ClickSpark';
import CountUp from '@/components/CountUp';
import ShinyText from '@/components/ShinyText';
import { AccountDialog, ChatList } from './Account';
import { Icon, KaniMark } from './Brand';
import { ACCEPT, spokenAnswer } from './engine';
import { EnglishWelcome, ModeSwitch, SpokenText, TutorCard, UNCLEAR, VoiceStatus, useMode, useVoice, useWords, type Voice, type Words } from './English';
import { canRecord, primeAudio, recordSpeech, speak, stopSpeaking, transcribe } from './voice';
import { spokenFeedback, tutor } from '../../site/english.js';
import { useAttachments } from './useAttachments';
import { MODELS, hasSvg, labelFor, useKani, type Item, type ModelId } from './useKani';
import { SvgPreview } from './SvgPreview';
import { ThemeMenu, useTheme } from './theme';
import { Welcome } from './Welcome';
import { CodeBlock } from './CodeBlock';
import { PlacesReply } from './PlacesReply';
import { SendButton } from './SiteMotion';

export default function App() {
  const [error, setLocalError] = useState('');
  const files = useAttachments(setLocalError);
  const kani = useKani(files.clear);
  const [prompt, setPrompt] = useState('');
  const [temperature, setTemperature] = useState('0');
  const [dialogOpen, setDialogOpen] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [infoOpen, setInfoOpen] = useState(false);
  const [dragging, setDragging] = useState(false);
  const { resolved, palette } = useTheme();
  const light = resolved === 'light';
  const promptRef = useRef<HTMLTextAreaElement>(null);
  const composerRef = useRef<HTMLDivElement>(null);
  const pickerRef = useRef<HTMLInputElement>(null);
  const historyDialog = useRef<HTMLDialogElement>(null);
  const infoDialog = useRef<HTMLDialogElement>(null);
  const hasConversation = kani.items.length > 0;
  const [mode, setMode] = useMode();
  const english = mode !== 'chat';
  const words = useWords(english);
  const voice = useVoice();
  const turn = useRef(0);
  const [talking, setTalking] = useState<'idle' | 'recording' | 'checking'>('idle');
  const [level, setLevel] = useState(0);
  const stopper = useRef<AbortController | null>(null);
  const shownError = kani.error || error;
  const clearError = () => { kani.setError(''); setLocalError(''); };

  // Speak mode reads Kani's reply aloud: the answer without its sources, a few sentences at most.
  const sayReply = (reply: string, source: string) => {
    const text = source === 'model_web' ? spokenAnswer(reply) : source === 'places' ? reply.split('\n')[0] : reply;
    void speak(text.length > 450 ? (text.slice(0, 450).match(/^[\s\S]*[.!?](?=\s)/)?.[0] || text.slice(0, 450)) : text);
  };
  const coach = (feedback: string) => { if (feedback) void voice.ensure('speak').then(() => speak(feedback)).catch(() => {}); };

  const send = async (text: string) => {
    if (files.reading) return;
    clearError();
    // Train and Speak modes: the English coach checks the message first (not file chats).
    const note = english && words && text.trim() && !files.files.length ? tutor(text.trim(), words, turn.current++) : undefined;
    if (mode === 'speak') { primeAudio(); stopSpeaking(); if (note) coach(spokenFeedback(note)); }
    if (await kani.send(text, files.files, Number(temperature), { note, onReply: mode === 'speak' ? sayReply : undefined })) setPrompt('');
  };

  // Speak mode: record until the speaker stops, write it down, coach it out loud, then send it to Kani.
  const talk = async () => {
    if (talking === 'recording') { stopper.current?.abort(); return; }
    if (!words || kani.busy || !kani.model) return;
    primeAudio(); stopSpeaking(); clearError();
    try {
      await voice.ensure('listen');
      void voice.ensure('speak').catch(() => {});
      stopper.current = new AbortController(); setTalking('recording');
      const audio = await recordSpeech({ onLevel: setLevel, signal: stopper.current.signal });
      if (!audio) { setLocalError("I didn't hear anything. Tap Speak and talk."); return; }
      setTalking('checking');
      const { heard, scored } = await transcribe(audio);
      const text = heard.text.trim();
      if (!text) { setLocalError("I couldn't make out any words. Try again a little closer to the microphone."); return; }
      const note = tutor(text, words, turn.current++);
      // A word Whisper was unsure of (not a name), when there is no grammar to fix.
      const unclear = scored.find((w, i) => w.probability < UNCLEAR && /[a-z]/i.test(w.text) && !(i && /^[A-Z]/.test(w.text)))?.text.replace(/[^\w']/g, '');
      coach(spokenFeedback(note) || (unclear ? `I wasn't sure I heard "${unclear}". Say it like this: ${unclear}.` : ''));
      await kani.send(text, [], Number(temperature), { note, heard: scored, onReply: sayReply });
    } catch (failure) {
      setLocalError((failure as Error).name === 'NotAllowedError' ? 'Kani needs your microphone to listen. Allow it in the browser and try again.' : (failure as Error).message);
    } finally { setTalking('idle'); stopper.current = null; }
  };
  const newChat = () => { kani.startChat(); setPrompt(''); clearError(); setHistoryOpen(false); promptRef.current?.focus(); };

  useEffect(() => {
    const composer = composerRef.current;
    if (!composer) return;
    const observer = new ResizeObserver(() => document.documentElement.style.setProperty('--composer-height', `${Math.ceil(composer.getBoundingClientRect().height) + 32}px`));
    observer.observe(composer);
    return () => observer.disconnect();
  }, [hasConversation]);

  useEffect(() => {
    const element = historyDialog.current;
    if (historyOpen && !element?.open) element?.showModal();
    if (!historyOpen && element?.open) element.close();
  }, [historyOpen]);
  useEffect(() => {
    const element = infoDialog.current;
    if (infoOpen && !element?.open) element?.showModal();
    if (!infoOpen && element?.open) element.close();
  }, [infoOpen]);
  const lastText = kani.items.at(-1)?.content;
  useEffect(() => { if (hasConversation) window.scrollTo({ top: document.body.scrollHeight, behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth' }); }, [kani.items.length, lastText, hasConversation]);
  useEffect(() => { if (!kani.busy) promptRef.current?.focus(); }, [kani.busy]);

  const onSubmit = (event: FormEvent) => { event.preventDefault(); void send(prompt); };
  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); void send(prompt); }
  };
  const locked = kani.busy || files.reading;
  const stats = kani.model?.manifest;
  const chosen = MODELS.find(m => m.id === kani.modelId)!;
  const composer = <div className="composer-wrap" ref={composerRef}>
    <div className="composer-modes">
      <ModeSwitch mode={mode} disabled={kani.busy || talking !== 'idle'} onChange={next => { stopSpeaking(); setMode(next); }} />
      {english && <VoiceStatus voice={voice} />}
    </div>
    <form id="chat-form" className={dragging ? 'file-drag' : undefined} onSubmit={onSubmit}
      onDragOver={event => { if ([...event.dataTransfer.types].includes('Files')) { event.preventDefault(); setDragging(true); } }}
      onDragLeave={() => setDragging(false)}
      onDrop={event => { if (event.dataTransfer.files.length) { event.preventDefault(); setDragging(false); void files.add([...event.dataTransfer.files]); } }}>
      {files.files.length > 0 && <div className="attachment-list" aria-live="polite">
        {files.files.map((file, index) => <Attached key={`${file.name}-${index}`} file={file} disabled={locked} onRemove={() => files.remove(index)} />)}
      </div>}
      <label htmlFor="prompt" className="sr-only">Message {chosen.name}</label>
      <textarea id="prompt" ref={promptRef} rows={2} maxLength={8000} value={prompt}
        placeholder={mode === 'train' ? 'Write to Kani in English. It checks your message and gives tips…' : mode === 'speak' ? 'Tap Speak and talk, or type here…' : 'Message Kani…'}
        onChange={event => setPrompt(event.target.value)} onKeyDown={onKeyDown}
        onPaste={event => {
          const pasted = [...event.clipboardData.items].filter(i => i.kind === 'file').map(i => i.getAsFile()).filter((f): f is File => Boolean(f));
          if (pasted.length) { event.preventDefault(); void files.add(pasted); }
        }} />
      <div className="composer-bottom">
        <div className="composer-tools">
          <button type="button" className="attach-button" title="Attach PDFs, images, HTML or text files" aria-label="Attach files" disabled={locked} onClick={() => pickerRef.current?.click()}><Icon name="attach" /><span>{files.reading ? 'Reading…' : 'Attach'}</span></button>
          <span className="tool-divider" />
          <label className="model-picker" title={chosen.about}><span className="sr-only">Model</span>
            <span className="model-indicator" /><select id="model" value={kani.modelId} disabled={kani.busy} onChange={event => kani.chooseModel(event.target.value as ModelId)}>
              {MODELS.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
            </select>
          </label>
        </div>
        <div className="composer-actions">
          <details className="generation-options"><summary aria-label="Generation settings" title="Generation settings"><Icon name="tune" /></summary><div><label>Creativity <input id="temperature" type="number" min="0" max="2" step="0.05" value={temperature} onChange={event => setTemperature(event.target.value)} /></label><small>0 gives repeatable replies.</small></div></details>
          {kani.busy && <button id="stop" type="button" onClick={kani.stop}>Stop</button>}
          {mode === 'speak' && <button id="mic" type="button" className={`mic-button${talking === 'recording' ? ' live' : ''}`} style={{ '--level': level } as CSSProperties}
            disabled={!canRecord() || !words || !kani.model || (kani.busy && talking === 'idle') || talking === 'checking'} onClick={() => void talk()}
            aria-label={talking === 'recording' ? 'Stop recording' : 'Speak'}>{talking === 'recording' ? '■ Done' : talking === 'checking' ? 'Listening…' : voice.state.listen === 'loading' ? 'Loading…' : '● Speak'}</button>}
          {!kani.busy && <ClickSpark sparkColor={palette.accent} sparkCount={8} sparkRadius={24} sparkSize={8} duration={400} style={{ width: 'auto', height: 'auto', display: 'flex' }}>
            <SendButton disabled={!kani.model} />
          </ClickSpark>}
        </div>
      </div>
      <input ref={pickerRef} type="file" multiple accept={ACCEPT} hidden onChange={event => { void files.add([...(event.target.files || [])]); event.target.value = ''; }} />
    </form>
    <div className="composer-caption"><span><span className="live-dot" />Files &amp; model stay on your device</span><span className="keyboard-hint">Enter ↵ to send</span></div>
    {hasConversation && <p id="generation-info" role="status">{kani.status === 'loading' ? `Loading ${chosen.name}…` : kani.info}</p>}
  </div>;

  return (
    <>
      <a className="skip-link" href="#prompt">Skip to conversation</a>
      <aside className="workspace-rail" aria-label="Workspace navigation">
        <a className="rail-brand" href="/" aria-label="Kani home"><KaniMark /></a>
        <nav>
          <button id="new-chat" className="rail-button rail-new" aria-label="New conversation" title="New conversation" disabled={kani.busy} onClick={newChat}><Icon name="plus" /></button>
          <button className="rail-button" aria-label="Saved conversations" title="Saved conversations" disabled={kani.busy} onClick={() => setHistoryOpen(true)}><Icon name="history" /></button>
          <button className="rail-button" aria-label="About Kani" title="About Kani" onClick={() => setInfoOpen(true)}><Icon name="info" /></button>
        </nav>
        <div className="rail-bottom"><a className="rail-button" href="/downloads/" aria-label="Get the iOS app" title="Get the iOS app"><Icon name="download" /></a></div>
      </aside>
      <main className={hasConversation ? 'studio is-chat' : 'studio is-home'}>
        <header className="studio-header">
          <a className="wordmark" href="/">kani<span>.</span><small>by kelus</small></a>
          <div className="header-right"><button className="mobile-nav" aria-label="Open conversations" disabled={kani.busy} onClick={() => setHistoryOpen(true)}><Icon name="history" /></button><ThemeMenu /><button id="account-button" className="header-button account-button" disabled={kani.accountState !== 'ready'} onClick={() => setDialogOpen(true)}>{kani.user ? (kani.user.displayName || kani.user.email || 'Account').split(' ')[0] : kani.accountState === 'unavailable' ? 'Account offline' : 'Sign in'}<Icon name="diagonal" /></button></div>
        </header>
        {kani.updateAvailable && <div id="update-banner" className="update-banner" role="status">A new version of Kani is ready. <button id="reload" className="link" onClick={() => location.reload()}>Reload to use it</button></div>}
        <div className="studio-content">
          {hasConversation ? <>
            <div className="conversation-heading"><button className="text-button" disabled={kani.busy} onClick={newChat}>New chat <Icon name="plus" /></button></div>
            <section id="messages" aria-live="polite" aria-label="Conversation">{kani.items.map(item => <Message key={item.id} item={item} model={chosen.name} words={words} voice={voice} />)}</section>
            {composer}
          </> : english ? <EnglishWelcome mode={mode} composer={composer} disabled={!kani.model || kani.busy} onPick={text => void send(text)} words={words} voice={voice} />
            : <Welcome composer={composer} disabled={!kani.model || kani.busy} onPick={text => void send(text)} />}
          {shownError && <div id="error" role="alert"><Icon name="info" /><span>{shownError}</span><button aria-label="Dismiss error" onClick={clearError}><Icon name="close" /></button></div>}
        </div>
        {!hasConversation && <footer className="studio-footer"><span id="connection" className={'connection-status ' + kani.status} title={kani.engine ? `WebGPU · ${kani.engine}` : 'WebAssembly on the CPU'}><span className="live-dot" />{kani.status === 'ready' ? (kani.engine ? 'Running on your GPU' : 'Running in your browser') : kani.status === 'failed' ? 'Model unavailable' : <ShinyText text={`Loading ${chosen.name} · ${chosen.about.split(' · ')[1]}`} color={light ? '#6d726b' : '#a0a9bb'} shineColor={palette.accent} speed={2} />}</span><span className="footer-note">Kani can make mistakes. Check important answers.</span></footer>}
      </main>
      <dialog ref={historyDialog} className="history-dialog" aria-labelledby="history-title" onCancel={() => setHistoryOpen(false)} onClose={() => setHistoryOpen(false)}>
        <div className="dialog-head"><div><h2 id="history-title">Your conversations</h2></div><button className="icon-button" aria-label="Close conversations" onClick={() => setHistoryOpen(false)}><Icon name="close" /></button></div>
        {kani.user ? <ChatList chats={kani.savedChats} current={kani.chatId} onOpen={chat => { kani.startChat(chat.messages || [], chat.id, true); setHistoryOpen(false); }} onDelete={kani.deleteChat} /> : <div className="history-empty"><Icon name="chat" /><h3>Save your conversations</h3><p>Sign in to save your conversations and pick them up on another device.</p><button className="solid-button" disabled={kani.accountState !== 'ready'} onClick={() => { setHistoryOpen(false); setDialogOpen(true); }}>Sign in to Kani <Icon name="diagonal" /></button></div>}
        <button className="text-button" onClick={newChat}>Start something new <Icon name="plus" /></button>
        <button className="text-button" onClick={() => { setHistoryOpen(false); setInfoOpen(true); }}>About Kani <Icon name="info" /></button>
      </dialog>
      <dialog ref={infoDialog} className="about-dialog" aria-labelledby="about-title" onCancel={() => setInfoOpen(false)} onClose={() => setInfoOpen(false)}>
        <div className="dialog-head"><h2 id="about-title">About Kani</h2><button className="icon-button" aria-label="Close about Kani" onClick={() => setInfoOpen(false)}><Icon name="close" /></button></div>
        <p className="dialog-copy">Kani is an independent, experimental model trained from scratch by Kelus. Its model runs on your device. It can make mistakes; check important answers.</p>
        <div className="stats"><span>Parameters<strong>{kani.model ? <CountUp to={kani.model.parameterCount} separator="," duration={1} trigger="load" /> : 'Loading…'}</strong></span><span>Training steps<strong>{stats?.training.steps?.toLocaleString() || '—'}</strong></span><span>Context<strong>{stats ? `${stats.config.context} tokens` : '—'}</strong></span></div>
        <ul className="about-facts"><li>Choose a Kani model for chat or a Cide model for code. Releases have different download sizes and capabilities.</li><li>Web and place searches send the search query to Kani’s search service. Results are attributed. Ordinary model replies run locally.</li><li>Arithmetic and date/time use labelled local helpers. HTML replies show a live preview in a locked frame with no network and no access to this site or your account; other generated code is displayed, never run.</li><li>Signed-in conversations are saved to your account. Chats with attached PDFs, text, code, HTML, or images stay local. Image support inspects pixels, rather than understanding a screenshot.</li><li>Messages about suicide or self-harm receive a fixed support message.</li><li>Train English and Speak English check your English with fixed grammar rules on this device. Speak English downloads two open voice models once from Hugging Face (Whisper by OpenAI to listen, Kokoro by hexgrad to talk); they are not Kani's weights, and your voice never leaves this device.</li><li>Kani Otter 4.12 also learned from <a href="https://huggingface.co/datasets/allenai/soda" target="_blank" rel="noreferrer">SODA</a> by Kim et al. (2023), <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noreferrer">CC BY 4.0</a>. The dialogues were filtered and reformatted for training.</li><li>Cide 4.5 also learned from <a href="https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k" target="_blank" rel="noreferrer">StarCoder2 Self-Alignment</a> by BigCode and <a href="https://huggingface.co/datasets/OpenCoder-LLM/opc-annealing-corpus" target="_blank" rel="noreferrer">OpenCoder</a> (<a href="https://opendatacommons.org/licenses/by/1-0/" target="_blank" rel="noreferrer">ODC-BY</a>), and <a href="https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K" target="_blank" rel="noreferrer">Magicoder OSS-Instruct</a> (MIT) and <a href="https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K" target="_blank" rel="noreferrer">Evol-Instruct</a> (Apache-2.0). The rows were filtered and reformatted for training.</li></ul>
        <a className="solid-button" href="/downloads/">Take Kani with you <Icon name="download" /></a>
      </dialog>
      <AccountDialog open={dialogOpen} onClose={() => setDialogOpen(false)} account={kani.account} user={kani.user} chats={kani.savedChats} current={kani.chatId} onOpenChat={chat => kani.startChat(chat.messages || [], chat.id, true)} onDeleteChat={kani.deleteChat} onNewChat={newChat} />
    </>
  );
}

function Message({ item, model, words, voice }: { item: Item; model: string; words: Words | null; voice: Voice }) {
  const helper = item.role === 'assistant' && item.source !== 'model';
  // Before the first words: thinking, searching, reading or looking for places.
  const waiting = item.role === 'assistant' && !item.done && (!item.content || item.content === 'Thinking…'
    || (['web', 'web_search', 'places'].includes(item.source) && !item.places));
  const { resolved, palette } = useTheme();
  return (
    <AnimatedContent trigger="mount" distance={18} duration={0.45} ease="power2.out" initialOpacity={0}
      className={`message ${item.role}${helper ? ' helper' : ''}`}>
      <div className="message-label">{item.role === 'user' ? 'YOU' : labelFor(item.source, item.by || model)}</div>
      {item.role === 'assistant' && item.done && item.source === 'model' && hasSvg(item.content) && <SvgPreview reply={item.content} />}
      {waiting ? <div className="reply"><ShinyText text={item.content || 'Thinking…'} speed={1.8} color={resolved === 'light' ? '#5f6368' : '#9a9fa7'} shineColor={palette.shine} /></div>
        : item.places ? <PlacesReply result={item.places} /> : item.role === 'assistant' ? <Reply text={item.content} done={Boolean(item.done)} />
          : item.heard ? <SpokenText heard={item.heard} words={words} voice={voice} /> : <pre>{item.content}</pre>}
      {item.note && <TutorCard note={item.note} voice={voice} />}
      {item.receipt && <div className="attachment-receipt">{item.receipt}</div>}
    </AnimatedContent>
  );
}


function Reply({ text, done }: { text: string; done: boolean }) {
  const parts: { code: boolean; text: string; lang?: string; closed?: boolean }[] = [];
  let last = 0;
  for (const match of text.matchAll(/```([\w+#-]*)[^\n]*\n?([\s\S]*?)(?:```|$)/g)) {
    if (match.index > last) parts.push({ code: false, text: text.slice(last, match.index).replace(/\n+$/, '') });
    parts.push({ code: true, lang: match[1], text: match[2].replace(/\n$/, ''), closed: done || match[0].endsWith('```') });
    last = match.index + match[0].length;
  }
  if (last < text.length) parts.push({ code: false, text: text.slice(last).replace(/^\n+/, '') });
  return (
    <div className="reply">
      {parts.map((part, i) => part.code
        ? <CodeBlock key={i} code={part.text} lang={part.lang} live={!part.closed} />
        : part.text && <div key={i} className="prose">{part.text}</div>)}
    </div>
  );
}

function Attached({ file, disabled, onRemove }: { file: { name: string; size: number; report: string; blob?: Blob }; disabled: boolean; onRemove: () => void }) {
  const [url, setUrl] = useState<string>();
  useEffect(() => {
    if (!file.blob) return;
    const preview = URL.createObjectURL(file.blob);
    setUrl(preview);
    return () => URL.revokeObjectURL(preview);
  }, [file.blob]);
  return (
    <div className="attachment-item">
      {url && <img src={url} alt={`Preview of ${file.name}`} />}
      <details><summary>{file.name} · {Math.ceil(file.size / 1024)} KB</summary><pre>{file.report}</pre></details>
      <button type="button" aria-label={`Remove ${file.name}`} disabled={disabled} onClick={onRemove}>×</button>
    </div>
  );
}
