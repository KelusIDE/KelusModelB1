// English practice: Train mode checks what you type and gives tips; Speak mode listens (Whisper), checks your grammar
// and pronunciation, and answers out loud (Kokoro). The grammar rules and sound tips are site/english.js's.
import { useCallback, useEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react';
import './english.css';
import { PRACTICE, readingFeedback, readingSummary } from '../../site/english.js';
import { SOUND_TIPS, ipa, parseDictionary, phonesFor } from '../../site/voice/pronounce.js';
import { PART_SIZE, loadPart, primeAudio, recordSpeech, speak, stopSpeaking, transcribe, type Part, type SpokenWord } from './voice';

export type Mode = 'chat' | 'train' | 'speak';
export type Words = Map<string, string[]>;
export type TutorNote = ReturnType<typeof import('../../site/english.js').tutor>;
const MODE_KEY = 'kani-mode';
export const MODES: { id: Mode; label: string; title: string }[] = [
  { id: 'chat', label: 'Chat', title: 'Talk with Kani' },
  { id: 'train', label: 'Train English', title: 'Kani checks your English and gives tips' },
  { id: 'speak', label: 'Speak English', title: 'Talk out loud: Kani listens, coaches your grammar and pronunciation, and answers out loud' },
];
// Sounds learners find hard, to point out in a word.
const HARD = ['TH', 'DH', 'R', 'L', 'V', 'W', 'Z', 'SH', 'CH', 'JH', 'NG', 'IH', 'IY', 'AE', 'ER', 'HH'];
const mb = (bytes: number) => `${Math.round(bytes / 1e6)} MB`;
export const UNCLEAR = 0.3;

export function useMode(): [Mode, (mode: Mode) => void] {
  const [mode, setMode] = useState<Mode>(() => {
    try { const saved = localStorage.getItem(MODE_KEY); return MODES.some(m => m.id === saved) ? saved as Mode : 'chat'; } catch { return 'chat'; }
  });
  const choose = useCallback((next: Mode) => { setMode(next); try { localStorage.setItem(MODE_KEY, next); } catch { /* private window */ } }, []);
  return [mode, choose];
}

// The pronouncing dictionary, fetched once English practice starts.
let dictionary: Promise<Words> | null = null;
export function useWords(wanted: boolean) {
  const [words, setWords] = useState<Words | null>(null);
  useEffect(() => {
    if (!wanted) return;
    dictionary ||= fetch(new URL('voice/cmudict.txt', document.baseURI)).then(r => r.text()).then(parseDictionary);
    let live = true;
    dictionary.then(w => { if (live) setWords(w); }).catch(() => { dictionary = null; });
    return () => { live = false; };
  }, [wanted]);
  return words;
}

type PartState = 'off' | 'loading' | 'ready' | 'failed';
export function useVoice() {
  const [state, setState] = useState<Record<Part, PartState>>({ listen: 'off', speak: 'off' });
  const [progress, setProgress] = useState<{ part: Part; loaded: number; total: number } | null>(null);
  const [error, setError] = useState('');
  const loads = useRef<Partial<Record<Part, Promise<void>>>>({});
  const ensure = useCallback((part: Part) => {
    loads.current[part] ||= (async () => {
      setState(s => ({ ...s, [part]: 'loading' })); setError('');
      try {
        await loadPart(part, (loaded, total) => setProgress({ part, loaded, total }));
        setState(s => ({ ...s, [part]: 'ready' }));
      } catch (failure) {
        loads.current[part] = undefined;
        setState(s => ({ ...s, [part]: 'failed' })); setError((failure as Error).message);
        throw failure;
      } finally { setProgress(p => p?.part === part ? null : p); }
    })();
    return loads.current[part]!;
  }, []);
  return { state, progress, error, ensure };
}
export type Voice = ReturnType<typeof useVoice>;

export function ModeSwitch({ mode, onChange, disabled }: { mode: Mode; onChange: (mode: Mode) => void; disabled: boolean }) {
  return (
    <div className="mode-switch" role="radiogroup" aria-label="Mode">
      {MODES.map(m => (
        <button key={m.id} type="button" role="radio" aria-checked={mode === m.id} title={m.title} disabled={disabled && mode !== m.id}
          className={mode === m.id ? 'active' : undefined} onClick={() => onChange(m.id)}>{m.label}</button>
      ))}
    </div>
  );
}

// "Listen" for any text, loading the speaking model on first use.
export function ListenButton({ text, voice, label = 'Listen', slow = false }: { text: string; voice: Voice; label?: string; slow?: boolean }) {
  const [busy, setBusy] = useState(false);
  return (
    <button type="button" className="listen-button" disabled={busy} onClick={async () => {
      primeAudio(); stopSpeaking(); setBusy(true);
      try { await voice.ensure('speak'); await speak(text, { speed: slow ? 0.8 : 1 }); } catch { /* shown by the voice status */ } finally { setBusy(false); }
    }}>{busy ? (voice.state.speak === 'loading' ? 'Loading voice…' : 'Playing…') : `▶ ${label}`}</button>
  );
}

export function TutorCard({ note, voice }: { note: TutorNote; voice: Voice }) {
  if (!note.fixes.length) {
    return (
      <div className="tutor-card good">
        <div className="tutor-title">✓ Looks good!</div>
        {note.upgrade && <p>Try a stronger word: instead of <q>{note.upgrade.phrase}</q>, say <b>{note.upgrade.better}</b>.</p>}
        {note.tip && <p className="tutor-tip">Tip: {note.tip}</p>}
      </div>
    );
  }
  return (
    <div className="tutor-card">
      <div className="tutor-title">English coach · {note.fixes.length === 1 ? '1 fix' : `${note.fixes.length} fixes`}</div>
      <ul>
        {note.fixes.map((fix, i) => <li key={i}><s>{fix.wrong}</s> → <b>{fix.right}</b><span>{fix.why}</span></li>)}
      </ul>
      <p className="tutor-better">Better: <q>{note.corrected}</q> <ListenButton text={note.corrected} voice={voice} /></p>
    </div>
  );
}

function WordHelp({ word, words, voice, onClose }: { word: string; words: Words | null; voice: Voice; onClose: () => void }) {
  const clean = word.toLowerCase().replace(/[^a-z']/g, '');
  const phones: string[] = words ? phonesFor(clean, words) : [];
  const hard = [...new Set(phones.map(p => p.replace(/\d$/, '')).filter(p => HARD.includes(p)))];
  return (
    <div className="word-help" role="dialog" aria-label={`How to say ${clean}`}>
      <div className="word-help-head"><b>{clean}</b> <span className="ipa">{phones.length ? ipa(phones) : ''}</span>
        <button type="button" className="link" aria-label="Close" onClick={onClose}>×</button></div>
      {phones.some(p => p.endsWith('1')) && phones.filter(p => /\d$/.test(p)).length > 1 && <p>Stress the part after <b>ˈ</b>.</p>}
      {hard.map(p => <p key={p}><b>{SOUND_TIPS[p as keyof typeof SOUND_TIPS].sound}:</b> {SOUND_TIPS[p as keyof typeof SOUND_TIPS].tip}</p>)}
      <div className="word-help-actions"><ListenButton text={clean} voice={voice} /><ListenButton text={clean} voice={voice} label="Slowly" slow /></div>
    </div>
  );
}

// A sentence you said, with the words Whisper was unsure of marked; tap a word to learn how to say it.
export function SpokenText({ heard, words, voice }: { heard: SpokenWord[]; words: Words | null; voice: Voice }) {
  const [open, setOpen] = useState<number | null>(null);
  return (
    <div className="spoken">
      <pre>{heard.map((w, i) => (
        <span key={i}>{i ? ' ' : ''}<button type="button" className={`spoken-word${w.probability < UNCLEAR ? ' unclear' : ''}`}
          title={w.probability < UNCLEAR ? 'Not clear: tap to practice it' : 'Tap to hear it'} onClick={() => setOpen(open === i ? null : i)}>{w.text}</button></span>
      ))}</pre>
      {heard.some(w => w.probability < UNCLEAR) && <p className="spoken-hint">Underlined words were hard to hear. Tap one to practice it.</p>}
      {open !== null && heard[open] && <WordHelp word={heard[open].text} words={words} voice={voice} onClose={() => setOpen(null)} />}
    </div>
  );
}

type Feedback = ReturnType<typeof readingFeedback>;
export function ReadAloud({ words, voice, disabled }: { words: Words | null; voice: Voice; disabled: boolean }) {
  const [index, setIndex] = useState(() => Math.floor(Math.random() * PRACTICE.length));
  const [stage, setStage] = useState<'idle' | 'recording' | 'checking'>('idle');
  const [level, setLevel] = useState(0);
  const [result, setResult] = useState<Feedback | null>(null);
  const [problem, setProblem] = useState('');
  const [open, setOpen] = useState<number | null>(null);
  const stopper = useRef<AbortController | null>(null);
  const { sentence, focus } = PRACTICE[index];
  const next = () => { setIndex((index + 1 + Math.floor(Math.random() * (PRACTICE.length - 1))) % PRACTICE.length); setResult(null); setOpen(null); setProblem(''); };

  const read = async () => {
    if (stage === 'recording') { stopper.current?.abort(); return; }
    primeAudio(); stopSpeaking(); setProblem(''); setResult(null); setOpen(null);
    try {
      await voice.ensure('listen');
      void voice.ensure('speak').catch(() => {});
      stopper.current = new AbortController(); setStage('recording');
      const audio = await recordSpeech({ onLevel: setLevel, signal: stopper.current.signal });
      if (!audio) { setProblem("I didn't hear anything. Tap Read it and say the sentence."); return; }
      setStage('checking');
      const { heard, scored } = await transcribe(audio, sentence);
      if (!words) throw new Error('The dictionary is still loading. Try again in a moment.');
      const feedback = readingFeedback(sentence, heard, scored, words);
      setResult(feedback);
      if (voice.state.speak !== 'failed') void voice.ensure('speak').then(() => speak(readingSummary(feedback))).catch(() => {});
    } catch (failure) {
      setProblem((failure as Error).name === 'NotAllowedError' ? 'Kani needs your microphone. Allow it in the browser and try again.' : (failure as Error).message);
    } finally { setStage('idle'); stopper.current = null; }
  };

  return (
    <section className="read-aloud" aria-label="Read aloud practice">
      <div className="read-aloud-head"><span className="eyebrow">Read aloud · {focus === 'everyday' ? 'everyday English' : focus === 'stress' ? 'word stress' : `the ${SOUND_TIPS[focus as keyof typeof SOUND_TIPS]?.sound || focus} sound`}</span>
        <button type="button" className="link" onClick={next} disabled={stage !== 'idle'}>Next sentence →</button></div>
      <p className="read-aloud-sentence">{result
        ? result.words.map((w: Feedback['words'][number], i: number) => <span key={i}>{i ? ' ' : ''}<button type="button" className={`read-word ${w.status}`} onClick={() => setOpen(open === i ? null : i)}
          title={w.status === 'good' ? 'Clear' : w.status === 'wrong' ? `Sounded like "${w.heard}"` : w.status === 'missed' ? 'Not heard' : 'A little unclear'}>{w.text}</button></span>)
        : sentence}</p>
      {result && <p className="read-aloud-score"><b>{result.score}%</b> clear · {readingSummary(result)}</p>}
      {result && open !== null && result.words[open] && <WordHelp word={result.words[open].text} words={words} voice={voice} onClose={() => setOpen(null)} />}
      {problem && <p className="read-aloud-problem" role="alert">{problem}</p>}
      <div className="read-aloud-actions">
        <ListenButton text={sentence} voice={voice} label="Hear it" />
        <button type="button" className={`mic-button small${stage === 'recording' ? ' live' : ''}`} disabled={disabled || stage === 'checking'} onClick={read}
          style={{ '--level': level } as CSSProperties}>{stage === 'recording' ? '■ Done' : stage === 'checking' ? 'Checking…' : '● Read it'}</button>
      </div>
    </section>
  );
}

export function VoiceStatus({ voice }: { voice: Voice }) {
  const { progress, error } = voice;
  if (progress) {
    return <p className="voice-status">Downloading the {progress.part === 'listen' ? 'listening' : 'speaking'} model once ({mb(progress.loaded)} of {mb(progress.total || PART_SIZE[progress.part])})… It stays on this device.</p>;
  }
  return error ? <p className="voice-status problem" role="alert">Voice: {error}</p> : null;
}

const STARTERS = [
  { label: 'Talk about your day', prompt: 'Yesterday I go to the park with my friends and we play football.' },
  { label: 'Introduce yourself', prompt: 'Hello! My name is Ana and I am student from Brazil.' },
  { label: 'Describe something', prompt: 'My city is very big and the food is very good.' },
];

export function EnglishWelcome({ mode, composer, onPick, disabled, words, voice }: { mode: Mode; composer: ReactNode; onPick: (text: string) => void; disabled: boolean; words: Words | null; voice: Voice }) {
  return (
    <section id="english-welcome" aria-labelledby="english-title">
      <h1 id="english-title">{mode === 'train' ? 'Practice writing English' : 'Practice speaking English'}</h1>
      <p className="welcome-copy">{mode === 'train'
        ? 'Each message gets a quick check: what to fix, why, and a stronger word to try. Then Kani answers, so the conversation keeps going.'
        : 'Tap Speak and talk. Kani writes down what you said, points out grammar to fix and words that were hard to hear, then answers out loud.'}</p>
      {composer}
      {mode === 'train'
        ? <div className="starter-row"><span>Try a sentence with mistakes</span><div className="suggestions">
          {STARTERS.map(item => <button key={item.label} disabled={disabled} onClick={() => onPick(item.prompt)}>{item.label}</button>)}
        </div></div>
        : <ReadAloud words={words} voice={voice} disabled={disabled} />}
      <p className="english-note">Speech becomes text on this device with Whisper (OpenAI) and is read aloud with Kokoro (hexgrad): open voice models, downloaded once from Hugging Face (about 77 MB to listen, 93 MB to speak). They are not Kani's own weights. Your voice is never uploaded.</p>
    </section>
  );
}
