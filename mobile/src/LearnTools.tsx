import { useEffect, useRef, useState } from 'react';
import { Icon } from './Icon';
import { language, type LanguageId } from './learn/content';
import type { Card } from './learn/srs';
import type { Mine } from './learn/cards';
import { canScan, photo, readText, things, type Photo } from './learn/vision';
import { canTranslate, translate } from './learn/translate';
import { dictionary, sentencesWith, words, type Entry, type Example } from './learn/lookup';
import { joined, lookUp, pinyinOf, readCedict, segment, type Cedict, type Hanzi } from './learn/chinese';
import cedictUrl from './learn/cedict.txt?url';




export type Add = (pairs: [text: string, en: string, romanization?: string][]) => number;
type Say = (text: string) => void;
const CJK_SCRIPT: Partial<Record<LanguageId, RegExp>> = { ko: /[가-힣]/, ja: /[぀-ヿ一-鿿]/, 'zh-Hans': /[一-鿿]/, 'zh-Hant': /[一-鿿]/ };

const inLanguage = (text: string, id: LanguageId) => CJK_SCRIPT[id]?.test(text) ?? true;
const message = (error: unknown) => error instanceof Error ? error.message : 'Something went wrong.';
const HAN = /\p{Script=Han}/u;


let cedict: Promise<Cedict> | null = null;
function useChinese(id: LanguageId): Cedict | null {
  const [map, setMap] = useState<Cedict | null>(null);
  useEffect(() => {
    if (!id.startsWith('zh')) return;
    cedict ??= fetch(cedictUrl).then(r => r.text()).then(readCedict);
    void cedict.then(setMap).catch(() => { cedict = null; });
  }, [id]);
  return map;
}

const romanize = (map: Cedict | null, id: LanguageId, text: string) => map && id.startsWith('zh') && HAN.test(text) ? pinyinOf(map, text) : '';

export function Back({ onClick }: { onClick: () => void }) {
  return <button className="learn-back" onClick={onClick}><Icon name="chevron" size={16} />Back</button>;
}

export function Play({ text, say, voice }: { text: string; say: Say; voice: boolean | null }) {
  return <button className="play" onClick={() => say(text)} disabled={voice === false} aria-label="Listen">▶</button>;
}


export function AddCard({ text, en, romanization, add, label = 'Add card' }: { text: string; en: string; romanization?: string; add: Add; label?: string }) {
  const [done, setDone] = useState<'added' | 'had' | null>(null);
  return <button className="mini" disabled={done !== null || !text || !en} onClick={() => setDone(add([[text, en, romanization]]) ? 'added' : 'had')}>
    {done === 'added' ? <><Icon name="check" size={14} />Added</> : done === 'had' ? 'Already a card' : <><Icon name="plus" size={14} />{label}</>}</button>;
}

export function Scan({ id, add, understand, back, hidden }: { id: LanguageId; add: Add; understand: (text: string) => void; back: () => void; hidden?: boolean }) {
  const lang = language(id), name = lang.name.replace(/ \(.*\)$/, '');
  const [shot, setShot] = useState<Photo | null>(null);
  const [status, setStatus] = useState('');
  const [lines, setLines] = useState<{ text: string; en?: string }[] | null>(null);
  const [objects, setObjects] = useState<{ en: string; text?: string }[]>([]);
  const [made, setMade] = useState<number | null>(null);
  const run = useRef(0);
  const chinese = useChinese(id);
  const take = async (from: 'camera' | 'gallery') => {
    let picked: Photo | null;
    try { picked = await photo(from); } catch (error) { setStatus(message(error)); return; }
    if (!picked) return;
    const mine = ++run.current, current = () => run.current === mine;
    setShot(picked); setLines(null); setObjects([]); setMade(null); setStatus('Reading the photo…');
    try {
      const [read, seen] = await Promise.all([readText(picked.uri, id), things(picked.uri).catch(() => [] as string[])]);
      if (!current()) return;
      const found = read.slice(0, 40).map(text => ({ text }));
      setLines(found); setObjects(seen.map(en => ({ en })));
      if (!found.length && !seen.length) { setStatus('Nothing to read in this photo. Try a closer, sharper one.'); return; }
      setStatus('Translating…');
      for (const [i, line] of found.entries()) {
        if (!inLanguage(line.text, id)) continue;
        const en = await translate(line.text, id, 'en');
        if (!current()) return;
        setLines(ls => ls && ls.map((l, j) => j === i ? { ...l, en } : l));
      }
      for (const [i, thing] of seen.entries()) {
        const text = await translate(thing.toLowerCase(), 'en', id);
        if (!current()) return;
        setObjects(os => os.map((o, j) => j === i ? { ...o, text } : o));
      }
      setStatus('');
    } catch (error) { if (current()) setStatus(message(error)); }
  };
  const ready = (lines ?? []).filter(l => l.en && l.text.length <= 80);
  return <main className="page learn" hidden={hidden}>
    <Back onClick={back} />
    <p className="eyebrow">{lang.name.toUpperCase()} · SCAN</p><h1>Read a photo<span className="brand-dot">.</span></h1>
    <p className="page-subtitle">A menu, a sign, a page, or things around you. See what it says, then keep the words as cards.</p>
    {!canScan() ? <p className="learn-note">Scanning photos works in the Kani app on Android.</p> : <div className="scan-buttons">
      <button className="learn-primary" onClick={() => void take('camera')}><Icon name="camera" size={18} />Take a photo</button>
      <button className="learn-secondary" onClick={() => void take('gallery')}><Icon name="image" size={18} />Choose a photo</button>
    </div>}
    {shot && <img className="scan-photo" src={shot.shown} alt="Your photo" />}
    {status && <p className={/^(Reading|Translating)/.test(status) ? 'muted' : 'learn-warning'} role="status">{status}</p>}
    {lines && lines.length > 0 && <>
      <h2 className="learn-heading">Text in the photo</h2>
      <div className="phrase-list">{lines.map((line, i) => <div key={i} className="phrase scan-line">
        <div><strong lang={inLanguage(line.text, id) ? lang.speech : undefined}>{line.text}</strong>
          {romanize(chinese, id, line.text) && <em>{romanize(chinese, id, line.text)}</em>}
          {inLanguage(line.text, id) && <small>{line.en ?? '…'}</small>}
          {inLanguage(line.text, id) && <div className="mini-row">
            <button className="mini" onClick={() => understand(line.text)}><Icon name="translate" size={14} />Words</button>
            <AddCard text={line.text} en={line.en ?? ''} romanization={romanize(chinese, id, line.text)} add={add} /></div>}</div>
      </div>)}</div>
      {ready.length > 1 && <button className="learn-secondary" disabled={made !== null} onClick={() => setMade(add(ready.map(l => [l.text, l.en!, romanize(chinese, id, l.text)])))}>
        <Icon name="cards" size={18} />{made === null ? `Make ${ready.length} cards from this photo` : `${made} new card${made === 1 ? '' : 's'} added`}</button>}
    </>}
    {objects.length > 0 && <>
      <h2 className="learn-heading">Things in the photo</h2>
      <div className="phrase-list">{objects.map(o => <div key={o.en} className="phrase scan-line">
        <div><strong lang={lang.speech}>{o.text ?? '…'}</strong>{o.text && romanize(chinese, id, o.text) && <em>{romanize(chinese, id, o.text)}</em>}<small>{o.en}</small>
          <div className="mini-row"><AddCard text={o.text ?? ''} en={o.en} romanization={o.text ? romanize(chinese, id, o.text) : ''} add={add} /></div></div>
      </div>)}</div>
    </>}
    <p className="learn-credit">Text and things are recognized on this phone by Google ML Kit; the photo is not uploaded. Translations
      are made on the phone too, after a one-time download of the {name} pack (about 30 MB, over Wi-Fi), and can be imperfect.</p>
  </main>;
}

export function Understand({ id, start, add, ask, say, voice, back }: { id: LanguageId; start: string; add: Add; ask?: (prompt: string) => void;
  say: Say; voice: boolean | null; back: () => void }) {
  const lang = language(id), name = lang.name.replace(/ \(.*\)$/, '');
  const [text, setText] = useState(start);
  const [toEnglish, setToEnglish] = useState(true);
  const [result, setResult] = useState<{ said: string; en: string } | null>(null);
  const [busy, setBusy] = useState(false), [problem, setProblem] = useState('');
  const [word, setWord] = useState<string | null>(null);
  const chinese = useChinese(id);
  const go = async (value = text, direction = toEnglish) => {
    const clean = value.trim();
    if (!clean) return;
    setWord(null); setProblem('');
    if (!canTranslate()) { setResult({ said: clean, en: '' }); return; }
    setBusy(true);
    try {
      const out = await translate(clean, direction ? id : 'en', direction ? 'en' : id);
      setResult(direction ? { said: clean, en: out } : { said: out, en: clean });
    } catch (error) { setProblem(message(error)); setResult(null); } finally { setBusy(false); }
  };
  useEffect(() => { if (start) void go(start, CJK_SCRIPT[id] ? inLanguage(start, id) : true); }, []);  // eslint-disable-line react-hooks/exhaustive-deps
  const edit = (value: string) => {
    setText(value);
    if (CJK_SCRIPT[id] && value.trim()) setToEnglish(inLanguage(value, id));
  };
  const pieces = !result ? [] : (chinese ? [...new Set(segment(chinese, result.said).filter(p => HAN.test(p)))] : words(result.said, lang.speech)).slice(0, 40);
  const pinyin = result ? romanize(chinese, id, result.said) : '';
  return <main className="page learn">
    <Back onClick={back} />
    <p className="eyebrow">{lang.name.toUpperCase()} · UNDERSTAND</p><h1>What does it say<span className="brand-dot">?</span></h1>
    <p className="page-subtitle">Paste or type a sentence. See what it means, hear it, and look up each word. Or write in English to see it in {name}.</p>
    <form className="understand" onSubmit={e => { e.preventDefault(); void go(); }}>
      <textarea value={text} onChange={e => edit(e.target.value)} rows={3} maxLength={500} lang={toEnglish ? lang.speech : 'en'}
        placeholder={toEnglish ? `Something in ${name}` : 'Something in English'} aria-label="Text to understand" />
      <div className="understand-actions">
        <button type="button" className="mini"
          onClick={() => setToEnglish(t => !t)} title="Switch direction">
          {toEnglish ? `${name} → English` : `English → ${name}`} ⇄</button>
        <button className="learn-primary" disabled={!text.trim() || busy}>{busy ? 'Translating…' : canTranslate() ? 'Translate' : 'Look up words'}</button>
      </div>
    </form>
    {problem && <p className="learn-warning" role="alert">{problem}</p>}
    {result && <div className="flashcard understood">
      <div className="said"><strong lang={lang.speech}>{result.said}</strong><Play text={result.said} say={say} voice={voice} /></div>
      {pinyin && <em>{pinyin}</em>}
      {result.en && <p>{result.en}</p>}
      {id === 'zh-Hant' && !toEnglish && <small>The phone writes Chinese translations in Simplified characters.</small>}
      <div className="mini-row center">
        {result.en && result.said.length <= 120 && <AddCard key={result.said} text={result.said} en={result.en} romanization={pinyin} add={add} label="Save as a card" />}
      </div>
    </div>}
    {pieces.length > 0 && <>
      <h2 className="learn-heading">Tap a word to look it up</h2>
      <div className="chips">{pieces.map(w => <button key={w} lang={lang.speech} aria-pressed={word === w} onClick={() => setWord(word === w ? null : w)}>{w}</button>)}</div>
    </>}
    {word && <WordPanel key={word} word={word} id={id} chinese={chinese} add={add} say={say} voice={voice} />}
    {ask && <button className="learn-secondary" onClick={() => ask(`teach me about ${name} grammar`)}><Icon name="chat" size={18} />Ask Kani about {name} grammar</button>}
    {ask && <p className="learn-credit">Kani looks grammar questions up on the web and answers with sources. You can change the question before sending it.</p>}
  </main>;
}

function WordPanel({ word, id, chinese, add, say, voice }: { word: string; id: LanguageId; chinese: Cedict | null; add: Add; say: Say; voice: boolean | null }) {
  const lang = language(id);
  const [meaning, setMeaning] = useState('');
  const [entry, setEntry] = useState<Entry | null | 'loading' | 'error'>('loading');
  const [examples, setExamples] = useState<Example[] | 'loading' | 'error' | null>(null);
  const isChinese = id === 'zh-Hans' || id === 'zh-Hant';
  const found: Hanzi[] | null = isChinese && chinese ? lookUp(chinese, word, id === 'zh-Hant' ? 'Hant' : 'Hans') : null;
  useEffect(() => {
    if (canTranslate()) void translate(word, id, 'en').then(setMeaning).catch(() => {});
    if (!isChinese) dictionary(word, id).then(setEntry).catch(() => setEntry('error'));
  }, [word, id, isChinese]);
  const best = found?.length ? found[0].meanings.split('; ').slice(0, 2).join('; ')
    : typeof entry === 'object' && entry ? entry.meanings[0].meanings[0] : meaning;
  return <section className="word-panel" aria-live="polite">
    <div className="said"><strong lang={lang.speech}>{word}</strong><Play text={word} say={say} voice={voice} /></div>
    {meaning && <p className="muted">Phone translation: {meaning}</p>}
    {isChinese && (!found ? <p className="muted">Opening the dictionary…</p> : !found.length ? <p className="muted">Not in the dictionary.</p> : <>
      {found.slice(0, 6).map((h, i) => <div key={i} className="meaning hanzi"><small><span lang={lang.speech}>{h.word}</span> · {joined(h.pinyin)}</small><p>{h.meanings}</p></div>)}
      <p className="credit">From CC-CEDICT (CC BY-SA 4.0), on this phone.</p></>)}
    {!isChinese && entry === 'loading' && <p className="muted">Looking it up…</p>}
    {!isChinese && entry === 'error' && <p className="muted">The dictionary is unavailable (are you online?).</p>}
    {!isChinese && entry === null && <p className="muted">Not in the dictionary{meaning ? '' : ' (it may be part of a longer word, or a name)'}.</p>}
    {!isChinese && typeof entry === 'object' && entry && <>
      {entry.word !== word && <p className="muted">Dictionary entry: <strong lang={lang.speech}>{entry.word}</strong></p>}
      {entry.meanings.map((m, i) => <div key={i} className="meaning"><small>{m.part}</small><ol>{m.meanings.map(x => <li key={x}>{x}</li>)}</ol>
        {m.examples.map(e => <p key={e.text} className="example"><span lang={lang.speech}>{e.text}</span>{e.en && <small>{e.en}</small>}</p>)}</div>)}
      <a className="credit" href={entry.url} target="_blank" rel="noopener noreferrer">From Wiktionary (CC BY-SA 4.0)</a>
    </>}
    <div className="mini-row">
      <AddCard text={typeof entry === 'object' && entry ? entry.word : word} en={best} add={add}
        romanization={found?.length && found[0].word === word ? joined(found[0].pinyin) : romanize(chinese, id, word)} />
      {examples === null && <button className="mini" onClick={() => { setExamples('loading'); sentencesWith(word, id).then(setExamples).catch(() => setExamples('error')); }}>
        <Icon name="write" size={14} />Sentences with it</button>}
    </div>
    {examples === 'loading' && <p className="muted">Finding sentences…</p>}
    {examples === 'error' && <p className="muted">Sentences are unavailable (are you online?).</p>}
    {Array.isArray(examples) && (examples.length ? <div className="phrase-list">{examples.map(s => <div key={s.text} className="phrase">
      <div><strong lang={lang.speech}>{s.text}</strong>{romanize(chinese, id, s.text) && <em>{romanize(chinese, id, s.text)}</em>}<small>{s.en}</small>
        <small className="credit">{s.credit}</small>
        <div className="mini-row"><AddCard text={s.text} en={s.en} romanization={romanize(chinese, id, s.text)} add={add} /></div></div>
      <Play text={s.text} say={say} voice={voice} /></div>)}</div>
      : <p className="muted">No example sentences with this word yet.</p>)}
    {Array.isArray(examples) && examples.length > 0 && <p className="learn-credit">Sentences from Tatoeba (tatoeba.org), CC BY 2.0 FR.</p>}
  </section>;
}

const when = (card?: Card) => {
  if (!card) return '';
  const [y, m, d] = card.due.split('-').map(Number), due = new Date(y, m - 1, d);
  return due <= new Date() ? 'Due today' : `${card.box >= 3 ? 'Known well · next' : 'Next practice'} ${due.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}`;
};

export function MyCards({ id, mine, cards, add, remove, practice, say, voice, back }: { id: LanguageId; mine: Mine[]; cards: Record<string, Card>;
  add: Add; remove: (cardId: string) => void; practice: (ids: string[]) => void; say: Say; voice: boolean | null; back: () => void }) {
  const lang = language(id), name = lang.name.replace(/ \(.*\)$/, '');
  const [text, setText] = useState(''), [en, setEn] = useState(''), [note, setNote] = useState(''), [busy, setBusy] = useState(false);
  const chinese = useChinese(id);
  const submit = () => {
    const added = add([[text, en, romanize(chinese, id, text)]]);
    setNote(added ? '' : 'You already have that card.');
    if (added) { setText(''); setEn(''); }
  };
  const fill = async () => {
    setBusy(true);
    try { setEn(await translate(text.trim(), id, 'en')); setNote(''); } catch (error) { setNote(message(error)); } finally { setBusy(false); }
  };
  return <main className="page learn">
    <Back onClick={back} />
    <p className="eyebrow">{lang.name.toUpperCase()}</p><h1>My cards<span className="brand-dot">.</span></h1>
    <p className="page-subtitle">Words and sentences you want to remember. They come back for practice with the lesson cards.</p>
    <form className="my-card-form" onSubmit={e => { e.preventDefault(); submit(); }}>
      <input value={text} onChange={e => setText(e.target.value)} lang={lang.speech} placeholder={`In ${name}`} aria-label={`Word or sentence in ${name}`} maxLength={200} />
      <div className="meaning-row"><input value={en} onChange={e => setEn(e.target.value)} placeholder="Meaning in English" aria-label="Meaning in English" maxLength={200} />
        {canTranslate() && <button type="button" className="mini" disabled={!text.trim() || busy} onClick={() => void fill()}>{busy ? '…' : 'Translate'}</button>}</div>
      <button className="learn-primary" disabled={!text.trim() || !en.trim()}><Icon name="plus" size={18} />Add card</button>
      {note && <small className="learn-warning">{note}</small>}
    </form>
    {mine.length > 0 && <button className="learn-secondary" onClick={() => practice(mine.map(m => m.id))}><Icon name="spark" size={18} />Practice all {mine.length}</button>}
    {!mine.length && <p className="muted">No cards yet. Add one above, or make them from a photo or a word you look up.</p>}
    <div className="phrase-list">{[...mine].reverse().map(m => <div key={m.id} className="phrase">
      <div><strong lang={lang.speech}>{m.text}</strong>{m.romanization && <em>{m.romanization}</em>}<small>{m.en}</small>
        <small className="credit">{when(cards[m.id])}</small></div>
      <Play text={m.text} say={say} voice={voice} />
      <button className="icon-button" aria-label={`Delete ${m.text}`} onClick={() => remove(m.id)}><Icon name="trash" size={18} /></button>
    </div>)}</div>
  </main>;
}
