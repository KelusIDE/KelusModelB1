import { useEffect, useMemo, useState, type MutableRefObject } from 'react';
import { Preferences } from '@capacitor/preferences';
import { Icon } from './Icon';
import { LANGUAGES, SCRIPTS, TOPICS, english, language, type LanguageId, type Phrase } from './learn/content';
import SENTENCES from './learn/sentences.json';
import { day, due, known, practiced, review, type Card, type Grade, type Streak } from './learn/srs';
import { quiz, type Question } from './learn/quiz';
import { canSpeak, speak } from './learn/speak';
import { canListen, listen } from './learn/listen';
import { spoken, typed } from './learn/match';
import { remind } from './learn/remind';
import { items, own, wordOfTheDay, type Item, type Mine } from './learn/cards';
import { AddCard, Back, MyCards, Play, Scan, Understand, type Add } from './LearnTools';




const KEY = 'kani.learn.v1';
interface Saved { language?: LanguageId; cards: Partial<Record<LanguageId, Record<string, Card>>>; streak?: Streak;
  speaking?: boolean; reminder?: number | null; mine?: Partial<Record<LanguageId, Mine[]>> }
const REMINDERS: [number | null, string][] = [[null, 'Off'], [8, '8 am'], [12, 'Noon'], [19, '7 pm'], [21, '9 pm']];
type View = { name: 'home' } | { name: 'languages' } | { name: 'lesson'; topic: string } | { name: 'practice'; ids: string[] }
  | { name: 'quiz'; questions: Question[] } | { name: 'letters' } | { name: 'sentences' } | { name: 'scan' }
  | { name: 'understand'; text: string; from?: 'scan' } | { name: 'mine' };
type Sentence = { en: string; text: string; credit: string };
const random = Math.random;


export function Learn({ ask, backRef }: { ask?: (prompt: string) => void; backRef?: MutableRefObject<(() => boolean) | null> }) {
  const [saved, setSaved] = useState<Saved>({ cards: {} });
  const [ready, setReady] = useState(false);
  const [view, setView] = useState<View>({ name: 'home' });
  const [voice, setVoice] = useState<boolean | null>(null);
  const [ears, setEars] = useState(false);
  const [notice, setNotice] = useState('');
  useEffect(() => { void canListen().then(setEars); }, []);
  useEffect(() => {
    Preferences.get({ key: KEY }).then(({ value }) => { if (value) setSaved({ cards: {}, ...JSON.parse(value) }); })
      .catch(() => {}).finally(() => setReady(true));
  }, []);
  useEffect(() => { if (ready) void Preferences.set({ key: KEY, value: JSON.stringify(saved) }).catch(() => {}); }, [saved, ready]);
  const id = saved.language;
  useEffect(() => { setVoice(null); if (id) void canSpeak(language(id).speech).then(setVoice); }, [id]);
  const all = useMemo(() => id ? items(id, saved.mine?.[id]) : [], [id, saved.mine]);
  useEffect(() => {
    if (!backRef) return;
    const to: View | null = view.name === 'understand' && view.from === 'scan' ? { name: 'scan' } : view.name === 'home' || !id ? null : { name: 'home' };
    backRef.current = to && (() => { setView(to); return true; });
  });
  useEffect(() => () => { if (backRef) backRef.current = null; }, [backRef]);
  if (!ready) return <main className="page learn" />;
  if (!id || view.name === 'languages') {
    return <Languages chosen={id} choose={lang => { setSaved(s => ({ ...s, language: lang })); setView({ name: 'home' }); }} />;
  }
  const lang = language(id), cards = saved.cards[id] ?? {}, today = day();
  const say = (text: string) => void speak(text, lang.speech);
  const grade = (phraseId: string, value: Grade) => setSaved(s => ({ ...s, streak: practiced(s.streak, today),
    cards: { ...s.cards, [id]: { ...(s.cards[id] ?? {}), [phraseId]: review((s.cards[id] ?? {})[phraseId], value, today) } } }));
  const schedule = (ids: string[]) =>
    setSaved(s => ({ ...s, cards: { ...s.cards, [id]: { ...Object.fromEntries(ids.map(p => [p, { box: 0, due: today }])), ...(s.cards[id] ?? {}) } } }));
  const start = (ids: string[]) => { schedule(ids); setView({ name: 'practice', ids }); };
  const back = () => setView({ name: 'home' });
  const mine = saved.mine?.[id] ?? [];
  const add: Add = pairs => {
    const list = [...mine], fresh: Mine[] = [], stamp = Date.now();
    for (const [text, en, romanization] of pairs) {
      const card = own(text, en, today, list, stamp, romanization);
      if (card) { list.push(card); fresh.push(card); }
    }
    if (fresh.length) setSaved(s => ({ ...s, mine: { ...s.mine, [id]: [...(s.mine?.[id] ?? []), ...fresh] },
      cards: { ...s.cards, [id]: { ...(s.cards[id] ?? {}), ...Object.fromEntries(fresh.map(c => [c.id, { box: 0, due: today }])) } } }));
    return fresh.length;
  };
  const remove = (cardId: string) => setSaved(s => {
    const rest = { ...(s.cards[id] ?? {}) };
    delete rest[cardId];
    return { ...s, mine: { ...s.mine, [id]: (s.mine?.[id] ?? []).filter(m => m.id !== cardId) }, cards: { ...s.cards, [id]: rest } };
  });

  if (view.name === 'lesson') {
    const topic = TOPICS.find(t => t.id === view.topic)!;
    return <main className="page learn">
      <Back onClick={back} />
      <p className="eyebrow">{lang.name.toUpperCase()} · LESSON</p><h1>{topic.title}</h1>
      <div className="phrase-list">{topic.phrases.map(p => <PhraseRow key={p.id} phrase={p} id={id} say={say} voice={voice} />)}</div>
      <button className="learn-primary" onClick={() => start(topic.phrases.map(p => p.id))}><Icon name="spark" size={18} />Practice these {topic.phrases.length}</button>
    </main>;
  }
  if (view.name === 'practice') return <Practice ids={view.ids} all={all} id={id} say={say} voice={voice} grade={grade} done={back} speaking={ears && Boolean(saved.speaking)} />;
  if (view.name === 'quiz') return <Quiz questions={view.questions} id={id} say={say} done={back}
    wrong={p => grade(p, 'again')} right={() => setSaved(s => ({ ...s, streak: practiced(s.streak, today) }))} />;
  if (view.name === 'letters') {
    return <main className="page learn">
      <Back onClick={back} />
      <p className="eyebrow">{lang.name.toUpperCase()}</p><h1>Letters and sounds</h1>
      {(SCRIPTS[id] ?? []).map(script => <section key={script.title} className="settings-section">
        <h2>{script.title}</h2><p className="muted">{script.note}{voice ? ' Tap one to hear it.' : ''}</p>
        <div className="letters">{script.letters.map(l => <button key={l.letter + l.sound} lang={lang.speech} onClick={() => say(l.say)} aria-label={`${l.letter}, ${l.sound}`}>
          <strong>{l.letter}</strong><small>{l.sound}</small></button>)}</div>
      </section>)}
    </main>;
  }
  if (view.name === 'sentences') return <Sentences sentences={(SENTENCES as Record<string, Sentence[]>)[id] ?? []} id={id} say={say} voice={voice} add={add} back={back} />;
  if (view.name === 'scan' || (view.name === 'understand' && view.from === 'scan')) return <>  { }
    <Scan hidden={view.name !== 'scan'} id={id} add={add} understand={text => setView({ name: 'understand', text, from: 'scan' })} back={back} />
    {view.name === 'understand' && <Understand key={view.text} id={id} start={view.text} add={add} ask={ask} say={say} voice={voice} back={() => setView({ name: 'scan' })} />}
  </>;
  if (view.name === 'understand') return <Understand id={id} start={view.text} add={add} ask={ask} say={say} voice={voice} back={back} />;
  if (view.name === 'mine') return <MyCards id={id} mine={mine} cards={cards} add={add} remove={remove} say={say} voice={voice} back={back}
    practice={ids => setView({ name: 'practice', ids })} />;

  const waiting = due(cards, today), started = Object.keys(cards);
  const pool = started.length >= 4 ? all.filter(x => cards[x.id]) : all.filter(x => x.id.startsWith(`${TOPICS[0].id}.`));
  const word = wordOfTheDay(id, today);
  return <main className="page learn">
    <p className="eyebrow">LEARNING</p>
    <div className="learn-title"><h1 lang={lang.speech}>{lang.native}</h1><button className="secondary" onClick={() => setView({ name: 'languages' })}>Change</button></div>
    <p className="page-subtitle">{lang.name}</p>
    <div className="learn-stats">
      <div><strong>{saved.streak && saved.streak.last >= day(new Date(Date.now() - 864e5)) ? saved.streak.days : 0}</strong><small>day streak</small></div>
      <div><strong>{waiting.length}</strong><small>cards due</small></div>
      <div><strong>{known(cards)}</strong><small>cards known</small></div>
    </div>
    {voice === false && <p className="learn-note">This phone has no {lang.name} voice yet. To hear phrases, add one in the phone's settings
      (Text-to-speech, or Spoken content on iPhone).</p>}
    <button className="learn-primary" disabled={!waiting.length} onClick={() => setView({ name: 'practice', ids: waiting.slice(0, 20) })}>
      <Icon name="spark" size={18} />{waiting.length ? `Practice ${Math.min(20, waiting.length)} card${waiting.length === 1 ? '' : 's'}` : 'No cards due. Start a lesson below'}</button>
    <button className="learn-secondary" onClick={() => setView({ name: 'quiz', questions: quiz(pool, all, id, 8, random, voice !== false) })}>
      <Icon name="check" size={18} />Quiz me</button>
    <h2 className="learn-heading">Tools</h2>
    <div className="tools">
      <button onClick={() => setView({ name: 'scan' })}><Icon name="camera" /><strong>Scan</strong><small>Read a photo</small></button>
      <button onClick={() => setView({ name: 'understand', text: '' })}><Icon name="translate" /><strong>Understand</strong><small>Any sentence</small></button>
      <button onClick={() => setView({ name: 'mine' })}><Icon name="cards" /><strong>My cards</strong><small>{mine.length ? `${mine.length} saved` : 'Add your own'}</small></button>
    </div>
    <section className="today below"><small>PHRASE OF THE DAY</small>
      <div className="phrase"><div><strong lang={lang.speech}>{word.text}</strong>{word.romanization && <em>{word.romanization}</em>}<small>{word.en}</small></div>
        <Play text={word.say} say={say} voice={voice} /></div>
      <button className="mini" disabled={Boolean(cards[word.id])} onClick={() => schedule([word.id])}>
        {cards[word.id] ? <><Icon name="check" size={14} />In your cards</> : <><Icon name="plus" size={14} />Add to my cards</>}</button></section>
    <h2 className="learn-heading">Lessons</h2>
    <div className="lessons">{TOPICS.map(t => {
      const done = t.phrases.filter(p => cards[p.id]).length;
      return <button key={t.id} className="lesson" onClick={() => setView({ name: 'lesson', topic: t.id })}>
        <span><strong>{t.title}</strong><small>{done ? `${done} of ${t.phrases.length} practiced` : `${t.phrases.length} phrases`}</small></span><Icon name="chevron" size={16} /></button>;
    })}</div>
    <h2 className="learn-heading">More practice</h2>
    <div className="lessons">
      {SCRIPTS[id] && <button className="lesson" onClick={() => setView({ name: 'letters' })}><span><strong>Letters and sounds</strong><small>
        {id === 'ko' ? 'Hangul, Korean’s alphabet' : id === 'ja' ? 'Hiragana and katakana' : id.startsWith('zh') ? 'The four tones' : 'Special letters'}</small></span><Icon name="chevron" size={16} /></button>}
      <button className="lesson" onClick={() => setView({ name: 'sentences' })}><span><strong>Real sentences</strong><small>Read, listen, then check the meaning</small></span><Icon name="chevron" size={16} /></button>
    </div>
    <h2 className="learn-heading">Settings</h2>
    {ears && <label className="switch-row"><span><strong>Speaking practice</strong><small>Adds “Say it” to flashcards: say the phrase and see what the phone heard.
      Your phone’s speech service turns your voice into text; on Android that is Google, which may send the recording online.</small></span>
      <input type="checkbox" role="switch" checked={Boolean(saved.speaking)} onChange={e => setSaved(s => ({ ...s, speaking: e.target.checked }))} /></label>}
    <div className="reminder"><strong>Daily reminder</strong><small>A notification from the phone, at the same time every day.</small>
      <div className="segmented" role="group" aria-label="Daily reminder">{REMINDERS.map(([hour, label]) =>
        <button key={label} aria-pressed={(saved.reminder ?? null) === hour} onClick={async () => {
          const ok = await remind(hour, lang.name);
          setNotice(ok ? '' : 'Notifications are turned off for Kani. Allow them in the phone settings.');
          if (ok) setSaved(s => ({ ...s, reminder: hour }));
        }}>{label}</button>)}</div>
      {notice && <small className="learn-warning">{notice}</small>}</div>
    <p className="learn-credit">Sentences from Tatoeba (tatoeba.org), CC BY 2.0 FR. Dictionary from Wiktionary, CC BY-SA 4.0.
      Photos are read and translated on the phone by Google ML Kit.</p>
  </main>;
}

function Languages({ chosen, choose }: { chosen?: LanguageId; choose: (id: LanguageId) => void }) {
  return <main className="page learn">
    <p className="eyebrow">LEARN A LANGUAGE</p><h1>Which language<span className="brand-dot">?</span></h1>
    <p className="page-subtitle">Everyday phrases, flashcards and quizzes, and tools to read photos and look words up. Your progress stays on this phone.</p>
    <div className="language-grid">{LANGUAGES.map(l => <button key={l.id} className={l.id === chosen ? 'selected' : ''} onClick={() => choose(l.id)}>
      <strong lang={l.speech}>{l.native}</strong><small>{l.name}</small></button>)}</div>
  </main>;
}

function PhraseRow({ phrase, id, say, voice }: { phrase: Phrase; id: LanguageId; say: (text: string) => void; voice: boolean | null }) {
  const form = phrase.forms[id];
  return <div className="phrase">
    <div><strong lang={language(id).speech}>{form.text}</strong>{form.romanization && <em>{form.romanization}</em>}<small>{english(phrase, id)}</small></div>
    <Play text={form.say} say={say} voice={voice} />
  </div>;
}

function Practice({ ids, all, id, say, voice, grade, done, speaking }: { ids: string[]; all: Item[]; id: LanguageId; say: (text: string) => void;
  voice: boolean | null; grade: (card: string, value: Grade) => void; done: () => void; speaking: boolean }) {
  const [queue, setQueue] = useState(() => ids.filter(card => all.some(x => x.id === card))), [shown, setShown] = useState(false), [count, setCount] = useState(0);
  const [heard, setHeard] = useState<{ text: string; verdict: string } | 'listening' | null>(null);
  const form = all.find(x => x.id === queue[0]);
  if (!form) return <main className="page learn"><p className="eyebrow">PRACTICE</p><h1>Done<span className="brand-dot">!</span></h1>
    <p className="page-subtitle">{count} card{count === 1 ? '' : 's'} practiced. Forgotten ones come back soon; known ones wait longer.</p>
    <button className="learn-primary" onClick={done}>Back to lessons</button></main>;
  const answer = (value: Grade) => {
    grade(form.id, value);
    setCount(c => c + 1); setShown(false); setHeard(null);
    setQueue(q => value === 'again' && q.filter(p => p === form.id).length < 2 ? [...q.slice(1), form.id] : q.slice(1));
  };
  return <main className="page learn">
    <Back onClick={done} />
    <p className="eyebrow">PRACTICE · {queue.length} LEFT</p>
    <div className="flashcard">
      <small>How do you say</small><h2>{form.en}</h2>
      {shown ? <div className="answer"><strong lang={language(id).speech}>{form.text}</strong>{form.romanization && <em>{form.romanization}</em>}
        <Play text={form.say} say={say} voice={voice} />
        {speaking && <button className="learn-secondary" disabled={heard === 'listening'} onClick={async () => {
          setHeard('listening');
          try {
            const result = spoken(form.say, await listen(language(id).speech));
            setHeard({ text: result.heard, verdict: result.verdict });
          } catch (error) { setHeard({ text: error instanceof Error ? error.message : 'Could not listen.', verdict: 'error' }); }
        }}>🎤 {heard === 'listening' ? 'Listening…' : 'Say it'}</button>}
        {heard && heard !== 'listening' && <p className={`heard ${heard.verdict}`}>{heard.verdict === 'error' ? heard.text
          : <>{heard.verdict === 'great' ? 'Great!' : heard.verdict === 'almost' ? 'Almost.' : 'Try again.'} I heard: <span lang={language(id).speech}>{heard.text || '(nothing)'}</span></>}</p>}
      </div>
        : <button className="learn-primary" onClick={() => { setShown(true); if (voice) say(form.say); }}>Show the answer</button>}
    </div>
    {shown && <div className="grades">
      <button onClick={() => answer('again')}>Forgot<small>see it again soon</small></button>
      <button onClick={() => answer('good')}>Knew it<small>a bit later</small></button>
      <button onClick={() => answer('easy')}>Easy<small>much later</small></button>
    </div>}
  </main>;
}

function Quiz({ questions, id, say, done, wrong, right }: { questions: Question[]; id: LanguageId; say: (text: string) => void; done: () => void;
  wrong: (card: string) => void; right: () => void }) {
  const [index, setIndex] = useState(0), [picked, setPicked] = useState<number | null>(null), [score, setScore] = useState(0);
  const [entry, setEntry] = useState(''), [checked, setChecked] = useState<ReturnType<typeof typed> | null>(null);
  const q = questions[index];
  useEffect(() => { if (q?.kind === 'listen') say(q.item.say); }, [index]);  // eslint-disable-line react-hooks/exhaustive-deps
  if (!q) return <main className="page learn"><p className="eyebrow">QUIZ</p><h1>{score} of {questions.length}<span className="brand-dot">.</span></h1>
    <p className="page-subtitle">{score === questions.length ? 'Perfect!' : 'Missed phrases go back into your cards for practice.'}</p>
    <button className="learn-primary" onClick={done}>Back to lessons</button></main>;
  const form = q.item;
  const choose = (i: number) => {
    if (picked !== null) return;
    setPicked(i);
    if (i === q.answer) { setScore(s => s + 1); right(); } else wrong(q.item.id);
  };
  return <main className="page learn">
    <Back onClick={done} />
    <p className="eyebrow">QUIZ · {index + 1} OF {questions.length}</p>
    <div className="flashcard">
      {q.kind === 'meaning' && <><small>What does this mean?</small><h2 lang={language(id).speech}>{form.text}</h2>{form.romanization && <em>{form.romanization}</em>}</>}
      {q.kind === 'phrase' && <><small>How do you say</small><h2>{form.en}</h2></>}
      {q.kind === 'listen' && <><small>Listen, then choose what it means</small><button className="play big" onClick={() => say(form.say)} aria-label="Listen again">▶</button></>}
      {q.kind === 'type' && <><small>Type it in {language(id).name.replace(/ \(.*\)$/, '')}</small><h2>{form.en}</h2></>}
    </div>
    {q.kind === 'type' && <form className="type-answer" onSubmit={e => {
      e.preventDefault();
      if (checked) return;
      const result = typed(form.text.replace(/\(.*?\)/g, ''), entry);
      setChecked(result);
      if (result === 'right' || result === 'accents') { setScore(s => s + 1); right(); } else wrong(q.item.id);
    }}>
      <input lang={language(id).speech} value={entry} onChange={e => setEntry(e.target.value)} autoCapitalize="off" autoCorrect="off" spellCheck={false} disabled={checked !== null} aria-label="Your answer" />
      {checked === null ? <button className="learn-primary" disabled={!entry.trim()}>Check</button>
        : <p className={`heard ${checked === 'right' ? 'great' : checked === 'wrong' ? 'again' : 'almost'}`}>{
          { right: 'Right!', accents: 'Right, but watch the accents:', close: 'Close! It is', wrong: 'Not quite. It is' }[checked]} {checked !== 'right' && <strong lang={language(id).speech}>{form.text}</strong>}</p>}
    </form>}
    {q.kind !== 'type' && <div className="options">{q.options.map((option, i) => <button key={option} lang={q.kind === 'phrase' ? language(id).speech : undefined}
      className={picked === null ? '' : i === q.answer ? 'right' : i === picked ? 'wrong' : ''} onClick={() => choose(i)}>{option}</button>)}</div>}
    {(picked !== null || checked !== null) && <button className="learn-primary" onClick={() => { setPicked(null); setChecked(null); setEntry(''); setIndex(i => i + 1); }}>
      {index + 1 < questions.length ? 'Next' : 'See my score'}</button>}
  </main>;
}

function Sentences({ sentences, id, say, voice, add, back }: { sentences: Sentence[]; id: LanguageId; say: (text: string) => void; voice: boolean | null;
  add: Add; back: () => void }) {
  const [start, setStart] = useState(() => Math.floor(random() * Math.max(1, sentences.length - 10)));
  const [open, setOpen] = useState<Set<number>>(new Set());
  const page = useMemo(() => sentences.slice(start, start + 10), [sentences, start]);
  return <main className="page learn">
    <Back onClick={back} />
    <p className="eyebrow">{language(id).name.toUpperCase()}</p><h1>Real sentences</h1>
    <p className="page-subtitle">Read each one (and listen), guess what it means, then tap to check.</p>
    <div className="phrase-list">{page.map((s, i) => <div key={s.text} className="phrase">
      <div><strong lang={language(id).speech}>{s.text}</strong>
        <button className="reveal" onClick={() => setOpen(o => new Set(o).add(start + i))}>{open.has(start + i) ? s.en : 'Show the meaning'}</button>
        {open.has(start + i) && <><small className="credit">{s.credit}</small><div className="mini-row"><AddCard text={s.text} en={s.en} add={add} /></div></>}</div>
      <Play text={s.text} say={say} voice={voice} />
    </div>)}</div>
    <button className="learn-secondary" onClick={() => { setStart((start + 10) % Math.max(1, sentences.length - 10)); setOpen(new Set()); }}>More sentences</button>
    <p className="learn-credit">From Tatoeba (tatoeba.org), CC BY 2.0 FR. Each sentence's authors are shown with its meaning.</p>
  </main>;
}
