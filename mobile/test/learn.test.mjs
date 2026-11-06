
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { LANGUAGES, PHRASES, SCRIPTS, TOPICS, english } from '../src/learn/content.ts';
import { addDays, due, known, practiced, review } from '../src/learn/srs.ts';
import { quiz } from '../src/learn/quiz.ts';
import { items, own, wordOfTheDay } from '../src/learn/cards.ts';

const SIMPLIFIED_ONLY = /[们这说时国会来对发后个为学经动现机开关长问见进实两车东门应电气头从书难爱让认识读写听买卖钱银医药话语请谢体题华业产网线结给红绿鸟鱼马龙风飞云万与区么吗饭]/;
const TRADITIONAL_ONLY = /[們這說時國會來對發後個為學經動現機開關長問見進實兩車東門應電氣頭從書難愛讓認識讀寫聽買賣錢銀醫藥話語請謝體題華業產網線結給紅綠鳥魚馬龍風飛雲萬與區麼嗎飯]/;

test('every lesson phrase exists in all eight forms, romanized where the script needs it', () => {
  assert.equal(TOPICS.length, 10);
  assert.equal(PHRASES.length, 100);
  assert.equal(new Set(PHRASES.map(p => p.id)).size, 100);
  for (const phrase of PHRASES) {
    for (const lang of LANGUAGES) {
      const form = phrase.forms[lang.id];
      assert.ok(form.text && form.say && !form.text.includes('|'), `${phrase.id} ${lang.id}`);
      assert.equal(Boolean(form.romanization), lang.romanized, `${phrase.id} ${lang.id} romanization`);
      assert.ok(!english(phrase, lang.id).includes('{'), phrase.id);
    }
    assert.ok(!SIMPLIFIED_ONLY.test(phrase.forms['zh-Hant'].text), `${phrase.id}: ${phrase.forms['zh-Hant'].text} is not Traditional`);
    assert.ok(!TRADITIONAL_ONLY.test(phrase.forms['zh-Hans'].text), `${phrase.id}: ${phrase.forms['zh-Hans'].text} is not Simplified`);
    assert.ok(/[가-힣]/.test(phrase.forms.ko.text) && /[぀-ヿ一-鿿]/.test(phrase.forms.ja.text), phrase.id);
  }
  const sorry = PHRASES.find(p => p.en === "I'm sorry").forms.fr;
  assert.deepEqual([sorry.text, sorry.say], ['Je suis désolé(e)', 'Je suis désolé']);
  assert.equal(english(PHRASES.find(p => p.en.startsWith("I'm learning")), 'zh-Hant'), "I'm learning Chinese");
  assert.equal(SCRIPTS.ja[0].letters.length, 46);
  assert.deepEqual(SCRIPTS.ja[0].letters[5], { letter: 'か', sound: 'ka', say: 'か' });
});

test('flashcards come back sooner when forgotten and later each time they are known', () => {
  const today = '2026-10-25';
  let card = review(undefined, 'good', today);
  assert.deepEqual(card, { box: 1, due: '2026-10-26' });
  card = review(card, 'good', card.due);
  assert.deepEqual(card, { box: 2, due: '2026-10-29' });
  card = review(card, 'easy', card.due);
  assert.deepEqual(card, { box: 4, due: '2026-11-12' });
  assert.deepEqual(review(card, 'again', '2026-11-12'), { box: 0, due: '2026-11-12' });
  assert.deepEqual(review({ box: 5, due: today }, 'easy', today), { box: 5, due: '2026-11-24' });
  const cards = { a: { box: 2, due: '2026-10-24' }, b: { box: 0, due: '2026-10-25' }, c: { box: 0, due: '2026-10-26' }, d: { box: 4, due: '2026-10-20' } };
  assert.deepEqual(due(cards, today), ['d', 'a', 'b']);
  assert.equal(known(cards), 1);
  assert.equal(addDays('2026-12-31', 1), '2027-01-01');
  assert.deepEqual(practiced(undefined, today), { last: today, days: 1 });
  assert.deepEqual(practiced({ last: '2026-10-24', days: 4 }, today), { last: today, days: 5 });
  assert.deepEqual(practiced({ last: today, days: 5 }, today), { last: today, days: 5 });
  assert.deepEqual(practiced({ last: '2026-10-20', days: 9 }, today), { last: today, days: 1 });
});

test('quizzes ask about the chosen phrases with four different options and one right answer', () => {
  let seed = 7;
  const random = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const ids = TOPICS[3].phrases.map(p => p.id);
  for (const lang of ['ko', 'es', 'zh-Hant']) {
    const all = items(lang), pool = all.filter(x => ids.includes(x.id));
    const questions = quiz(pool, all, lang, 8, random);
    assert.equal(questions.length, 8);
    assert.deepEqual(new Set(questions.map(q => q.kind)), new Set(['meaning', 'phrase', 'listen', ...(lang === 'es' ? ['type'] : [])]));
    for (const q of questions) {
      const phrase = PHRASES.find(p => p.id === q.item.id);
      assert.ok(ids.includes(q.item.id));
      assert.equal(q.options.length, 4);
      assert.equal(new Set(q.options).size, 4);
      assert.equal(q.options[q.answer], q.kind === 'phrase' ? phrase.forms[lang].text : english(phrase, lang));
    }
  }
  const all = items('fr');
  assert.ok(quiz(all.slice(0, 10), all, 'fr', 6, random, false).every(q => q.kind !== 'listen'));
});

test('bundled practice sentences: 300 per language, credited, in the right Chinese script', async () => {
  const sentences = JSON.parse(await readFile(new URL('../src/learn/sentences.json', import.meta.url), 'utf8'));
  assert.deepEqual(Object.keys(sentences).sort(), LANGUAGES.map(l => l.id).sort());
  for (const [id, list] of Object.entries(sentences)) {
    assert.equal(list.length, 300, id);
    assert.ok(list.every(s => s.en && s.text && /^tatoeba\.org #\d+/.test(s.credit)), id);
  }
  assert.ok(sentences['zh-Hant'].every(s => !SIMPLIFIED_ONLY.test(s.text)));
  assert.ok(sentences['zh-Hans'].every(s => !TRADITIONAL_ONLY.test(s.text)));
});

test('spoken and typed answers: case, spaces and punctuation never count; accents are pointed out', async () => {
  const { similarity, spoken, typed } = await import('../src/learn/match.ts');
  assert.equal(similarity('¿Cómo está?', 'cómo está'), 1);
  assert.equal(spoken('안녕하세요', ['안녕 하세요', '안녕하세오']).verdict, 'great');
  assert.equal(spoken('Buenos días', ['buenas tardes']).verdict, 'again');
  assert.equal(spoken('Muchas gracias', ['mucha gracias']).verdict, 'great');
  assert.equal(spoken('Wie viel kostet das?', ['wie viel kostet']).verdict, 'almost');
  assert.deepEqual(spoken('Hola', []), { heard: '', score: 0, verdict: 'again' });
  assert.equal(typed('¿Dónde está el baño?', 'donde esta el bano'), 'accents');
  assert.equal(typed('¿Dónde está el baño?', 'Dónde está el baño'), 'right');
  assert.equal(typed('Je ne comprends pas', 'je ne compren pas'), 'close');
  assert.equal(typed('Je ne comprends pas', 'merci'), 'wrong');
});

test('typing questions come only for languages typed on an ordinary keyboard', () => {
  let seed = 3;
  const random = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const pool = lang => items(lang).filter(x => x.id.startsWith(`${TOPICS[1].id}.`));
  assert.ok(quiz(pool('es'), items('es'), 'es', 8, random).some(q => q.kind === 'type'));
  assert.ok(quiz(pool('ko'), items('ko'), 'ko', 8, random).every(q => q.kind !== 'type'));
  assert.ok(quiz(pool('zh-Hant'), items('zh-Hant'), 'zh-Hant', 8, random).every(q => q.kind !== 'type'));
});

test("the learner's own cards: kept once each, studied and quizzed like lesson phrases", () => {
  const today = '2026-10-25';
  const first = own('  사과 ', 'apple', today, [], 1000);
  assert.deepEqual(first, { id: 'my.rs.0', text: '사과', en: 'apple', added: today });
  assert.equal(own('사과', 'an apple', today, [first]), null);
  assert.equal(own('물', '  ', today, [first]), null);
  const mine = [first, own('물', 'water', today, [first], 1001), own('책', 'book', today, [first], 1002), own('고양이', 'cat', today, [first], 1003)];
  assert.equal(new Set(mine.map(m => m.id)).size, 4);
  const all = items('ko', mine);
  assert.equal(all.length, 104);
  assert.deepEqual(all.at(-1), { id: mine[3].id, text: '고양이', say: '고양이', en: 'cat', mine: true });
  let seed = 11;
  const random = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const questions = quiz(all.filter(x => x.mine), all, 'ko', 4, random);
  assert.equal(questions.length, 4);
  for (const q of questions) {
    assert.ok(q.item.mine);
    assert.equal(new Set(q.options).size, 4);
    assert.ok(q.options.includes(q.kind === 'phrase' ? q.item.text : q.item.en));
  }

  assert.deepEqual(wordOfTheDay('ja', today), wordOfTheDay('ja', today));
  assert.ok(new Set(['2026-10-25', '2026-10-26', '2026-10-27', '2026-10-28'].map(d => wordOfTheDay('es', d).id)).size > 1);
});

test('looking words up: words split in every script, dictionary forms of Korean words, Wiktionary and Tatoeba read', async () => {
  const { words, stems, plain, dictionary, sentencesWith } = await import('../src/learn/lookup.ts');
  assert.deepEqual(words('저는 학생이에요. 저는!', 'ko-KR'), ['저는', '학생이에요']);
  assert.deepEqual(words('私は学生です。', 'ja-JP'), ['私', 'は', '学生', 'です']);
  assert.deepEqual(words('我是学生，123。', 'zh-CN').join(''), '我是学生');
  assert.deepEqual(words('¿Dónde está el baño?', 'es-MX'), ['Dónde', 'está', 'el', 'baño']);
  assert.deepEqual(stems('책을', 'ko'), ['책을', '책']);
  assert.deepEqual(stems('사랑해요', 'ko'), ['사랑해요', '사랑하다']);
  assert.ok(stems('먹었어요', 'ko').includes('먹다'));
  assert.ok(stems('학생이에요', 'ko').includes('학생'));
  assert.deepEqual(stems('hund', 'de'), ['hund', 'Hund']);
  assert.deepEqual(stems('我是', 'zh-Hans'), ['我是', '我', '是']);
  assert.deepEqual(stems('です', 'ja'), ['です', 'で', 'す']);
  assert.equal(plain('to <a href="/wiki/eat">eat</a> &amp; <b>drink</b> , here'), 'to eat & drink, here');

  const asked = [];
  const reply = (status, body) => ({ status, ok: status < 400, json: async () => body });
  const wiktionary = async url => {
    asked.push(decodeURIComponent(url.split('/').pop()));
    return url.endsWith(encodeURIComponent('사랑하다')) ? reply(200, { ko: [{ partOfSpeech: 'Verb', definitions: [
      { definition: 'to <a>love</a>', parsedExamples: [{ example: '<b>사랑해</b>', translation: 'I love you' }] }, { definition: '' }, { definition: '愛:' }] },
      { partOfSpeech: 'Syllable', definitions: [{ definition: '<span>舍:</span> to dwell' }] }],
      en: [{ partOfSpeech: 'Noun', definitions: [{ definition: 'not this one' }] }] }) : reply(404, {});

  };
  const entry = await dictionary('사랑해요', 'ko', wiktionary);
  assert.deepEqual(asked, ['사랑해요', '사랑하다']);
  assert.deepEqual(entry, { word: '사랑하다', url: 'https://en.wiktionary.org/wiki/' + encodeURIComponent('사랑하다'),
    meanings: [{ part: 'Verb', meanings: ['to love'], examples: [{ text: '사랑해', en: 'I love you' }] }] });
  assert.equal(await dictionary('zzz', 'es', async () => reply(404, {})), null);
  await assert.rejects(dictionary('hola', 'es', async () => reply(503, {})), /unavailable/);

  let query;
  const sentence = (id, text, script, translations, extra = {}) => ({ id, text, script, owner: 'ana', is_unapproved: false, translations, ...extra });
  const en = (id, text, direct = true) => ({ id, text, owner: 'bo', is_unapproved: false, is_direct: direct });
  const tatoeba = async url => { query = new URL(url).searchParams; return reply(200, { data: [
    sentence(1, '說！', 'Hant', [en(10, 'Talk!')]),
    sentence(2, '他說得很慢。', 'Hant', [en(11, 'He speaks slowly.', false), en(12, 'He talks slowly.')]),
    sentence(3, '他说得很慢。', 'Hans', [en(13, 'He speaks slowly.')]),
    sentence(4, '我們說中文吧。', 'Hant', []),
    sentence(5, '請說慢一點。', 'Hant', [en(14, 'Please speak slowly.')], { is_unapproved: true }),
  ] }); };
  assert.deepEqual(await sentencesWith('說', 'zh-Hant', tatoeba), [{ text: '他說得很慢。', en: 'He talks slowly.', credit: 'tatoeba.org #2 (ana) & #12 (bo)' }]);
  assert.deepEqual([query.get('lang'), query.get('q'), query.get('trans:lang'), query.get('sort')], ['cmn', '說', 'eng', 'words']);
});

test('Chinese: words found in either script with pinyin, split the way a reader would', async () => {
  const { readCedict, segment, pinyinOf, lookUp } = await import('../src/learn/chinese.ts');
  const { pinyin } = await import('../scripts/make-cedict.mjs');
  assert.equal(pinyin('xue2 sheng5'), 'xué sheng');
  assert.equal(pinyin('nu:3 er2'), 'nǚ ér');
  assert.equal(pinyin('Bei3 jing1'), 'Běi jīng');
  assert.equal(pinyin('gou3 xiong2'), 'gǒu xióng');
  assert.equal(pinyin('liu2'), 'liú');
  assert.equal(pinyin('shui3 guo3 A1'), 'shuǐ guǒ Ā');
  const map = readCedict([
    ['我', '我', 'wǒ', 'I; me'], ['不', '不', 'bù', 'not'], ['明白', '明白', 'míng bai', 'to understand'], ['不明', '不明', 'bù míng', 'unknown'],
    ['白', '白', 'bái', 'white'], ['學生', '学生', 'xué sheng', 'student'], ['是', '是', 'shì', 'to be'], ['說', '说', 'shuì', 'to persuade'],
    ['說', '说', 'shuō', 'to speak; to say'], ['了', '了', 'liǎo', 'to finish; to understand clearly'], ['了', '了', 'le', 'completed action marker'],
    ['西安', '西安', 'Xī ān', "Xi'an city"],
  ].map(row => row.join('\t')).join('\n'));
  assert.deepEqual(segment(map, '我不明白'), ['我', '不', '明白']);
  assert.equal(pinyinOf(map, '我是學生。'), 'wǒ shì xuésheng.');
  assert.equal(pinyinOf(map, '我是学生，不是ABC 123！'), 'wǒ shì xuésheng, bù shì ABC 123!');
  assert.equal(pinyinOf(map, '我說了'), 'wǒ shuō le');
  assert.equal(pinyinOf(map, '西安'), "Xī'ān");
  assert.deepEqual(lookUp(map, '说', 'Hant').map(h => [h.word, h.pinyin]), [['說', 'shuō'], ['說', 'shuì']]);
  assert.deepEqual(lookUp(map, '我是', 'Hans').map(h => h.word), ['我', '是']);
});

test('translation questions in chat go to the phone translator; everything else stays with Kani', async () => {
  const { translationAsk, translationReply, guessLanguage } = await import('../src/learn/ask.ts');
  const cases = [
    ['How do you say "thank you" in Korean?', { text: 'thank you', from: 'en', to: 'ko' }],
    ['how do i say where is the bathroom in spanish', { text: 'where is the bathroom', from: 'en', to: 'es' }],
    ['Translate to Japanese: Good morning', { text: 'Good morning', from: 'en', to: 'ja' }],
    ['What is "love" in French?', { text: 'love', from: 'en', to: 'fr' }],
    ['Can you translate "I am hungry" into Mandarin?', { text: 'I am hungry', from: 'en', to: 'zh' }],
    ['German for "good night"?', { text: 'good night', from: 'en', to: 'de' }],
    ['translate hello to italian please', { text: 'hello', from: 'en', to: 'it' }],
    ['What does "감사합니다" mean?', { text: '감사합니다', from: 'ko', to: 'en' }],
    ['What does 我不明白 mean in English?', { text: '我不明白', from: 'zh', to: 'en' }],
    ['what is the meaning of "Je ne comprends pas"', { text: 'Je ne comprends pas', from: 'fr', to: 'en' }],
    ['Translate to English: ¿Dónde está el baño?', { text: '¿Dónde está el baño?', from: 'es', to: 'en' }],
    ['What does "Tudo bem" mean?', { text: 'Tudo bem', from: 'pt', to: 'en' }],
    ['What does "doch" mean in German?', { text: 'doch', from: 'de', to: 'en' }],
  ];
  for (const [message, ask] of cases) assert.deepEqual(translationAsk(message), ask, message);
  for (const message of ['What does "serendipity" mean?', 'What is the capital of France?', 'How do you say no to a friend nicely?',
    'teach me about Korean particles', 'What is love?', 'Translate this sentence for me', 'How do you say thank you in English?']) {
    assert.equal(translationAsk(message), null, message);
  }
  assert.equal(guessLanguage('Wie geht es dir?'), 'de');
  assert.equal(guessLanguage('Obrigado, tudo bem'), 'pt');
  assert.equal(guessLanguage('こんにちは'), 'ja');
  assert.equal(translationReply({ text: 'thank you', from: 'en', to: 'ko' }, '감사합니다'), 'In Korean: 감사합니다');
  assert.equal(translationReply({ text: '감사합니다', from: 'ko', to: 'en' }, 'Thank you.'), 'It means "Thank you." (Korean).');
});

test('chat messages in other languages are recognized and English is left alone', async () => {
  const { chatLanguage, previousLanguage, keptLines } = await import('../src/learn/ask.ts');
  for (const [message, code] of [['¿Cuál es la capital de España?', 'es'], ['Hola, tengo un problema con mi amigo.', 'es'], ['Hola', 'es'], ['quiero morir', 'es'], ['quiero comer', 'es'], ['quero morrer', 'pt'], ['je veux mourir', 'fr'], ['ich will sterben', 'de'], ['voglio morire', 'it'], ['죽고 싶어', 'ko'], ['死にたい', 'ja'], ['我想死', 'zh'],
    ['Bonjour, comment ça va ?', 'fr'], ['merci', 'fr'], ['Was ist die Hauptstadt von Deutschland?', 'de'], ['Wie geht es dir?', 'de'],
    ['Qual é a capital do Brasil?', 'pt'], ['Obrigado, tudo bem', 'pt'], ['Dove sono le chiavi? Non lo so', 'it'],
    ['한국의 수도는 어디예요?', 'ko'], ['日本の首都はどこですか？', 'ja'], ['中国的首都是哪里？', 'zh'], ['Как дела?', 'ru'], ['Busca en internet noticias de Madrid', 'es'], ['Encuentra pizza cerca de mí', 'es'], ['Ich bin traurig', 'de'], ['Estou muito triste hoje', 'pt'], ['Je suis triste', 'fr'], ['QUIERO MORIR', 'es'], ['Suche Pizza in meiner Nähe', 'de'], ['Estoy en Madrid', 'es']]) assert.equal(chatLanguage(message), code, message);
  for (const message of ['What is the capital of France?', 'hello', 'Thanks!', 'Die Hard', 'die hard war movie', 'MIT', 'da vinci', 'O Canada', 'La La Land', 'El Paso',
    'as far as i know', 'do as i say', 'Las Vegas', 'Was Einstein smart?', 'AI', 'is 5pm est ok', 'What is 東京?', 'What is El Niño?', 'comment this code', 'for i in range(10): print(i)',
    'pizza', 'Paris', 'ok', '42', '', 'El Paso Texas weather', 'Buenos Aires hotels', 'Costa Rica', 'Rio de Janeiro', 'Santa Cruz beach', 'La Paz Bolivia', 'Puerto Rico flights',
    'Porto Alegre', 'Los Angeles Lakers', 'Les Miserables', 'Der Spiegel', 'Das Boot', 'Mi Band 8', 'Carpe diem', 'If sin^2 A - cos^2 A = 2 then find cos 2A.']) assert.equal(chatLanguage(message), null, message);
  assert.equal(chatLanguage('ok', 'es'), 'es');
  assert.equal(chatLanguage('Paris', 'fr'), 'fr');
  assert.equal(chatLanguage('What is the capital of France?', 'es'), null);
  assert.equal(chatLanguage('thanks', 'es'), null);
  const user = (content, english) => ({ role: 'user', content, english });
  assert.equal(previousLanguage([user('Hola', 'Hello'), { role: 'assistant', content: 'x' }, user('ok', 'ok')]), 'es');
  assert.equal(previousLanguage([user('Hello')]), null);
  assert.deepEqual(keptLines('Here is code:\n```js\nlet a;\n```\n\n2 + 2 = 4\nSources:\n[1] Paris — https://x.test'), [false, true, true, true, true, true, false, true]);
  assert.deepEqual(keptLines('Here are places.\n1. Joe’s Pizza · 4.6\n   1435 Broadway\n   https://m.test\nOpen late', 'places'), [false, true, true, true, false]);
});
