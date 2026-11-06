import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { applyFixes, checkGrammar, PRACTICE, readingFeedback, readingSummary, spokenFeedback, tutor, upgrade } from '../../site/english.js';
import { ipa, numberWords, parseDictionary, phonesFor, soundDifferences, speakable, toKokoro } from '../../site/voice/pronounce.js';
import { kokoroChunks, kokoroIds } from '../../site/voice/kokoro.js';
import { logMel, resample, WhisperTokens } from '../../site/voice/whisper.js';

const words = parseDictionary(readFileSync(new URL('../../site/voice/cmudict.txt', import.meta.url), 'utf8'));
const fixed = text => applyFixes(text, checkGrammar(text, words));

test('learner mistakes are corrected', () => {
  const cases = [
    ['He go to school every day.', 'He goes to school every day.'],
    ['She always watch TV after dinner.', 'She always watches TV after dinner.'],
    ['My mom work in a hospital.', 'My mom works in a hospital.'],
    ['They likes pizza.', 'They like pizza.'],
    ['I is happy today.', 'I am happy today.'],
    ['She are my best friend.', 'She is my best friend.'],
    ['We was late.', 'We were late.'],
    ["He don't like coffee.", "He doesn't like coffee."],
    ['Does they know?', 'Do they know?'],
    ['He have a new car.', 'He has a new car.'],
    ['I can speaks English.', 'I can speak English.'],
    ["I didn't went to the party.", "I didn't go to the party."],
    ['Did you saw the movie?', 'Did you see the movie?'],
    ['She must to finish it.', 'She must finish it.'],
    ['I have went there twice.', 'I have gone there twice.'],
    ['She has ate already.', 'She has eaten already.'],
    ['Yesterday I go to the park.', 'Yesterday I went to the park.'],
    ['I play football last weekend.', 'I played football last weekend.'],
    ['I want go home now.', 'I want to go home now.'],
    ['I like play the guitar.', 'I like to play the guitar.'],
    ['I enjoy to swim.', 'I enjoy swimming.'],
    ['I am agree with you.', 'I agree with you.'],
    ['I am student.', 'I am a student.'],
    ['He is engineer.', 'He is an engineer.'],
    ['I ate a apple.', 'I ate an apple.'],
    ['It takes a hour.', 'It takes an hour.'],
    ['She goes to an university.', 'She goes to a university.'],
    ['This is more better.', 'This is better.'],
    ['He is taller then me.', 'He is taller than me.'],
    ['I need some informations.', 'I need some information.'],
    ['There are too much people here.', 'There are too many people here.'],
    ['I do not have many money.', 'I do not have much money.'],
    ['There is two cats.', 'There are two cats.'],
    ['People is nice here.', 'People are nice here.'],
    ['Can you explain me this?', 'Can you explain to me this?'],
    ["Let's discuss about the plan.", "Let's discuss the plan."],
    ['I listen music every day.', 'I listen to music every day.'],
    ['She is married with a doctor.', 'She is married to a doctor.'],
    ['I have 25 years.', 'I am 25 years old.'],
    ['I live here since three years.', 'I live here for three years.'],
    ['I go to home after work.', 'I go home after work.'],
    ['I make my homework.', 'I do my homework.'],
    ['It depends of the weather.', 'It depends on the weather.'],
    ["I don't know nothing.", "I don't know anything."],
    ['I could of done it.', 'I could have done it.'],
    ['i think so', 'I think so'],
    ['Your welcome!', "You're welcome!"],
    ['Its raining.', "It's raining."],
    ['I go there every days.', 'I go there every day.'],
    ['Me and my friend went to the park.', 'My friend and I went to the park.'],
    ['I am interested on music.', 'I am interested in music.'],
    ['We arrived to the airport.', 'We arrived at the airport.'],
    ['I was born in 1990 and I am good in math.', 'I was born in 1990 and I am good at math.'],
    ['Yesterday I go to the park and my friend play football.', 'Yesterday I went to the park and my friend played football.'],
    ['I buyed a apple and goed home.', 'I bought an apple and went home.'],
    ['She teached us and we catched the bus.', 'She taught us and we caught the bus.'],
  ];
  for (const [wrong, right] of cases) assert.equal(fixed(wrong), right, wrong);
});

test('correct English is left alone', () => {
  const good = [
    'He goes to school every day.', 'Does he like it?', 'What did she say?', 'Can you help me?', 'Let it go.', 'Make it work.',
    'I think he left yesterday.', 'I read a book yesterday.', 'There is a lot of water.', 'I need help.', 'I love music.',
    'He and I go running.', 'Yesterday was fun.', 'I will go tomorrow.', 'In May I went to Spain.', 'His will was clear.',
    'I have five years of experience.', 'Thank you so much!', 'A can of beans, please.', 'It has to work.', 'I sent you plans.',
    'We have visited Japan twice.', 'If I were you, I would wait.', 'That is an honest answer.', 'It is a one-time offer.',
    'I want to go home.', 'She is a teacher.', 'Her house is bigger than mine.', 'I like it.', 'They have been here since 2020.',
    'Everyday life is busy.', 'What does it mean?', 'I used to play there.', 'I did it yesterday.', "Don't you think so?",
    'I saw him last week and he looks happy.', 'You did a great job.', 'Have you seen it?', 'I go to the home page.',
    'Kani is a small model.', 'My mom and my dad work here.', 'I hope you and your dog enjoy it.', 'I usually walk to work in the morning.', 'How much does this coffee cost?', 'i.e. this one',
  ];
  for (const sentence of good) assert.deepEqual(checkGrammar(sentence, words).map(f => f.wrong + ' -> ' + f.right), [], sentence);
});

test('every fix explains itself', () => {
  for (const f of checkGrammar('He go to an university and buyed many informations yesterday.', words)) {
    assert.ok(f.why.length > 10 && f.right && f.wrong !== f.right, JSON.stringify(f));
  }
});

test('tutor notes and spoken feedback', () => {
  const wrong = tutor('He go to school.', words);
  assert.equal(wrong.corrected, 'He goes to school.');
  assert.match(spokenFeedback(wrong), /^Almost! Say: "He goes to school\." With "He", add -s/);
  const plain = tutor('The movie was very good.', words);
  assert.deepEqual(plain.fixes, []);
  assert.deepEqual(plain.upgrade, { phrase: 'very good', better: 'great, excellent' });
  assert.match(spokenFeedback(plain), /try "great"/);
  assert.ok(tutor('I like cats.', words, 3).tip);
  assert.equal(upgrade('It was fine.'), null);
});

test('pronunciation: dictionary, IPA, Kokoro phonemes', () => {
  assert.deepEqual(phonesFor('think', words), ['TH', 'IH1', 'NG', 'K']);
  assert.equal(ipa(phonesFor('water', words)), '/ˈwɔːtɚ/');
  assert.equal(ipa(phonesFor('important', words)), '/ˌɪmˈpɔːɹtənt/');
  assert.equal(ipa(phonesFor('kani', words)), '/ˈkɑːniː/');
  assert.deepEqual(phonesFor('AI', words), ['EY1', 'AY1']);
  assert.equal(numberWords(2026), 'two thousand twenty-six');
  assert.equal(speakable('It costs $5 in 2026 (**45%** off)'), 'It costs five dollars in twenty twenty-six (forty-five percent off)');
  assert.equal(toKokoro('Hello, water!', words), 'həlˈO, wˈɔɾɚ!');
  assert.deepEqual(soundDifferences(phonesFor('think', words), phonesFor('sink', words)), [{ wanted: 'TH', heard: 'S' }]);
  for (const { sentence } of PRACTICE) {
    for (const w of sentence.split(/\s+/)) assert.ok(words.has(w.toLowerCase().replace(/[^a-z']/g, '')), w);
    const phonemes = toKokoro(sentence, words);
    assert.equal(kokoroIds(phonemes).length, [...phonemes].length, sentence);
  }
  const long = 'This is a long sentence about many different things. '.repeat(30);
  assert.ok(kokoroChunks(long, words).every(chunk => chunk.length <= 400));
});

test('reading feedback finds the word that sounded wrong', () => {
  const heard = { words: [{ text: 'I' }, { text: 'sink' }, { text: '3' }, { text: 'is' }, { text: 'fine.' }] };
  const scored = [{ probability: 0.9 }, { probability: 0.05 }, { probability: 0.2 }, { probability: 0.99 }, { probability: 0.98 }];
  const feedback = readingFeedback('I think three is fine.', heard, scored, words);
  assert.deepEqual(feedback.words.map(w => w.status), ['good', 'wrong', 'good', 'good', 'good']);
  assert.equal(feedback.words[1].heard, 'sink');
  assert.equal(feedback.words[1].sounds[0].wanted, 'th (as in think)');
  assert.equal(feedback.score, 80);
  assert.match(readingSummary(feedback), /"think" sounded like "sink"\. Put the tip of your tongue/);
  const missed = readingFeedback('I like tea.', { words: [{ text: 'I' }, { text: 'tea.' }] }, [{ probability: 0.9 }, { probability: 0.1 }, { probability: 0.9 }], words);
  assert.deepEqual(missed.words.map(w => w.status), ['good', 'missed', 'good']);
});

test('whisper features and tokenizer', () => {
  const tokens = new WhisperTokens(JSON.parse(readFileSync(new URL('../../site/voice/whisper-tokens.json', import.meta.url), 'utf8')));
  const ids = tokens.encode(" I think it's fine.");
  assert.equal(tokens.decode(ids), " I think it's fine.");
  assert.deepEqual(ids, [314, 892, 340, 338, 3734, 13]);
  const tone = Float32Array.from({ length: 16000 }, (_, i) => Math.sin(2 * Math.PI * 440 * i / 16000) * 0.5);
  const mel = logMel(tone);
  assert.equal(mel.length, 80 * 3000);
  // 440 Hz lands in a low mel band; silence after one second sits at the floor.
  const column = t => Array.from({ length: 80 }, (_, m) => mel[m * 3000 + t]);
  const loud = column(50), quiet = column(2000);
  assert.ok(loud.indexOf(Math.max(...loud)) < 20 && Math.max(...quiet) < Math.max(...loud) - 1);
  assert.equal(resample(new Float32Array(48000), 48000, 16000).length, 16000);
});
