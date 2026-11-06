// Pronunciation from the CMU Pronouncing Dictionary (site/voice/cmudict.txt, ARPAbet): the phonemes Kokoro reads aloud,
// IPA to show a learner, and what to do with the mouth for the sounds learners of English find hard.

const VOWELS = new Set(['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']);
// Kokoro v1.0 was trained on misaki's American phonemes: one letter for each diphthong (A = eɪ, I = aɪ, O = oʊ, W = aʊ, Y = ɔɪ).
const KOKORO = {
  AA: 'ɑ', AE: 'æ', AH: 'ʌ', AO: 'ɔ', AW: 'W', AY: 'I', EH: 'ɛ', ER: 'ɜɹ', EY: 'A', IH: 'ɪ', IY: 'i', OW: 'O', OY: 'Y', UH: 'ʊ', UW: 'u',
  B: 'b', CH: 'ʧ', D: 'd', DH: 'ð', F: 'f', G: 'ɡ', HH: 'h', JH: 'ʤ', K: 'k', L: 'l', M: 'm', N: 'n', NG: 'ŋ', P: 'p', R: 'ɹ',
  S: 's', SH: 'ʃ', T: 't', TH: 'θ', V: 'v', W: 'w', Y: 'j', Z: 'z', ZH: 'ʒ',
};
const IPA = {
  ...KOKORO, AA: 'ɑː', AO: 'ɔː', AW: 'aʊ', AY: 'aɪ', ER: 'ɝ', EY: 'eɪ', IY: 'iː', OW: 'oʊ', OY: 'ɔɪ', UW: 'uː', CH: 'tʃ', JH: 'dʒ',
};
const ONSETS = new Set(['P R', 'P L', 'B R', 'B L', 'T R', 'D R', 'K R', 'K L', 'K W', 'G R', 'G L', 'F R', 'F L', 'TH R', 'S T', 'S P', 'S K',
  'S L', 'S M', 'S N', 'S W', 'SH R', 'T W', 'D W', 'S T R', 'S P R', 'S K R', 'S P L', 'S K W', 'P Y', 'B Y', 'K Y', 'F Y', 'M Y', 'H Y', 'V Y']);

const vowel = phone => VOWELS.has(phone.replace(/\d$/, ''));
const bare = phone => phone.replace(/\d$/, '');

export function parseDictionary(text) {
  const words = new Map();
  for (const line of text.split('\n')) {
    const space = line.indexOf(' ');
    if (space > 0) words.set(line.slice(0, space), line.slice(space + 1).split(' '));
  }
  return words;
}

const ONES = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen',
  'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen'];
const TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'];

export function numberWords(n) {
  if (n < 20) return ONES[n];
  if (n < 100) return TENS[Math.floor(n / 10)] + (n % 10 ? '-' + ONES[n % 10] : '');
  if (n < 1000) return ONES[Math.floor(n / 100)] + ' hundred' + (n % 100 ? ' ' + numberWords(n % 100) : '');
  for (const [size, name] of [[1e9, 'billion'], [1e6, 'million'], [1e3, 'thousand']]) {
    if (n >= size) return numberWords(Math.floor(n / size)) + ' ' + name + (n % size ? ' ' + numberWords(n % size) : '');
  }
  return String(n);
}

const year = n => n % 100 === 0 ? numberWords(n / 100) + ' hundred' : numberWords(Math.floor(n / 100)) + ' ' + (n % 100 < 10 ? 'oh ' + ONES[n % 100] : numberWords(n % 100));

// Text as it should be read aloud: no markdown or code, numbers and symbols as words.
export function speakable(text) {
  return text
    .replace(/```[\s\S]*?(```|$)/g, ' (code on screen) ')
    .replace(/`([^`]*)`/g, '$1')
    .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/https?:\/\/\S+/g, 'a link')
    .replace(/[*_#>]+/g, '').replace(/\|/g, ' ')
    .replace(/\$(\d+(?:\.\d\d)?)/g, (_, amount) => amount + ' dollars')
    .replace(/(\d+(?:\.\d+)?)\s?%/g, '$1 percent')
    .replace(/&/g, ' and ')
    .replace(/\b(1[1-9]\d\d|20\d\d)\b/g, (_, y) => year(Number(y)))
    .replace(/\b\d{1,3}(,\d{3})+\b/g, n => n.replace(/,/g, ''))
    .replace(/\b(\d+)\.(\d+)\b/g, (_, whole, part) => numberWords(Number(whole)) + ' point ' + [...part].map(d => ONES[Number(d)]).join(' '))
    .replace(/\b(\d+)(st|nd|rd|th)\b/gi, (_, n) => ordinal(Number(n)))
    .replace(/\b\d+\b/g, n => n.length > 12 ? [...n].map(d => ONES[Number(d)]).join(' ') : numberWords(Number(n)))
    .replace(/[–—]/g, ', ')
    .replace(/[“”]/g, '"').replace(/[‘’]/g, "'")
    .replace(/\s+/g, ' ')
    .trim();
}

function ordinal(n) {
  const words = numberWords(n);
  const irregular = { one: 'first', two: 'second', three: 'third', five: 'fifth', eight: 'eighth', nine: 'ninth', twelve: 'twelfth' };
  return words.replace(/(\w+)$/, last => irregular[last] || (last.endsWith('y') ? last.slice(0, -1) + 'ieth' : last + 'th'));
}

const SUFFIXES = [['ing', ['IH0', 'NG']], ['ed', ['D']], ['es', ['IH0', 'Z']], ['s', ['Z']], ['er', ['ER0']], ['ly', ['L', 'IY0']], ['ness', ['N', 'AH0', 'S']]];
// Names the dictionary does not have.
const NAMES = { kani: 'K AA1 N IY0', kelus: 'K EH1 L AH0 S', cide: 'S AY1 D', otter: 'AA1 T ER0', webgpu: 'W EH1 B JH IY1 P IY1 Y UW1' };
const LETTER_NAMES = { a: 'EY1', b: 'B IY1', c: 'S IY1', d: 'D IY1', e: 'IY1', f: 'EH1 F', g: 'JH IY1', h: 'EY1 CH', i: 'AY1', j: 'JH EY1',
  k: 'K EY1', l: 'EH1 L', m: 'EH1 M', n: 'EH1 N', o: 'OW1', p: 'P IY1', q: 'K Y UW1', r: 'AA1 R', s: 'EH1 S', t: 'T IY1', u: 'Y UW1', v: 'V IY1',
  w: 'D AH1 B AH0 L Y UW0', x: 'EH1 K S', y: 'W AY1', z: 'Z IY1' };
const ACRONYMS = new Set(['ai', 'api', 'usa', 'uk', 'pdf', 'cpu', 'gpu', 'html', 'css', 'url', 'tv', 'faq', 'diy']);
// Letter groups to phones, longest first, for words nobody listed: good enough to sound a name out.
const LETTERS = [['tch', 'CH'], ['igh', 'AY1'], ['ough', 'AO1'], ['tion', 'SH AH0 N'], ['sion', 'ZH AH0 N'], ['ch', 'CH'], ['sh', 'SH'], ['th', 'TH'],
  ['ph', 'F'], ['ck', 'K'], ['ng', 'NG'], ['qu', 'K W'], ['wh', 'W'], ['ee', 'IY1'], ['ea', 'IY1'], ['oo', 'UW1'], ['ou', 'AW1'], ['ow', 'OW1'],
  ['ai', 'EY1'], ['ay', 'EY1'], ['oa', 'OW1'], ['oi', 'OY1'], ['oy', 'OY1'], ['au', 'AO1'], ['aw', 'AO1'], ['ew', 'UW1'], ['ie', 'IY1'],
  ['ar', 'AA1 R'], ['or', 'AO1 R'], ['er', 'ER0'], ['ir', 'ER1'], ['ur', 'ER1'], ['a', 'AE1'], ['e', 'EH1'], ['i', 'IH1'], ['o', 'AA1'], ['u', 'AH1'],
  ['y', 'IY0'], ['b', 'B'], ['c', 'K'], ['d', 'D'], ['f', 'F'], ['g', 'G'], ['h', 'HH'], ['j', 'JH'], ['k', 'K'], ['l', 'L'], ['m', 'M'],
  ['n', 'N'], ['p', 'P'], ['r', 'R'], ['s', 'S'], ['t', 'T'], ['v', 'V'], ['w', 'W'], ['x', 'K S'], ['z', 'Z']];

function soundOut(word) {
  const phones = [];
  let text = word.replace(/([^aeiou])e$/, '$1').replace(/(.)\1/g, '$1'), stressed = false;
  while (text) {
    const soft = /^c[eiy]/.test(text) ? ['c', 'S'] : /^g[eiy]/.test(text) && text.length > 2 ? ['g', 'JH'] : null;
    const [letters, sound] = soft || LETTERS.find(([l]) => text.startsWith(l)) || [text[0], ''];
    for (let phone of sound.split(' ').filter(Boolean)) {
      // Only the first vowel is stressed; later ones are reduced.
      if (/1$/.test(phone)) { if (stressed) phone = phone.replace('1', '0'); stressed = true; }
      phones.push(phone.replace(/^(AE|EH|AA|AH)0$/, 'AH0').replace(/^IH0$/, 'IY0'));
    }
    text = text.slice(letters.length);
  }
  return phones;
}

// ARPAbet phones for one word: names, then the dictionary (short capitals are spelled: AI, PDF), then a known word plus an
// ending, then two known words, then sounded out from its letters.
export function phonesFor(word, words) {
  const w = word.toLowerCase().replace(/[’]/g, "'"), clean = w.replace(/[^a-z']/g, '');
  if (!clean) return [];
  if (Object.hasOwn(NAMES, clean)) return NAMES[clean].split(' ');
  if (ACRONYMS.has(clean) || (/^[A-Z]{2,5}$/.test(word) && !words.has(clean))) {
    return [...clean].flatMap(letter => LETTER_NAMES[letter]?.split(' ') || []);
  }
  if (words.has(w)) return words.get(w);
  if (words.has(clean)) return words.get(clean);
  for (const [ending, phones] of SUFFIXES) {
    const stem = clean.slice(0, -ending.length);
    if (clean.endsWith(ending) && stem.length > 2) {
      for (const root of [stem, stem + 'e', stem.replace(/(.)\1$/, '$1'), stem.replace(/i$/, 'y')]) {
        if (words.has(root)) return [...words.get(root), ...phones];
      }
    }
  }
  for (let cut = clean.length - 3; cut >= 3; cut--) {
    const a = clean.slice(0, cut), b = clean.slice(cut);
    if (words.has(a) && words.has(b)) return [...words.get(a), ...words.get(b)];
  }
  return soundOut(clean.replace(/'/g, ''));
}

// The US flap: t between a vowel (or r) and an unstressed vowel sounds like a quick d ("water", "better").
const flapped = (phones, i) => phones[i] === 'T' && i > 0 && i + 1 < phones.length
  && (vowel(phones[i - 1]) || phones[i - 1] === 'R') && vowel(phones[i + 1]) && phones[i + 1].endsWith('0');

function wordToKokoro(phones) {
  return phones.map((phone, i) => {
    if (flapped(phones, i)) return 'ɾ';
    const base = bare(phone), stress = phone.endsWith('1') ? 'ˈ' : phone.endsWith('2') ? 'ˌ' : '';
    if (base === 'AH' && phone.endsWith('0')) return 'ə';
    if (base === 'ER' && phone.endsWith('0')) return 'ɚ';
    return stress + KOKORO[base];
  }).join('');
}

// Text to Kokoro's phoneme string: words become phonemes, punctuation Kokoro knows stays.
export function toKokoro(text, words) {
  return speakable(text)
    .replace(/[^A-Za-z0-9'’\s;:,.!?…"()-]/g, ' ')
    .split(/(\s+|[;:,.!?…"()]+)/)
    .map(part => /^[A-Za-z0-9'’-]+$/.test(part) ? part.split('-').filter(Boolean).map(w => wordToKokoro(phonesFor(w, words))).join(' ') : part.replace(/\s+/g, ' '))
    .join('')
    .replace(/\s+/g, ' ')
    .trim();
}

// IPA the way dictionaries print it, with the stress mark before the stressed syllable: think → /θɪŋk/, water → /ˈwɔːtɚ/.
export function ipa(phones) {
  const marks = new Map();
  if (phones.filter(vowel).length > 1) {
    let previous = -1;
    phones.forEach((phone, i) => {
      if (!vowel(phone)) return;
      if (/[12]$/.test(phone)) {
        // The syllable starts at the consonants before its vowel that can begin an English syllable (im-PORT, ex-TRA).
        let start = i;
        if (previous < 0) start = 0;
        else if (i - previous > 1) {
          start = i - 1;
          for (let s = previous + 1; s < i - 1; s++) if (ONSETS.has(phones.slice(s, i).map(bare).join(' '))) { start = s; break; }
        }
        marks.set(start, phone.endsWith('1') ? 'ˈ' : 'ˌ');
      }
      previous = i;
    });
  }
  return '/' + phones.map((phone, i) => {
    const base = bare(phone);
    return (marks.get(i) || '') + (base === 'AH' && phone.endsWith('0') ? 'ə' : base === 'ER' && phone.endsWith('0') ? 'ɚ' : IPA[base]);
  }).join('') + '/';
}

// What to do with the mouth, for the sounds learners of English find hard.
export const SOUND_TIPS = {
  TH: { sound: 'th (as in think)', tip: 'Put the tip of your tongue lightly between your teeth and blow air out, with no voice. It is not s, t or f.' },
  DH: { sound: 'th (as in this)', tip: 'Tongue tip between your teeth, like for "think", but turn your voice on so it buzzes. It is not d or z.' },
  R: { sound: 'r', tip: 'Curl your tongue back a little without touching the roof of your mouth, and round your lips slightly. Do not tap or roll it.' },
  L: { sound: 'l', tip: 'Press the tip of your tongue on the ridge just behind your top teeth and let air flow around the sides.' },
  V: { sound: 'v', tip: 'Rest your top teeth on your bottom lip and make your voice buzz. Your lips should not touch each other (that is b).' },
  W: { sound: 'w', tip: 'Round your lips into a small circle, then open them quickly. Your teeth do not touch your lip (that is v).' },
  F: { sound: 'f', tip: 'Top teeth on your bottom lip, then blow air with no voice. Not p: your lips stay apart.' },
  Z: { sound: 'z', tip: 'Like s, but turn your voice on so it buzzes. Put a hand on your throat: you should feel it.' },
  S: { sound: 's', tip: 'Tongue tip behind your top teeth, teeth almost closed, and a thin stream of air. Not sh: your lips are not rounded.' },
  SH: { sound: 'sh', tip: 'Round your lips forward and pull your tongue a little back from s. Air should sound soft and wide.' },
  ZH: { sound: 'zh (as in measure)', tip: 'Like sh, but with your voice on.' },
  CH: { sound: 'ch', tip: 'Start with t, then release it straight into sh: t-sh, quickly, as one sound.' },
  JH: { sound: 'j (as in job)', tip: 'Start with d, then release into the buzzing sound in "measure": d-zh, quickly, as one sound.' },
  NG: { sound: 'ng (as in sing)', tip: 'Lift the back of your tongue to the soft roof of your mouth and hum through your nose. Do not add a hard g after it.' },
  HH: { sound: 'h', tip: 'Just breathe out from your throat, like fogging a mirror. Do not scrape the back of your mouth.' },
  IH: { sound: 'short i (as in sit)', tip: 'Short and relaxed: smile less and keep your tongue a little lower than for "seat".' },
  IY: { sound: 'long ee (as in seat)', tip: 'Smile, keep your tongue high and forward, and make the sound longer than the i in "sit".' },
  AE: { sound: 'a (as in cat)', tip: 'Open your mouth wide and push your tongue forward and low. It is between "e" in bed and "a" in father.' },
  EH: { sound: 'e (as in bed)', tip: 'Mouth half open and relaxed, tongue in the middle. Shorter than the a in "cake".' },
  AH: { sound: 'u (as in cup)', tip: 'A short, relaxed "uh" from the middle of your mouth. Do not round your lips.' },
  AA: { sound: 'o (as in hot)', tip: 'Open your mouth wide and relax your tongue low, like when a doctor checks your throat.' },
  AO: { sound: 'aw (as in law)', tip: 'Open your mouth and round your lips a little, a bit deeper than "o" in hot.' },
  UH: { sound: 'oo (as in book)', tip: 'Short and relaxed, lips only a little rounded. Shorter than the oo in "food".' },
  UW: { sound: 'oo (as in food)', tip: 'Round your lips into a tight circle and make it long.' },
  ER: { sound: 'er (as in bird)', tip: 'Keep your tongue pulled back and curled a little for the whole vowel, like a long r.' },
  EY: { sound: 'ay (as in cake)', tip: 'Two parts: start at "e" and glide up to "ee". Do not cut it short.' },
  OW: { sound: 'oh (as in go)', tip: 'Two parts: start at "o" and glide into "oo" by rounding your lips.' },
  AY: { sound: 'eye (as in time)', tip: 'Two parts: open "ah" gliding up to "ee".' },
  AW: { sound: 'ow (as in now)', tip: 'Two parts: open "ah" gliding to "oo" with rounded lips.' },
  OY: { sound: 'oy (as in boy)', tip: 'Two parts: rounded "aw" gliding to "ee".' },
  B: { sound: 'b', tip: 'Press your lips together and release them with your voice on.' },
  P: { sound: 'p', tip: 'Lips together, then release with a small puff of air and no voice.' },
  D: { sound: 'd', tip: 'Tongue tip on the ridge behind your top teeth, release with your voice on.' },
  T: { sound: 't', tip: 'Tongue tip on the ridge behind your top teeth, release with a little puff of air.' },
  G: { sound: 'g', tip: 'Back of your tongue against the soft roof of your mouth, release with your voice on.' },
  K: { sound: 'k', tip: 'Back of your tongue against the soft roof of your mouth, release with a puff of air.' },
  M: { sound: 'm', tip: 'Close your lips and hum.' },
  N: { sound: 'n', tip: 'Tongue tip on the ridge behind your top teeth and hum through your nose.' },
  Y: { sound: 'y (as in yes)', tip: 'Start from "ee" and glide quickly into the next vowel.' },
};

// Line up two phone lists (edit distance) and return the sounds that differ: what was wanted and what was heard instead.
export function soundDifferences(wanted, heard) {
  const a = wanted.map(bare), b = heard.map(bare);
  const cost = Array.from({ length: a.length + 1 }, (_, i) => [i, ...Array(b.length).fill(0)]);
  for (let j = 1; j <= b.length; j++) cost[0][j] = j;
  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      cost[i][j] = Math.min(cost[i - 1][j] + 1, cost[i][j - 1] + 1, cost[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    }
  }
  const differences = [];
  for (let i = a.length, j = b.length; i > 0 || j > 0;) {
    if (i > 0 && j > 0 && cost[i][j] === cost[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1)) {
      if (a[i - 1] !== b[j - 1]) differences.unshift({ wanted: a[i - 1], heard: b[j - 1] });
      i--; j--;
    } else if (i > 0 && cost[i][j] === cost[i - 1][j] + 1) {
      differences.unshift({ wanted: a[i - 1], heard: null }); i--;
    } else {
      differences.unshift({ wanted: null, heard: b[j - 1] }); j--;
    }
  }
  return differences;
}

export const soundName = phone => (Object.hasOwn(SOUND_TIPS, phone) && SOUND_TIPS[phone].sound) || IPA[phone] || phone.toLowerCase();
