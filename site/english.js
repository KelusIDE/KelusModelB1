// Kani's English tutor: the grammar mistakes learners make most, found with fixed rules (so a correction is never made
// up), each with a fix and a short reason; better words for a plain sentence; and feedback on a sentence read aloud
// (from site/voice/whisper.js's transcript and scores, with site/voice/pronounce.js's sound tips).
import { SOUND_TIPS, ipa, phonesFor, soundDifferences, soundName } from './voice/pronounce.js';

// Verbs: [base, he/she/it form, past, past participle]. Regular verbs only need their base.
const IRREGULAR = `be is was been|have has had had|do does did done|go goes went gone|say says said said|make makes made made|get gets got gotten
know knows knew known|think thinks thought thought|take takes took taken|see sees saw seen|come comes came come|give gives gave given
find finds found found|tell tells told told|become becomes became become|leave leaves left left|feel feels felt felt|bring brings brought brought
begin begins began begun|keep keeps kept kept|hold holds held held|write writes wrote written|stand stands stood stood|hear hears heard heard
mean means meant meant|meet meets met met|run runs ran run|pay pays paid paid|sit sits sat sat|speak speaks spoke spoken|read reads read read
grow grows grew grown|lose loses lost lost|fall falls fell fallen|send sends sent sent|build builds built built|understand understands understood understood
draw draws drew drawn|break breaks broke broken|spend spends spent spent|drive drives drove driven|buy buys bought bought|wear wears wore worn
choose chooses chose chosen|eat eats ate eaten|drink drinks drank drunk|sing sings sang sung|swim swims swam swum|sleep sleeps slept slept
teach teaches taught taught|catch catches caught caught|fight fights fought fought|fly flies flew flown|forget forgets forgot forgotten
win wins won won|sell sells sold sold|throw throws threw thrown|wake wakes woke woken|ride rides rode ridden|steal steals stole stolen
feed feeds fed fed|put puts put put|cut cuts cut cut|hurt hurts hurt hurt|cost costs cost cost|forgive forgives forgave forgiven
freeze freezes froze frozen|hide hides hid hidden|lend lends lent lent|bite bites bit bitten|blow blows blew blown|shake shakes shook shaken
show shows showed shown|learn learns learned learned|let lets let let|set sets set set`;
const REGULAR = `like love want need play work live study watch walk talk help start finish use call try ask look listen open close cook clean
wash visit travel dance jump move change stay stop plan enjoy hate hope wait answer arrive believe decide explain happen improve invite join
laugh miss order paint pass practice prefer prepare promise rain remember rest return share shop smile snow thank touch turn agree follow
carry cry worry marry reply check climb collect count cross deliver describe drop enter fix guess kick knock lock mix notice offer pick
plant pray pull push rent repeat save seem smell suggest support taste type wish chat add allow borrow celebrate complain cycle download
exercise hurry imagine jog post relax rush search solve shout surf upload vote need cook listen`;
const DOUBLE = new Set(['stop', 'plan', 'shop', 'drop', 'jog', 'chat', 'hug', 'rob', 'beg', 'nod', 'skip', 'slip', 'trip', 'grab']);

const VERBS = new Map(); // any form -> { base, s, past, participle, ing, form }
const FORMS = [];
function addVerb(base, s, past, participle) {
  const ing = base === 'be' ? 'being' : base.endsWith('ie') ? base.slice(0, -2) + 'ying'
    : base.endsWith('e') && !/(ee|ye|oe)$/.test(base) ? base.slice(0, -1) + 'ing'
      : DOUBLE.has(base) || ['run', 'swim', 'sit', 'get', 'put', 'cut', 'begin', 'win', 'hit'].includes(base) ? base + base.at(-1) + 'ing' : base + 'ing';
  const verb = { base, s, past, participle, ing };
  FORMS.push(verb);
  for (const [form, name] of [[participle, 'participle'], [past, 'past'], [s, 's'], [ing, 'ing'], [base, 'base']]) VERBS.set(form, { ...verb, form: name });
}
for (const row of IRREGULAR.split(/[|\n]/)) addVerb(...row.trim().split(' '));
for (const base of new Set(REGULAR.split(/\s+/))) {
  const s = /(s|sh|ch|x|z|o)$/.test(base) ? base + 'es' : /[^aeiou]y$/.test(base) ? base.slice(0, -1) + 'ies' : base + 's';
  const past = base.endsWith('e') ? base + 'd' : /[^aeiou]y$/.test(base) ? base.slice(0, -1) + 'ied' : DOUBLE.has(base) ? base + base.at(-1) + 'ed' : base + 'ed';
  if (!VERBS.has(base)) addVerb(base, s, past, past);
}
const verb = word => VERBS.get(word);
// Irregular verbs given a regular -ed ("buyed"): the real past.
const OVER_REGULAR = new Map(FORMS.filter(v => v.past !== v.base + 'ed' && v.past !== v.base + 'd' && !/[^aeiou]y$/.test(v.base) && v.base !== 'be' && !DOUBLE.has(v.base))
  .flatMap(v => [[v.base.replace(/e$/, '') + 'ed', v.past], [v.base + v.base.at(-1) + 'ed', v.past]])
  .filter(([wrong]) => !VERBS.has(wrong) && !['fed', 'bed', 'red', 'led', 'wed', 'sped', 'shed', 'seed', 'need', 'feed', 'speed', 'reed', 'leaved', 'winged'].includes(wrong)));
OVER_REGULAR.set('payed', 'paid'); OVER_REGULAR.set('sayed', 'said');

const SINGULAR = new Set(['he', 'she', 'it']);
const PLURAL = new Set(['i', 'you', 'we', 'they']);
const MODALS = new Set(['can', 'could', 'will', 'would', 'should', 'must', 'may', 'might', 'shall', "can't", 'cannot', "won't", "wouldn't",
  "shouldn't", "couldn't", "mustn't"]);
const DO = new Set(['do', 'does', 'did', "don't", "doesn't", "didn't"]);
const ADVERBS = new Set(['never', 'always', 'often', 'usually', 'sometimes', 'really', 'also', 'just', 'still', 'only', 'rarely', 'seldom', 'even', 'not', 'already', 'ever']);
const CAUSES = new Set(['make', 'makes', 'made', 'let', 'lets', 'help', 'helps', 'helped', 'watch', 'watched', 'see', 'saw', 'hear', 'heard', 'have', 'had', 'to']);
const CONJUNCTIONS = new Set(['and', 'but', 'so', 'because', 'when', 'if', 'or', 'while', 'although', 'though', 'since', 'until', 'before', 'after', 'that', 'then']);
const PREPOSITIONS = new Set(['of', 'with', 'between', 'for', 'to', 'from', 'among', 'about', 'by', 'and', 'other', 'like']);
// Verbs after which "he go" is right ("they asked that he go", "I suggest she wait").
const ASKS = new Set(['suggest', 'suggested', 'demand', 'demanded', 'insist', 'insisted', 'recommend', 'recommended', 'require', 'required',
  'request', 'requested', 'propose', 'proposed', 'letting', 'making', 'helping', 'watching', 'watches', 'sees', 'hears', 'lets']);
const BE = new Set(['is', 'was', 'are', 'were', 'am', "isn't", "wasn't", "aren't", "weren't"]);
// Words after "a"/"many" that show they are not an article or a quantity: Town A and Town B, if A is true.
const FUNCTION_WORDS = new Set(['is', 'are', 'was', 'were', 'and', 'or', 'but', 'as', 'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'has',
  'have', 'had', 'will', 'would', 'can', 'could', 'be', 'that', 'because', 'so', 'then', 'than', 'from', 'into', 'if']);
const TIME_WORDS = new Set(['yesterday', 'today', 'tonight', 'tomorrow', 'now', 'recently', 'finally', 'later', 'usually', 'sometimes', 'often',
  'always', 'ago', 'week', 'night', 'month', 'year', 'weekend', 'morning', 'evening', 'afternoon']);
const JOINS = new Set(['and', 'but', 'so', 'because', 'although', 'though', 'while', 'or']);
const POSSESSIVES = new Set(['my', 'your', 'his', 'her', 'our', 'their']);
// Words before "can", "will", "may"... that make them nouns or months: a can of, his will, in May.
const NOT_HELPER_BEFORE = new Set(['a', 'the', 'his', 'her', 'my', 'your', 'their', 'our', 'in', 'on', 'of', 'since', 'until', 'by', 'during', 'from', 'free', 'last', 'this', 'next']);
const HAVE = new Set(['have', 'has', 'had', "haven't", "hasn't", "hadn't", "i've", "you've", "we've", "they've", "he's", "she's"]);
// Verbs whose past looks like their base (he read it yesterday), so the tense cannot be checked.
const SAME_PAST = new Set(['read', 'put', 'cut', 'let', 'set', 'hurt', 'cost']);
// Verbs about the present even in a sentence about the past (I think he left yesterday).
const REPORTING = new Set(['think', 'know', 'believe', 'guess', 'hope', 'remember', 'want', 'need', 'like', 'love', 'mean', 'wonder', 'forget',
  'understand', 'hate', 'prefer', 'wish', 'feel', 'miss', 'seem', 'agree', 'worry', 'imagine', 'notice']);
const FUTURE = /\b(will|going to|tomorrow|next|'ll)\b/i;
const TO_PAST = new Set(['went', 'ate', 'came', 'took', 'gave', 'wrote', 'spoke', 'drove', 'drank', 'swam', 'began', 'did', 'made', 'said', 'got',
  'bought', 'brought', 'thought', 'taught', 'caught', 'told', 'sold', 'knew', 'flew', 'threw', 'chose', 'broke', 'forgot', 'woke', 'wore']);
const WANT_BEFORE = new Set(["don't", "doesn't", "didn't", 'would', "i'd", "you'd", "we'd", "they'd", "he'd", "she'd", 'really']);
const FAMILY = new Set(['mom', 'dad', 'mother', 'father', 'brother', 'sister', 'friend', 'teacher', 'son', 'daughter', 'wife', 'husband', 'boss',
  'cat', 'dog', 'baby', 'grandma', 'grandpa', 'grandmother', 'grandfather', 'aunt', 'uncle', 'cousin', 'boyfriend', 'girlfriend', 'neighbor']);
const UNCOUNTABLE = new Set(['water', 'money', 'time', 'information', 'advice', 'homework', 'furniture', 'milk', 'sugar', 'rice', 'bread', 'music',
  'traffic', 'news', 'luggage', 'coffee', 'tea', 'juice', 'snow', 'rain', 'help', 'work', 'food', 'knowledge', 'equipment', 'electricity']);
const COUNTABLE_PLURALS = new Set(['people', 'friends', 'books', 'things', 'cars', 'children', 'students', 'times', 'days', 'years', 'questions',
  'problems', 'ideas', 'places', 'apples', 'games', 'movies', 'songs', 'words', 'countries', 'cities', 'hours', 'minutes', 'animals', 'dogs', 'cats']);
const NO_PLURAL = { informations: 'information', advices: 'advice', furnitures: 'furniture', homeworks: 'homework', equipments: 'equipment',
  luggages: 'luggage', knowledges: 'knowledge', feedbacks: 'feedback', softwares: 'software', evidences: 'evidence', traffics: 'traffic',
  moneys: 'money', musics: 'music', advertisings: 'advertising' };
const COMPARATIVES = new Set(`better worse bigger smaller easier harder faster slower older younger taller shorter higher lower cheaper happier
nicer larger longer stronger weaker warmer colder hotter richer poorer prettier busier funnier earlier later closer safer simpler smarter cleaner
darker louder quieter heavier lighter newer greater wider deeper thicker thinner softer sweeter healthier friendlier cooler braver calmer kinder
lazier luckier`.split(/\s+/));
const SUPERLATIVES = new Set([...COMPARATIVES].map(c => c === 'better' ? 'best' : c === 'worse' ? 'worst' : c.slice(0, -2) + 'est'));
const JOBS = new Set(['student', 'teacher', 'doctor', 'nurse', 'engineer', 'programmer', 'developer', 'lawyer', 'designer', 'artist', 'singer',
  'writer', 'driver', 'farmer', 'chef', 'waiter', 'waitress', 'manager', 'pilot', 'scientist', 'dentist', 'accountant', 'actor', 'musician',
  'photographer', 'journalist', 'mechanic', 'cashier', 'soldier', 'employee', 'beginner', 'nurse', 'housewife', 'architect', 'electrician',
  'plumber', 'secretary', 'librarian', 'baker', 'tourist', 'freelancer', 'volunteer', 'researcher', 'professor', 'police officer', 'vet']);
const TO_VERBS = new Set(['go', 'come', 'eat', 'buy', 'see', 'meet', 'learn', 'know', 'become', 'find', 'get', 'make', 'take', 'give', 'tell',
  'speak', 'write', 'read', 'watch', 'be', 'have', 'do', 'say', 'leave', 'stay', 'study', 'improve', 'understand', 'swim', 'drive', 'ride',
  'sing', 'travel', 'live', 'ask', 'sell', 'start', 'finish', 'open', 'close', 'explain', 'visit', 'play']);
const WANTS = new Set(['want', 'wants', 'wanted', 'need', 'needs', 'needed', 'hope', 'hopes', 'hoped', 'decide', 'decided', 'plan', 'plans',
  'planned', 'try', 'tries', 'tried', 'forget', 'forgot', 'like', 'likes', 'liked', 'love', 'loves', 'loved', 'hate', 'hates', 'prefer', 'learn', 'learned']);
const PAST_MARKERS = /\b(yesterday|last (night|week|month|year|weekend|summer|winter|spring|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|\w+ ago)\b/i;
const UPGRADES = { good: 'great, excellent', bad: 'terrible, awful', big: 'huge, enormous', small: 'tiny', tired: 'exhausted', happy: 'delighted, thrilled',
  hungry: 'starving', cold: 'freezing', hot: 'boiling', angry: 'furious', scared: 'terrified', interesting: 'fascinating', pretty: 'gorgeous',
  funny: 'hilarious', sad: 'heartbroken, miserable', important: 'essential, crucial', clean: 'spotless', dirty: 'filthy', easy: 'effortless',
  difficult: 'challenging, tough', beautiful: 'stunning', nice: 'lovely, delightful', fast: 'speedy, rapid', quiet: 'silent', loud: 'deafening',
  busy: 'swamped', boring: 'dull, tedious', smart: 'brilliant', surprised: 'amazed, astonished', old: 'ancient' };

const VOWEL_SOUND_EXCEPTIONS = { hour: 'an', honest: 'an', honour: 'an', honor: 'an', heir: 'an', university: 'a', unit: 'a', user: 'a',
  use: 'a', usual: 'a', unique: 'a', uniform: 'a', one: 'a', once: 'a', european: 'a', useful: 'a', union: 'a', unicorn: 'a', ufo: 'a', uber: 'an' };

function article(word, words) {
  const lower = word.toLowerCase();
  if (Object.hasOwn(VOWEL_SOUND_EXCEPTIONS, lower)) return VOWEL_SOUND_EXCEPTIONS[lower];
  // Short capitals are spelled (an LED, an FBI agent); longer ones with a vowel are said as words (a SWOT analysis).
  if (/^[A-Z]{2,3}$/.test(word) || /^[A-Z]{2,}/.test(word) && !/^[A-Z]*[AEIOU]/.test(word)) return /^[AEFHILMNORSX]/.test(word) ? 'an' : 'a';
  if (words) return /^(AA|AE|AH|AO|AW|AY|EH|ER|EY|IH|IY|OW|OY|UH|UW)/.test(phonesFor(lower, words)[0] || '') ? 'an' : 'a';
  return /^[aeio]/.test(lower) || /^un/.test(lower) && !/^uni/.test(lower) || /^u[^aeiou]{2}/.test(lower) ? 'an' : 'a';
}

const capital = (like, word) => like[0] === like[0].toUpperCase() && like[0] !== like[0].toLowerCase() ? word[0].toUpperCase() + word.slice(1) : word;

// Words with where they are in the text.
function tokenize(text) {
  return [...text.matchAll(/[A-Za-z]+(?:['’][A-Za-z]+)?|\d+|[^\sA-Za-z\d]/g)].map(m => ({ text: m[0], lower: m[0].toLowerCase().replace('’', "'"), start: m.index, end: m.index + m[0].length }));
}

// Every mistake as { start, end, wrong, right, why, rule } in the text's characters.
export function checkGrammar(text, words) {
  const tokens = tokenize(text), found = [];
  const at = i => tokens[i]?.lower ?? '';
  const word = i => /^[a-z']+$/.test(at(i));
  const fix = (from, to, right, why, rule) => {
    const start = tokens[from].start, end = tokens[to].end;
    if (found.some(f => f.start < end && start < f.end)) return;
    found.push({ start, end, wrong: text.slice(start, end), right: capital(text.slice(start, end), right), why, rule });
  };
  const replace = (i, right, why, rule) => fix(i, i, right, why, rule);
  const skipAdverbs = i => { while (ADVERBS.has(at(i)) && at(i) !== 'not') i++; return i; };
  // The clause around a word: up to the sentence's end or a joining word (I saw him last week | and he looks happy).
  const clauseOf = i => {
    const edge = j => /^[.!?;]$/.test(at(j)) || JOINS.has(at(j));
    let s = i, e = i;
    while (s > 0 && !edge(s - 1)) s--;
    while (e < tokens.length - 1 && !edge(e + 1)) e++;
    return text.slice(tokens[s].start, tokens[e].end);
  };
  // "it" and "you" are only subjects at the start of a clause ("make it work", "I sent you plans").
  const clauseStart = i => i === 0 || /^[.!?,;:]$/.test(at(i - 1)) || CONJUNCTIONS.has(at(i - 1)) || TIME_WORDS.has(at(i - 1));
  const subjectSingular = i => (SINGULAR.has(at(i)) && (at(i) !== 'it' || clauseStart(i)) && !BE.has(at(i - 1)))
    || (FAMILY.has(at(i)) && POSSESSIVES.has(at(i - 1)) && !CAUSES.has(at(i - 2)) && !DO.has(at(i - 2)) && !ASKS.has(at(i - 2)) && at(i - 2) !== 'that'
      && !(['and', 'or'].includes(at(i - 2)) && (SINGULAR.has(at(i - 3)) || PLURAL.has(at(i - 3)) || FAMILY.has(at(i - 3)) || /^[A-Z]/.test(tokens[i - 3]?.text || ''))));
  const subjectPlural = i => PLURAL.has(at(i)) && (!['you', 'i'].includes(at(i)) || clauseStart(i)) && !BE.has(at(i - 1));

  for (let i = 0; i < tokens.length; i++) {
    const t = at(i), next = at(i + 1), previous = at(i - 1);
    // A word on its own, not a variable: V_i, 7i, i7, i = 1, 1 - i, `i`.
    const alone = !/[\w(\[{=+\-*/^`]\s*$/.test(text.slice(Math.max(0, tokens[i].start - 2), tokens[i].start))
      && !/^([\w)\]`]|\s*[=+\-*/^<>])/.test(text.slice(tokens[i].end, tokens[i].end + 3));
    if (/^i('(m|ve|ll|d))?$/.test(tokens[i].text) && alone && !(next === '.' && at(i + 2) === 'e')) {
      replace(i, 'I' + tokens[i].text.slice(1), 'The word "I" is always a capital letter.', 'capital-i');
    }

    if ((t === 'a' || t === 'an') && word(i + 1) && previous !== 'plan' && !FUNCTION_WORDS.has(next) && !/^[-+*/=^]$/.test(previous)) {
      const right = article(tokens[i + 1].text, words);
      if (right !== t) replace(i, right, right === 'an' ? `Use "an" before a vowel sound: an ${tokens[i + 1].text}.` : `Use "a" before a consonant sound: a ${tokens[i + 1].text}.`, 'a-an');
    }

    // be, do and have with their subject.
    if (t === 'i' && clauseStart(i) && (next === 'is' || next === 'are')) fix(i, i + 1, tokens[i].text + ' am', 'With "I", use "am": I am.', 'be-agreement');
    if (SINGULAR.has(t) && (next === 'are' || next === 'am')) fix(i, i + 1, tokens[i].text + ' is', `With "${t}", use "is".`, 'be-agreement');
    if ((t === 'we' || t === 'they' || t === 'you' && clauseStart(i)) && (next === 'is' || next === 'am')) fix(i, i + 1, tokens[i].text + ' are', `With "${t}", use "are".`, 'be-agreement');
    if (['we', 'they'].includes(t) && next === 'was' || t === 'you' && next === 'was' && clauseStart(i)) fix(i, i + 1, tokens[i].text + ' were', `With "${t}", the past of "be" is "were".`, 'be-agreement');
    if (t === 'people' && ['is', 'was', 'has'].includes(next) && ![1, 2, 3, 4].some(k => PREPOSITIONS.has(at(i - k)))) fix(i, i + 1, tokens[i].text + ' ' + { is: 'are', was: 'were', has: 'have' }[next], '"People" is plural: people are, people were, people have.', 'people-plural');
    if (t === 'there' && (next === 'is' || next === 'was') && /^(two|three|four|five|six|seven|eight|nine|ten|many|several|[2-9]|\d\d+)$/.test(at(i + 2))) {
      fix(i, i + 1, tokens[i].text + (next === 'is' ? ' are' : ' were'), `"There ${next === 'is' ? 'are' : 'were'}" goes with more than one thing.`, 'there-are');
    }
    if (SINGULAR.has(t) && next === "don't" && !DO.has(previous)) fix(i, i + 1, tokens[i].text + " doesn't", `With "${t}", use "doesn't".`, 'do-agreement');
    if (PLURAL.has(t) && next === "doesn't") fix(i, i + 1, tokens[i].text + " don't", `With "${t}", use "don't".`, 'do-agreement');
    if (t === 'does' && PLURAL.has(next) && verb(at(i + 2))?.form === 'base') replace(i, 'do', `Use "do" with "${next}": do ${next}...?`, 'do-agreement');
    if (t === 'do' && (next === 'he' || next === 'she')) replace(i, 'does', `Use "does" with "${next}": does ${next}...?`, 'do-agreement');
    if (subjectSingular(i) && next === 'have' && !DO.has(previous) && !MODALS.has(previous) && !CAUSES.has(previous)) {
      fix(i, i + 1, tokens[i].text + ' has', `With "${tokens[i].text}", use "has".`, 'have-agreement');
    }
    if (subjectPlural(i) && next === 'has' && !DO.has(previous)) fix(i, i + 1, tokens[i].text + ' have', `With "${t}", use "have".`, 'have-agreement');

    // he/she/it (or "my mom") + verb: add -s. I/you/we/they + verb-s: take it off.
    if (subjectSingular(i) && !DO.has(previous) && !MODALS.has(previous) && !CAUSES.has(previous) && !(previous === 'and' && (SINGULAR.has(at(i - 2)) || PLURAL.has(at(i - 2)) || /^[A-Z]/.test(tokens[i - 2]?.text || ''))) && !ASKS.has(previous) && !(previous === 'that' && ASKS.has(at(i - 2)))) {
      const v = skipAdverbs(i + 1), form = verb(at(v));
      if (form && form.form === 'base' && !['be', 'have', 'do'].includes(form.base) && !SAME_PAST.has(at(v))) {
        const sentence = text.slice(tokens[i].start).split(/[.!?]/)[0] + ' ' + text.slice(0, tokens[i].start).split(/[.!?]/).at(-1);
        if (PAST_MARKERS.test(sentence) && !FUTURE.test(sentence)) replace(v, form.past, `This happened in the past, so use the past tense: ${form.past}.`, 'past-tense');
        else replace(v, form.s, `With "${tokens[i].text}", add -s to the verb: ${tokens[i].text} ${form.s}.`, 'third-person-s');
      }
    }
    if (subjectPlural(i) && !DO.has(previous)) {
      const v = skipAdverbs(i + 1), form = verb(at(v));
      if (form && form.form === 's' && !['is', 'has', 'does'].includes(at(v)) && !SAME_PAST.has(form.base) && at(v + 1) !== 'for') {
        replace(v, form.base, `With "${t}", the verb has no -s: ${t} ${form.base}.`, 'no-s');
      }
    }

    // After can/will/did/doesn't... (also "did you ...?"), the verb is in its base form.
    const name = /^[A-Z]/.test(tokens[i].text) && !clauseStart(i);
    if ((MODALS.has(t) || DO.has(t)) && !NOT_HELPER_BEFORE.has(previous) && !name && previous !== 'to') {
      let v = i + 1;
      if (PLURAL.has(next) || SINGULAR.has(next)) v++;
      if (at(v) === 'not') v++;
      v = skipAdverbs(v);
      const form = verb(at(v));
      // "all we can to help" and "what you could to fix it" are fine: something was left out after "can".
      const elided = [at(i - 2), at(i - 3)].some(w => /^(what|all|everything|anything|whatever|best|much|as)$/.test(w));
      if (MODALS.has(t) && at(v) === 'to' && verb(at(v + 1))?.form === 'base' && !elided && (SINGULAR.has(previous) || PLURAL.has(previous))) {
        fix(i, v, t === 'would' ? tokens[i].text + ' like to' : tokens[i].text, t === 'would' ? 'Say "would like to" (or "want to").' : `No "to" after "${t}": ${t} ${at(v + 1)}.`, 'modal-to');
      } else if (form && form.base !== 'be' && (form.form === 's' && (MODALS.has(t) || v > i + 1) || (form.form === 'past' && form.past !== form.base))) {
        const between = tokens.slice(i + 1, v).map(x => x.text).join(' ');
        replace(v, form.base, `After "${tokens[i].text}", use the base verb: ${tokens[i].text} ${between ? between + ' ' : ''}${form.base}.`, 'base-after-helper');
      }
    }
    if (t === 'to' && TO_PAST.has(next)) replace(i + 1, verb(next).base, `After "to", use the base verb: to ${verb(next).base}.`, 'base-after-to');

    // have + past participle.
    if (HAVE.has(t)) {
      let v = i + 1;
      if (at(v) === 'not') v++;
      v = skipAdverbs(v);
      const form = verb(at(v));
      if (form && form.form === 'past' && form.past !== form.participle && at(v) !== 'got') {
        replace(v, form.participle, `After "have", use the past participle: have ${form.participle}.`, 'participle');
      }
    }

    // Time words in the past ("yesterday I go") need the past tense.
    if ((subjectSingular(i) || subjectPlural(i)) && !DO.has(previous) && !MODALS.has(previous) && PAST_MARKERS.test(clauseOf(i)) && !FUTURE.test(clauseOf(i))) {
      const v = skipAdverbs(i + 1), form = verb(at(v));
      const before = tokens.slice(Math.max(0, i - 4), i).map(x => x.lower).join(' ');
      const after = tokens.slice(v + 1, Math.min(tokens.length, v + 7)).map(x => x.lower);
      const cut = after.findIndex(w => /^(that|who|which|than|when|,|\.)$/.test(w));
      const near = (/\b(yesterday|last \w+)\b/.test(before) ? before : '') + ' ' + (cut < 0 ? after : after.slice(0, cut)).join(' ');
      const perfect = form?.base === 'have' && (at(v + 1) === 'been' || verb(at(v + 1))?.form === 'participle');
      if (form && (form.form === 'base' || form.form === 's') && !SAME_PAST.has(form.base) && !REPORTING.has(form.base) && PAST_MARKERS.test(near) && !perfect) {
        const right = form.base === 'be' ? (t === 'you' || PLURAL.has(t) && t !== 'i' ? 'were' : 'was') : form.past;
        const earlier = found.findIndex(f => f.start === tokens[v].start);
        if (earlier >= 0) found.splice(earlier, 1);
        replace(v, right, `This happened in the past, so use the past tense: ${right}.`, 'past-tense');
      }
    }

    // "I am agree", "I am born in".
    if (['am', 'is', 'are', "i'm", "he's", "she's", "we're", "they're", "you're"].includes(t) && (next === 'agree' || next === 'disagree')) {
      const he = t === 'is' && SINGULAR.has(previous) || t === "he's" || t === "she's";
      const subject = { "i'm": 'I', "he's": 'he', "she's": 'she', "we're": 'we', "they're": 'they', "you're": 'you' }[t];
      fix(i, i + 1, (subject ? subject + ' ' : '') + next + (he ? 's' : ''), `"Agree" is already a verb, so no "be": I ${next}, she ${next}s.`, 'be-agree');
    }
    if (['am', 'is', 'are'].includes(t) && next === 'born' && /^(in|on)$/.test(at(i + 2))) replace(i, t === 'are' ? 'were' : 'was', 'Talk about being born in the past: I was born in...', 'born');

    // Articles before jobs: "I am student".
    if (['am', 'is', 'was', "i'm", "he's", "she's"].includes(t) && JOBS.has(next) && /^[a-z]/.test(tokens[i + 1].text)
        && (!word(i + 2) || /^(and|at|in|from|of|for|who|with|now|too|here|there)$/.test(at(i + 2)))) {
      const a = article(tokens[i + 1].text, words);
      replace(i + 1, a + ' ' + tokens[i + 1].text, `Use "a" or "an" before a job: ${a} ${tokens[i + 1].text}.`, 'article-job');
    }

    // Comparisons.
    if (((t === 'more' && COMPARATIVES.has(next)) || (t === 'most' && SUPERLATIVES.has(next))) && !/^(or|and|\d+|two|three|four|five|ten)$/.test(previous)) fix(i, i + 1, tokens[i + 1].text, `"${next}" already means ${t}, so drop "${t}".`, 'double-comparative');
    if (t === 'then' && (COMPARATIVES.has(previous) || ['more', 'less'].includes(at(i - 2))) && !/^[A-Z]/.test(tokens[i + 1]?.text || 'x') && at(i - 2) !== 'or') replace(i, 'than', 'Compare with "than": bigger than, more than.', 'than');

    // Countable and uncountable.
    if (Object.hasOwn(NO_PLURAL, t)) replace(i, NO_PLURAL[t], `"${NO_PLURAL[t]}" has no plural.`, 'uncountable');
    if (t === 'many' && UNCOUNTABLE.has(next) && (!word(i + 2) || FUNCTION_WORDS.has(at(i + 2)))) replace(i, 'much', `Use "much" with things you can't count, like ${next}.`, 'much-many');
    if (t === 'much' && COUNTABLE_PLURALS.has(next)) replace(i, 'many', `Use "many" with things you can count, like ${next}.`, 'much-many');
    if (t === 'every' && /^(days|weeks|years|months|times|mornings|nights)$/.test(next)) replace(i + 1, next.slice(0, -1), `After "every", use one: every ${next.slice(0, -1)}.`, 'every');
    if (t === 'everyday' && i > 0 && (!word(i + 1) || /^(and|but|so|because|at|in|for)$/.test(next))) replace(i, 'every day', '"Everyday" (one word) means ordinary. For how often, write "every day".', 'every-day');
    if (t === 'a' && next === 'lots' && at(i + 2) === 'of') fix(i, i + 1, 'a lot', 'Say "a lot of" or "lots of".', 'a-lot');
    if (t === 'alot') replace(i, 'a lot', '"A lot" is two words.', 'a-lot');
    if (OVER_REGULAR.has(t)) replace(i, OVER_REGULAR.get(t), `"${verb(OVER_REGULAR.get(t)).base}" is irregular: its past is "${OVER_REGULAR.get(t)}".`, 'irregular-past');

    // Verbs that need "to" or "-ing", and fixed phrases.
    if (WANTS.has(t) && TO_VERBS.has(next) && (subjectPlural(i - 1) || subjectSingular(i - 1) || ADVERBS.has(previous) || WANT_BEFORE.has(previous))) {
      replace(i + 1, 'to ' + tokens[i + 1].text, `Use "to" after "${t}": ${t} to ${next}.`, 'want-to');
    }
    if (/^(enjoy|enjoys|enjoyed|avoid|avoids|avoided)$/.test(t) && next === 'to' && verb(at(i + 2))?.form === 'base') {
      fix(i + 1, i + 2, verb(at(i + 2)).ing, `After "${t}", use the -ing form: ${t} ${verb(at(i + 2)).ing}.`, 'enjoy-ing');
    }
    if (t === "let's" && next === 'to') fix(i, i + 1, tokens[i].text, 'No "to" after "let\'s": let\'s go.', 'lets');
    if (/^explain(s|ed|ing)?$/.test(t) && /^(me|us|him)$/.test(next)) replace(i + 1, 'to ' + tokens[i + 1].text, `Explain something to someone: ${t} to ${next}.`, 'explain-to');
    if (/^discuss(es|ed|ing)?$/.test(t) && next === 'about') fix(i, i + 1, tokens[i].text, '"Discuss" needs no "about": discuss the plan.', 'discuss');
    if (t === 'married' && next === 'with' && (/^(a|an|my|his|her|your|their)$/.test(at(i + 2)) || /^[A-Z]/.test(tokens[i + 2]?.text || '') && !word(i + 3))) replace(i + 1, 'to', 'You are married to someone.', 'married-to');
    if (/^return(s|ed|ing)?$/.test(t) && next === 'back') fix(i, i + 1, tokens[i].text, '"Return" already means go back.', 'return-back');
    if (/^(go|goes|went|going|come|comes|came|coming|get|gets|got)$/.test(t) && next === 'to' && at(i + 2) === 'home'
        && (!word(i + 3) || /^(now|today|tonight|early|late|after|at|by|with|and|because|soon)$/.test(at(i + 3)))) {
      fix(i + 1, i + 2, 'home', `No "to" before "home": ${t} home.`, 'go-home');
    }
    if (/^(make|makes|made|making)$/.test(t) && (next === 'homework' || POSSESSIVES.has(next) && at(i + 2) === 'homework')) {
      replace(i, { make: 'do', makes: 'does', made: 'did', making: 'doing' }[t], 'You "do" homework (and "make" things, like a cake).', 'do-homework');
    }
    if (/^(make|makes|made|making)$/.test(t) && /^(a|some)$/.test(next) && /^(photo|photos)$/.test(at(i + 2))) {
      replace(i, { make: 'take', makes: 'takes', made: 'took', making: 'taking' }[t], 'You "take" a photo.', 'take-photo');
    }
    if (/^listen(s|ed|ing)?$/.test(t) && /^(music|songs|podcasts|the|this|that|my|your|his|her|our|their|some)$/.test(next)) {
      replace(i + 1, 'to ' + tokens[i + 1].text, `Listen to something: ${t} to ${next}.`, 'listen-to');
    }
    if (/^depend(s|ed|ing)?$/.test(t) && next === 'of') replace(i + 1, 'on', 'It depends on something.', 'depend-on');
    if (t === 'interested' && /^(on|for|about)$/.test(next)) replace(i + 1, 'in', 'You are interested in something.', 'interested-in');
    if (t === 'good' && next === 'in' && (/ing$/.test(at(i + 2)) || /^(math|maths|english|sports|science|art|music|cooking|languages|chess|football|soccer|tennis)$/.test(at(i + 2)))) {
      replace(i + 1, 'at', 'You are good at something.', 'good-at');
    }
    if (t === 'afraid' && next === 'from') replace(i + 1, 'of', 'You are afraid of something.', 'afraid-of');
    if (/^arriv(e|es|ed|ing)$/.test(t) && next === 'to' && (/^(the|a|an|my|your|his|her|our|their|this|that|school|work|home)$/.test(at(i + 2)) || /^[A-Z]/.test(tokens[i + 2]?.text || ''))) {
      const city = /^[A-Z]/.test(tokens[i + 2]?.text || '');
      replace(i + 1, city ? 'in' : 'at', 'Arrive in a city or country, at a place: arrive in London, arrive at the airport.', 'arrive');
    }
    if (/^enter(s|ed|ing)?$/.test(t) && next === 'in' && /^(the|a)$/.test(at(i + 2)) && /^(room|house|building|class|classroom|shop|store|office|school|car|bus|train|kitchen|restaurant|hall|church|bank|hospital)$/.test(at(i + 3))) fix(i, i + 1, tokens[i].text, '"Enter" needs no "in": enter the room.', 'enter');
    if (/^(could|would|should|must|might)$/.test(t) && next === 'of') replace(i + 1, 'have', `It sounds like "of", but it is written "${t} have".`, 'could-have');
    if (t === 'your' && next === 'welcome' && !word(i + 2) && clauseStart(i)) replace(i, "you're", '"You\'re" means "you are": you\'re welcome.', 'youre');
    if (t === 'its' && /^(a|an|the|so|not|been|going|getting|raining|snowing|too|really|ok|okay|late|cold|hot|true|important|fine|my|your)$/.test(next)) {
      replace(i, "it's", '"It\'s" means "it is". "Its" (no apostrophe) means "belonging to it".', 'its');
    }
    if (t === 'since' && /^(\d+|one|two|three|four|five|six|seven|eight|nine|ten|many|a)$/.test(next) && /^(years?|months?|weeks?|days?|hours?|minutes?|long)$/.test(at(i + 2))
        && !/^(is|are|was|were|of)$/.test(at(i + 3)) && !(at(i + 2) === 'long' && at(i + 3) !== 'time')) {
      replace(i, 'for', 'Use "for" with a length of time (for three years) and "since" with a starting point (since 2020).', 'since-for');
    }
    if ((SINGULAR.has(t) || PLURAL.has(t)) && /^(have|has)$/.test(next) && /^(\d+|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|sixty|seventy)$/.test(at(i + 2))
        && at(i + 3) === 'years' && (at(i + 4) === 'old' || !word(i + 4))) {
      const be = t === 'i' ? 'am' : SINGULAR.has(t) ? 'is' : 'are';
      fix(i, at(i + 4) === 'old' ? i + 4 : i + 3, `${tokens[i].text} ${be} ${tokens[i + 2].text} years old`, 'In English you "are" an age: I am 20 years old.', 'age');
    }
    if (/^(didn't|don't|doesn't|can't|won't|wasn't|isn't|aren't)$/.test(t)) {
      for (let j = i + 1; j <= i + 4 && j < tokens.length && word(j); j++) {
        const any = { nothing: 'anything', nobody: 'anybody', nowhere: 'anywhere' }[at(j)];
        if (any) { replace(j, any, `One "not" is enough: ${t} ... ${any}.`, 'double-negative'); break; }
      }
    }
    if (t === 'me' && next === 'and' && (i === 0 || /^[.!?]$/.test(previous)) && word(i + 2)) {
      let v = i + 2;
      if (POSSESSIVES.has(at(v)) || at(v) === 'the') v++;
      if (word(v) && verb(at(v + 1)) && !PLURAL.has(at(v))) {
        fix(i, v, `${tokens.slice(i + 2, v + 1).map(x => x.text).join(' ')} and I`, 'When you are part of the subject, say "... and I", and put yourself last.', 'me-and');
      }
    }
  }
  return found.sort((a, b) => a.start - b.start);
}

export function applyFixes(text, fixes) {
  let out = '', last = 0;
  for (const f of fixes) { out += text.slice(last, f.start) + f.right; last = f.end; }
  return out + text.slice(last);
}

// One plain word that has a stronger one ("very tired" -> exhausted), or null.
export function upgrade(text) {
  const match = text.match(/\b(very|really|so) (\w+)\b/i);
  if (match && Object.hasOwn(UPGRADES, match[2].toLowerCase())) return { phrase: match[0], better: UPGRADES[match[2].toLowerCase()] };
  return null;
}

export const GENERAL_TIPS = [
  'Join two ideas with "because": "I stayed home because it was raining."',
  'Talk about the past with the past tense: yesterday I went, I saw, I ate.',
  'Ask me a question back! It keeps a conversation going: "What about you?"',
  'Try a longer sentence with "and", "but" or "so".',
  'Use "I usually..." to talk about habits: "I usually drink tea in the morning."',
  'Describe with feelings: "I felt nervous, but it went well."',
  'Practice "would like": "I would like a coffee, please" is more polite than "I want a coffee."',
  'Try the present perfect for life experience: "I have visited Japan twice."',
  'Use "since" with a starting point and "for" with a length of time: since Monday, for three days.',
  'Linking words make you sound fluent: first, then, after that, finally.',
];

export function tutor(text, words, turn = 0) {
  const fixes = checkGrammar(text, words);
  const better = fixes.length ? null : upgrade(text);
  return { fixes, corrected: fixes.length ? applyFixes(text, fixes) : text, upgrade: better, tip: !fixes.length && !better ? GENERAL_TIPS[turn % GENERAL_TIPS.length] : null };
}

// What Kani says out loud about a sentence spoken in Speak mode, before its reply.
export function spokenFeedback(note) {
  if (note.fixes.length) {
    const first = note.fixes[0];
    return `Almost! Say: "${note.corrected}" ${first.why}`;
  }
  if (note.upgrade) return `Good sentence! Tip: instead of "${note.upgrade.phrase}", try "${note.upgrade.better.split(', ')[0]}".`;
  return '';
}

// Read-aloud practice sentences, each about sounds learners of English often mix up.
export const PRACTICE = [
  { focus: 'TH', sentence: 'I think three thin birds are on the path.' },
  { focus: 'TH', sentence: 'This is the thing that they brought together.' },
  { focus: 'TH', sentence: 'Thank you for everything this month.' },
  { focus: 'R', sentence: 'The red car is right around the corner.' },
  { focus: 'L', sentence: 'Lily likes long walks along the lake.' },
  { focus: 'R', sentence: 'Please arrive early for the light rain.' },
  { focus: 'V', sentence: 'We have a very lovely view of the village.' },
  { focus: 'W', sentence: 'What will the weather be like on Wednesday?' },
  { focus: 'IY', sentence: 'The sheep will leave the green field.' },
  { focus: 'IH', sentence: 'The ship is sitting in the big river.' },
  { focus: 'AE', sentence: 'The black cat sat on my hat.' },
  { focus: 'AH', sentence: 'My mother loves fun in the sun.' },
  { focus: 'Z', sentence: 'His dogs always lose their toys at the zoo.' },
  { focus: 'SH', sentence: 'She shares fresh fish with her friends.' },
  { focus: 'CH', sentence: 'Which chair should the children choose?' },
  { focus: 'JH', sentence: 'The judge enjoyed a large orange juice.' },
  { focus: 'NG', sentence: 'They are singing a long song this morning.' },
  { focus: 'HH', sentence: 'Her house has a huge garden behind it.' },
  { focus: 'ER', sentence: 'The first girl learned the word early.' },
  { focus: 'OW', sentence: 'I hope you know the road home.' },
  { focus: 'stress', sentence: 'It is important to be comfortable at work.' },
  { focus: 'stress', sentence: 'I would like a vegetable sandwich, please.' },
  { focus: 'everyday', sentence: 'Could you tell me where the station is?' },
  { focus: 'everyday', sentence: 'I usually walk to work in the morning.' },
  { focus: 'everyday', sentence: 'How much does this coffee cost?' },
  { focus: 'everyday', sentence: 'Nice to meet you. Where are you from?' },
];

const norm = word => word.toLowerCase().replace(/[^a-z0-9']/g, '');
const NUMBERS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'];
const same = (a, b) => a === b || NUMBERS[Number(a)] === b || NUMBERS[Number(b)] === a || a.replace(/'/g, '') === b.replace(/'/g, '');

// Feedback on a sentence read aloud: each target word as good, unclear (Whisper was not sure it heard it) or wrong (it
// heard another word), with the sounds that differ and how to make them.
export function readingFeedback(target, heard, scored, words) {
  const wanted = target.split(/\s+/).filter(w => norm(w)), got = heard.words.map(w => norm(w.text)).filter(Boolean);
  const a = wanted.map(norm);
  // Edit-distance alignment of target words to heard words.
  const cost = Array.from({ length: a.length + 1 }, (_, i) => Array.from({ length: got.length + 1 }, (_, j) => i === 0 ? j : j === 0 ? i : 0));
  for (let i = 1; i <= a.length; i++) for (let j = 1; j <= got.length; j++) {
    cost[i][j] = Math.min(cost[i - 1][j] + 1, cost[i][j - 1] + 1, cost[i - 1][j - 1] + (same(a[i - 1], got[j - 1]) ? 0 : 1));
  }
  const matched = new Array(a.length).fill(null);
  for (let i = a.length, j = got.length; i > 0 && j >= 0;) {
    if (j > 0 && cost[i][j] === cost[i - 1][j - 1] + (same(a[i - 1], got[j - 1]) ? 0 : 1)) { matched[i - 1] = got[j - 1]; i--; j--; }
    else if (cost[i][j] === cost[i - 1][j] + 1) { matched[i - 1] = ''; i--; }
    else if (j > 0) j--;
    else break;
  }
  const scores = scored.map(w => w.probability);
  const result = wanted.map((text, i) => {
    const probability = scores[i] ?? 0, heardAs = matched[i];
    const clean = norm(text);
    const status = heardAs && same(clean, heardAs) ? (probability < 0.3 && !NUMBERS.includes(clean) ? 'unclear' : 'good') : heardAs === '' || heardAs === null ? 'missed' : 'wrong';
    const phones = phonesFor(clean, words);
    const entry = { text, status, probability, ipa: ipa(phones), heard: status === 'wrong' ? heardAs : null, sounds: [] };
    if (status === 'wrong') {
      entry.sounds = soundDifferences(phones, phonesFor(heardAs, words)).filter(d => d.wanted)
        .map(d => ({ wanted: soundName(d.wanted), heard: d.heard ? soundName(d.heard) : null, tip: Object.hasOwn(SOUND_TIPS, d.wanted) ? SOUND_TIPS[d.wanted].tip : '' }));
    }
    return entry;
  });
  const good = result.filter(w => w.status === 'good').length;
  return { words: result, score: Math.round(100 * good / Math.max(1, result.length)) };
}

// Kani's spoken comment on a reading.
export function readingSummary(feedback) {
  const misses = feedback.words.filter(w => w.status !== 'good');
  if (!misses.length) return 'Perfect! Every word was clear.';
  const first = misses[0];
  if (first.status === 'wrong') {
    const sound = first.sounds[0];
    return `${feedback.score >= 70 ? 'Nice reading!' : 'Good try!'} "${first.text.replace(/[^\w']/g, '')}" sounded like "${first.heard}".` + (sound ? ` ${sound.tip}` : '');
  }
  if (first.status === 'missed') return `Good try! I didn't hear "${first.text.replace(/[^\w']/g, '')}". Say every word clearly.`;
  return `Nice reading! "${first.text.replace(/[^\w']/g, '')}" was a little unclear. Listen and try it again.`;
}
