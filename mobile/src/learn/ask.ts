


import { COMMON_ENGLISH, COMMON_WORDS } from './chat-words.ts';

const CODES: Record<string, string> = {
  afrikaans: 'af', arabic: 'ar', belarusian: 'be', bulgarian: 'bg', bengali: 'bn', catalan: 'ca', czech: 'cs', welsh: 'cy', danish: 'da',
  german: 'de', greek: 'el', english: 'en', esperanto: 'eo', spanish: 'es', estonian: 'et', persian: 'fa', farsi: 'fa', finnish: 'fi',
  french: 'fr', irish: 'ga', galician: 'gl', gujarati: 'gu', hebrew: 'he', hindi: 'hi', croatian: 'hr', haitian: 'ht', 'haitian creole': 'ht',
  hungarian: 'hu', indonesian: 'id', icelandic: 'is', italian: 'it', japanese: 'ja', georgian: 'ka', kannada: 'kn', korean: 'ko',
  lithuanian: 'lt', latvian: 'lv', macedonian: 'mk', marathi: 'mr', malay: 'ms', maltese: 'mt', dutch: 'nl', norwegian: 'no', polish: 'pl',
  portuguese: 'pt', 'brazilian portuguese': 'pt', romanian: 'ro', russian: 'ru', slovak: 'sk', slovenian: 'sl', albanian: 'sq',
  swedish: 'sv', swahili: 'sw', tamil: 'ta', telugu: 'te', thai: 'th', tagalog: 'tl', filipino: 'tl', turkish: 'tr', ukrainian: 'uk',
  urdu: 'ur', vietnamese: 'vi', chinese: 'zh', mandarin: 'zh', 'simplified chinese': 'zh', 'traditional chinese': 'zh',
  'chinese (simplified)': 'zh', 'chinese (traditional)': 'zh',
};
const NAMES: Record<string, string> = Object.fromEntries(Object.entries(CODES).reverse().map(([name, code]) => [code, name.replace(/\b\w/g, c => c.toUpperCase())]));
NAMES.zh = 'Chinese'; NAMES.pt = 'Portuguese'; NAMES.fa = 'Persian'; NAMES.tl = 'Tagalog'; NAMES.ht = 'Haitian Creole';
export const languageName = (code: string) => NAMES[code] ?? code;

const LANG = `(${Object.keys(CODES).sort((a, b) => b.length - a.length).map(n => n.replace(/[()]/g, '\\$&')).join('|')})`;
const Q = `["“”'‘’]`;

const TO: [RegExp, words: number, language: number][] = [
  [new RegExp(`^(?:how (?:do|would|can|could) (?:you|i|we|one) say|how to say)\\s+${Q}?(.+?)${Q}?\\s+in\\s+${LANG}\\s*[?.!]*$`, 'i'), 1, 2],
  [new RegExp(`^(?:can you |could you |please )?(?:translate|say)\\s+${Q}?(.+?)${Q}?\\s+(?:in|into|to)\\s+${LANG}(?: please)?\\s*[?.!]*$`, 'i'), 1, 2],
  [new RegExp(`^(?:please )?translate (?:this |it )?(?:in|into|to) ${LANG}\\s*:\\s*(.+)$`, 'i'), 2, 1],
  [new RegExp(`^what(?:'s| is)\\s+${Q}(.+?)${Q}\\s+in\\s+${LANG}\\s*[?.!]*$`, 'i'), 1, 2],
  [new RegExp(`^${LANG} for\\s+${Q}(.+?)${Q}\\s*[?.!]*$`, 'i'), 2, 1],
];
const FROM: RegExp[] = [
  new RegExp(`^what does\\s+${Q}?(.+?)${Q}?\\s+mean(?:\\s+in\\s+${LANG})?\\s*[?.!]*$`, 'i'),
  new RegExp(`^(?:please )?translate (?:this |it )?(?:in|into|to) english\\s*:\\s*(.+)()$`, 'i'),
  new RegExp(`^what(?:'s| is) the meaning of\\s+${Q}?(.+?)${Q}?(?:\\s+in\\s+${LANG})?\\s*[?.!]*$`, 'i'),
];


const SCRIPTS: [RegExp, string][] = [[/[가-힯]/, 'ko'], [/[぀-ヿ]/, 'ja'], [/[一-鿿]/, 'zh'], [/[Ѐ-ӿ]/, 'ru'],
  [/[؀-ۿ]/, 'ar'], [/[֐-׿]/, 'he'], [/[฀-๿]/, 'th'], [/[Ͱ-Ͽ]/, 'el'], [/[ऀ-ॿ]/, 'hi']];
const WORDS: Record<string, string[]> = {
  en: 'the is are a an of to you i it this that what my your and in on for with do does be how good thank thanks hello'.split(' '),
  es: 'el la los las que es por para está estoy qué cómo dónde muy gracias hola tengo quiero sí buenos buenas una y del al con mucho usted soy eres adiós hasta luego'.split(' '),
  fr: "le la les des est et je ne pas vous tu nous il elle c'est merci bonjour très oui une du au avec suis comprends ça bonsoir revoir s'il plaît".split(' '),
  de: 'der die das und ist nicht ich du sie wir ein eine danke bitte guten morgen wie was wo sehr ja nein mit auf zu habe bin tschüss geht'.split(' '),
  pt: 'o os as que não você é obrigado obrigada olá bom dia muito um uma do da em com favor eu sou tudo bem tchau'.split(' '),
  it: 'il lo gli che è non sono grazie ciao buongiorno come dove molto per di io sei prego arrivederci'.split(' '),
};
const LETTERS: [RegExp, string[]][] = [[/[ñ¿¡]/, ['es']], [/[êëîœ]/, ['fr']], [/è/, ['fr', 'it']], [/[ßäöü]/, ['de']], [/[ãõ]/, ['pt']], [/ç/, ['fr', 'pt']],
  [/[áíóú]/, ['es', 'pt']], [/[àù]/, ['fr', 'it']], [/[âô]/, ['fr', 'pt']]];


export function guessLanguage(text: string): string | null {
  for (const [script, code] of SCRIPTS) if (script.test(text)) return code;
  const words = text.toLowerCase().match(/[\p{L}']+/gu) ?? [];
  const scores = Object.fromEntries(Object.entries(WORDS).map(([code, list]) => [code, words.filter(w => list.includes(w)).length]));
  for (const [letters, codes] of LETTERS) if (letters.test(text.toLowerCase())) for (const code of codes) scores[code] += 2 / codes.length;
  const [best, second] = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  return best[1] > 0 && best[0] !== 'en' && best[1] > second[1] ? best[0] : null;
}

export interface Ask { text: string; from: string; to: string }


export function translationAsk(message: string): Ask | null {
  const text = message.trim().replace(/\s+/g, ' ');
  for (const [pattern, wordsAt, languageAt] of TO) {
    const match = pattern.exec(text);
    if (!match) continue;
    const words = match[wordsAt], to = CODES[match[languageAt].toLowerCase()], from = guessLanguage(words) ?? 'en';
    return to && from !== to ? { text: words.trim(), from, to } : null;
  }
  for (const pattern of FROM) {
    const match = pattern.exec(text);
    if (!match) continue;
    const named = match[2] ? CODES[match[2].toLowerCase()] : undefined;
    const from = named && named !== 'en' ? named : guessLanguage(match[1]);
    return from ? { text: match[1].trim(), from, to: 'en' } : null;
  }
  return null;
}


export function translationReply(ask: Ask, translated: string): string {
  return ask.to === 'en' ? `It means "${translated}" (${languageName(ask.from)}).` : `In ${languageName(ask.to)}: ${translated}`;
}


// For chat, wrongly translating an English message is worse than missing a foreign one, so these lists leave out words
// that are common in English ("as", "war", "ai"), and two hits are needed. guessLanguage is looser: it only has a phrase to go on.
// COMMON_WORDS comes from web text, so the lists here add what people type in a chat.
const ENGLISH = new Set(`the is are was were am what who whom whose how why when where which this that these those you your my we our they their his she and or of not with for from have has had would can could should does did about tell write make give explain please thanks thank hello hi hey yes good i'm it's what's sin cos tan`.split(' '));
const USUAL_ENGLISH = new Set(COMMON_ENGLISH.split(' '));
const CHAT_WORDS: Record<string, Set<string>> = Object.fromEntries(Object.entries({
  es: 'el la los las que es por para está están estoy qué cómo cuál cuáles cuándo dónde quién porque muy gracias hola tengo tienes quiero puedes puedo hay necesito ayuda sobre pero también como cuando donde cual quien mis tus su sus este esta esto eso adiós buenos buenas días soy eres una un del al con mi mucho usted morir matarme vivir busca buscar encuentra cerca dime cuéntame ayúdame escribe siento',
  fr: "le la les des est et je ne pas vous nous il elle merci bonjour très oui une du au avec suis ça bonsoir comment quelle quel quels pourquoi où qui que dans pour sur mais sont être mon ma c'est s'il plaît j'ai veux voudrais peux mourir vivre en cherche raconte aide dis écris moi",
  de: 'der die das und ist nicht ich du sie wir ein eine danke bitte guten morgen wie wo sehr ja nein mit auf zu habe tschüss geht von warum wer wann wieso kann können mir dir mich dich für sind möchte sterben leben umbringen suche erzähl hilf schreib bin',
  pt: 'que não você é obrigado obrigada olá muito uma da em eu sou tudo bem tchau qual quem onde quando como porque está estão tenho quero preciso ajuda sobre fica morrer viver matar procure perto conte ajude escreva estou',
  it: 'il lo la le gli che è non sono grazie buongiorno buonasera dove molto di io sei prego arrivederci cosa perché chi quando mi ho hai una del della voglio morire vivere ammazzarmi uccidermi qual quale quali quanto quanta quanti quante stai sto stiamo stanno siamo siete hanno fare dire andare venire vedere sapere puoi posso può possono vorrei devo deve dello delle degli dei dal dalla dalle dagli nel nella nello nelle negli sul sulla sulle alla alle allo agli questo questa questi queste quello quella quelli quelle anche ancora sempre tutto tutti tutte aiuto aiutami aiutarmi spiegami spiegare scrivi cerca trova vicino oggi domani ieri adesso qui più meno però oppure mio mia miei mie tuo tua tuoi tue suo sua noi voi lui lei per con senza tra fra capitale bisogno persone vivono città scritto funziona imparare cucinare triste compiti',
}).map(([code, list]) => [code, new Set(`${list} ${COMMON_WORDS[code] ?? ''}`.split(' '))]));
// Plain "I want", "I have": alone these are enough, so that a short message like "quiero morir" is not missed.
const CHAT_STRONG = new Set("quiero tengo necesito puedo tienes puedes veux voudrais peux suis j'ai ich möchte habe quero tenho preciso voglio vorrei è".split(' '));
const CHAT_LETTERS: [RegExp, string[]][] = [[/[ñ¿¡]/, ['es']], [/[êëîœ]/, ['fr']], [/è/, ['fr', 'it']], [/[ßäöü]/, ['de']], [/[ãõ]/, ['pt']], [/ç/, ['fr', 'pt']], [/[áíóú]/, ['es', 'pt']]];
const GREETINGS: Record<string, string> = { hola: 'es', gracias: 'es', adiós: 'es', 'buenos días': 'es', bonjour: 'fr', salut: 'fr', merci: 'fr', bonsoir: 'fr',
  hallo: 'de', danke: 'de', 'guten tag': 'de', tschüss: 'de', olá: 'pt', obrigado: 'pt', obrigada: 'pt', tchau: 'pt', 'bom dia': 'pt', grazie: 'it', buongiorno: 'it', buonasera: 'it', 'grazie mille': 'it', 'come stai': 'it', 'come va': 'it', 'ciao come stai': 'it', 'ciao come va': 'it' };

/** The language to answer a chat message in, or null for English. `previous` keeps short replies like "ok" in the same language. */
export function chatLanguage(message: string, previous: string | null = null): string | null {
  const text = message.toLowerCase(), letters = text.match(/\p{L}/gu) ?? [];
  const other = letters.filter(letter => !/\p{Script=Latin}/u.test(letter)).length;
  if (other && other * 2 >= letters.length) for (const [script, code] of SCRIPTS) if (script.test(text)) return code;
  const typed = message.replaceAll('’', "'").match(/[\p{L}']+/gu) ?? [], words = typed.map(word => word.toLowerCase());
  if (GREETINGS[words.join(' ')]) return GREETINGS[words.join(' ')];
  const english = words.filter(word => ENGLISH.has(word) || USUAL_ENGLISH.has(word)).length;
  // A capitalized word after the first is likely a name, so "El Paso Texas" says nothing about the language.
  const shouting = message === message.toUpperCase();
  // Each word counts once, so "ha ha" and "ho ho ho" are laughing, not Spanish or Italian.
  const counted = [...new Set(words.filter((_, i) => !i || shouting || !/^\p{Lu}/u.test(typed[i])))];
  const scores = Object.entries(CHAT_WORDS).map(([code, list]): [string, number] => [code,
    counted.reduce((sum, word) => sum + (list.has(word) ? (CHAT_STRONG.has(word) ? 2 : 1) : 0), 0) + CHAT_LETTERS.reduce((sum, [found, codes]) => sum + (found.test(text) && codes.includes(code) ? 1 / codes.length : 0), 0)]);
  const [best, second] = scores.sort((a, b) => b[1] - a[1]);
  if (best[1] >= 2 && best[1] > second[1] && best[1] >= english + 2) return best[0];
  // Only a plainly English word ends a translated chat: "ok" and "Paris" are in every language.
  return previous && !words.some(word => ENGLISH.has(word)) ? previous : null;
}

/** The language of the latest translated turn in a chat. */
export function previousLanguage(messages: { role: string; content: string; english?: string }[]): string | null {
  for (const message of [...messages].reverse()) {
    const language = message.role === 'user' && message.english ? chatLanguage(message.content) : null;
    if (language) return language;
  }
  return null;
}

/** For each line of an English reply, whether it stays untranslated: code, source links, and place names and addresses. */
export function keptLines(reply: string, source?: string): boolean[] {
  let code = false;
  return reply.split('\n').map(line => {
    if (line.startsWith('```')) { code = !code; return true; }
    return code || !/\p{L}/u.test(line) || /https?:\/\//.test(line) || (source === 'places' && /^\s|^\d+\. /.test(line));
  });
}
