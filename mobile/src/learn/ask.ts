


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
const LETTERS: [RegExp, string[]][] = [[/[ñ¿¡]/, ['es']], [/[èêëîœ]/, ['fr']], [/[ßäöü]/, ['de']], [/[ãõ]/, ['pt']], [/ç/, ['fr', 'pt']],
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
