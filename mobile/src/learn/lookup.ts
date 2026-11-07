import type { LanguageId } from './content.ts';




const SITES: Record<LanguageId, { wiktionary: string; tatoeba: string; script?: string }> = {
  ko: { wiktionary: 'ko', tatoeba: 'kor' }, ja: { wiktionary: 'ja', tatoeba: 'jpn' },
  'zh-Hans': { wiktionary: 'zh', tatoeba: 'cmn', script: 'Hans' }, 'zh-Hant': { wiktionary: 'zh', tatoeba: 'cmn', script: 'Hant' },
  es: { wiktionary: 'es', tatoeba: 'spa' }, fr: { wiktionary: 'fr', tatoeba: 'fra' }, de: { wiktionary: 'de', tatoeba: 'deu' },
  pt: { wiktionary: 'pt', tatoeba: 'por' },
};
const CJK = /[぀-ヿ㐀-鿿가-힯豈-﫿]/;
type Fetch = (url: string, init?: RequestInit) => Promise<Response>;


export function words(text: string, speech: string): string[] {
  const found = typeof Intl.Segmenter === 'function'
    ? [...new Intl.Segmenter(speech, { granularity: 'word' }).segment(text)].filter(s => s.isWordLike).map(s => s.segment)
    : text.split(/[\s.,!?;:¿¡"“”«»()[\]、。！？，：；「」『』（）]+/);
  return [...new Set(found.map(w => w.trim()).filter(w => w && /\p{L}/u.test(w)))];
}


const PARTICLES = ['에서는', '에서', '에게', '한테', '까지', '부터', '으로', '이랑', '하고', '처럼', '보다', '은', '는', '이', '가', '을', '를',
  '에', '의', '도', '로', '와', '과', '만', '랑'];
const ENDINGS: [RegExp, string][] = [[/(해요|했어요|합니다|하세요|해서|하고|해|했다|한다)$/, '하다'], [/(이에요|예요|입니다|이다)$/, ''],
  [/(어요|아요|었어요|았어요|습니다|세요)$/, '다']];


export function stems(word: string, id: LanguageId): string[] {
  const out = [word];
  if (id === 'ko') {
    for (const [ending, replacement] of ENDINGS) if (ending.test(word)) out.push(word.replace(ending, replacement));
    for (const particle of PARTICLES) {
      if (word.length > particle.length && word.endsWith(particle)) { out.push(word.slice(0, -particle.length)); break; }
    }
  } else if (CJK.test(word)) {
    out.push(...[...word].filter(c => CJK.test(c)));
  } else {
    out.push(word.toLowerCase(), word[0].toUpperCase() + word.slice(1).toLowerCase());
  }
  return [...new Set(out.filter(Boolean))];
}


export function plain(html: string): string {
  return html.replace(/<[^>]*>/g, '').replace(/&nbsp;/g, ' ').replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"').replace(/&#0?39;/g, "'").replace(/\s+/g, ' ').replace(/\s+([,.;:)])/g, '$1').trim();
}


const SKIP = new Set(['Syllable', 'Romanization', 'Han character', 'Hanja', 'Hanzi']);
export interface Meaning { part: string; meanings: string[]; examples: { text: string; en?: string }[] }
export interface Entry { word: string; meanings: Meaning[]; url: string }


export async function dictionary(word: string, id: LanguageId, get: Fetch = fetch): Promise<Entry | null> {
  for (const stem of stems(word, id).slice(0, 5)) {
    const ask = () => get(`https://en.wiktionary.org/api/rest_v1/page/definition/${encodeURIComponent(stem)}`,
      { headers: { 'Api-User-Agent': 'Kani language app (https://kani.kelus.org)' } });
    let response = await ask();
    if (response.status === 429) {
      await new Promise(resolve => setTimeout(resolve, 1500));
      response = await ask();
    }
    if (response.status === 404) continue;
    if (!response.ok) throw new Error('The dictionary is unavailable right now.');
    const sections = ((await response.json()) as Record<string, { partOfSpeech: string; definitions: { definition: string;
      parsedExamples?: { example: string; translation?: string }[] }[] }[]>)[SITES[id].wiktionary] ?? [];
    const meanings = sections.filter(section => !SKIP.has(section.partOfSpeech)).map(section => ({
      part: section.partOfSpeech,
      meanings: section.definitions.map(d => plain(d.definition)).filter(m => m && !m.endsWith(':')).slice(0, 4),
      examples: section.definitions.flatMap(d => d.parsedExamples ?? []).map(e => ({ text: plain(e.example), en: e.translation ? plain(e.translation) : undefined }))
        .filter(e => e.text && e.text.length <= 160).slice(0, 2),
    })).filter(m => m.meanings.length).slice(0, 3);
    if (meanings.length) return { word: stem, meanings, url: `https://en.wiktionary.org/wiki/${encodeURIComponent(stem)}` };
  }
  return null;
}

export interface Example { text: string; en: string; credit: string }


export async function sentencesWith(word: string, id: LanguageId, get: Fetch = fetch, count = 6): Promise<Example[]> {
  const site = SITES[id];
  const query = new URLSearchParams({ lang: site.tatoeba, q: word, sort: 'words', 'trans:lang': 'eng', 'showtrans:lang': 'eng', limit: '40' });
  const response = await get(`https://api.tatoeba.org/unstable/sentences?${query}`);
  if (!response.ok) throw new Error('Example sentences are unavailable right now.');
  const { data } = await response.json() as { data: { id: number; text: string; script: string | null; owner: string | null;
    is_unapproved: boolean; translations: { id: number; text: string; owner: string | null; is_unapproved: boolean; is_direct: boolean }[] }[] };
  const short = CJK.test(word) ? 5 : 12;
  return data.filter(s => !s.is_unapproved && (!site.script || s.script === site.script) && s.text.length >= short && s.text.length <= 90)
    .map(s => {
      const en = s.translations.filter(t => !t.is_unapproved).sort((a, b) => Number(b.is_direct) - Number(a.is_direct))[0];
      return en && { text: s.text, en: en.text, credit: `tatoeba.org #${s.id} (${s.owner ?? 'unknown'}) & #${en.id} (${en.owner ?? 'unknown'})` };
    }).filter((s): s is Example => Boolean(s)).slice(0, count);
}
