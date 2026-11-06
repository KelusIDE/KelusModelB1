



import { RECIPE_QUERY, recipeAnswer } from './recipes.js';

export const NO_ANSWER = "The search results don't say. Try a more specific search.";
export const SEARCH_CALL = /^SEARCH: ?(\S.{0,299})$/;
const SENTENCE = /(?<=[.!?])\s+(?=[A-Z"(])/;
const WORD = /[a-z0-9]+/g;
const COMMON = new Set(('a an the of in on at to for and or is are was were be by with as from that this it its what who when where which how why '
  + 'do does did tell me about you know explain please kani hey').split(' '));


export const maybeSearch = text => 'SEARCH:'.startsWith(text) || text.startsWith('SEARCH:');

export const spokenAnswer = text => text.replace(/^Searching the web for “[^”]*”…\n\n/, '').replace(/\n\nSources:\n[\s\S]*$/, '').trim();


export function readingPrompt(results, question) {
  return ['Search results:', ...results.map(([title, passage], i) => `[${i + 1}] ${title}: ${passage}`)].join('\n') + `\n\nQuestion: ${question}`;
}

const words = text => (text.toLowerCase().match(WORD) || []).filter(w => !COMMON.has(w));

export function backed(question, answer, results) {
  if (answer.trim() === NO_ANSWER) return true;
  const stem = word => word.length > 4 ? word.replace(/(?:ies|es|s)$/, '') : word;
  const said = words(answer.replace(/\[\d+\]/g, ' ')).map(stem), asked = words(question).map(stem);
  const shown = new Set(words(results.map(([title, passage]) => `${title} ${passage}`).join(' ')).map(stem));
  const cited = [...answer.matchAll(/\[(\d+)\]/g)].map(m => Number(m[1]));
  if (cited.some(n => n < 1 || n > results.length)) return false;
  if (asked.length && !asked.some(w => said.includes(w))) return false;
  const about = n => words(results[n - 1].join(' ')).map(stem);
  if (cited.length && asked.length && !cited.some(n => asked.some(w => about(n).includes(w)))) return false;
  return said.length < 3 || said.filter(w => shown.has(w)).length / said.length >= 0.6;
}


export function selectSentences(results, question, budget = 90) {
  const wanted = new Set(words(question)), trimmed = [];
  for (const [title, text] of results) {

    const sentences = text.split(/\s+/).filter(Boolean).join(' ').split(SENTENCE).map(s => s.trim()).filter(Boolean)
      .map(s => s.split(/\s+/).slice(0, budget).join(' '));
    if (!sentences.length) continue;
    const score = i => (i === 0) * 0.5 + [...new Set(words(sentences[i]))].filter(w => wanted.has(w)).length / Math.sqrt(1 + sentences[i].split(/\s+/).length / 20);
    const scores = sentences.map((_, i) => score(i));
    const keep = new Set();
    let used = 0;
    for (const i of sentences.map((_, i) => i).sort((a, b) => scores[b] - scores[a])) {
      const size = sentences[i].split(/\s+/).length;
      if (used + size > budget && keep.size) continue;
      keep.add(i);
      used += size;
    }
    trimmed.push([title, [...keep].sort((a, b) => a - b).map(i => sentences[i]).join(' ')]);
  }
  return trimmed;
}


const BOILERPLATE = new RegExp(['From (?:Simple English )?Wikipedia, the free encyclopedia', 'Jump to (?:navigation|search)',
  'This article is about [^.]*\\.', 'For other uses, see [^.]*\\.', '"[^"]*" redirects here\\.', 'Coordinates?: [^.]*\\.',
  'Toggle the table of contents', 'Navigation menu', 'Retrieved from [^ ]+'].join('|'), 'gi');


export const cleanText = text => text.replace(BOILERPLATE, ' ').replace(/\s+/g, ' ').trim();






export function groundedQuery(query, question) {

  const stem = word => word.length > 3 && word.endsWith('s') ? word.slice(0, -1) : word;
  const asked = new Set((question.toLowerCase().match(/[\p{L}\p{N}]+/gu) || []).map(stem).concat('recipe'));
  const invented = (query.toLowerCase().match(/[\p{L}\p{N}]+/gu) || []).filter(word => word.length >= 3 && !asked.has(stem(word)));
  return invented.length ? question.trim().replace(/[?!.]+$/, '').slice(0, 300) : query;
}



export const READING_CHARACTERS = 1400;


const encyclopedicFirst = sources => [...sources].sort((a, b) => /wikipedia\.org\/wiki\//.test(b.url) - /wikipedia\.org\/wiki\//.test(a.url));






export async function* searchedAnswer(query, question, { generate, webAnswer, fetcher = fetch, signal, fits = text => text.length <= READING_CHARACTERS }) {
  query = groundedQuery(query, question);
  yield `Searching the web for “${query}”…\n\n`;
  const found = await webAnswer({ mode: 'search', query }, fetcher, signal);
  const listed = source => `[${source.n}] ${source.title} — ${source.url}`;
  if (RECIPE_QUERY.test(query)) {

    const results = (found.sources || []).filter(s => s.text && s.url).slice(0, 5).map((s, i) => ({ ...s, n: i + 1 }));
    const card = recipeAnswer(query, results);
    if (card) {
      yield card + '\n\nSources:\n' + results.map(listed).join('\n');
      return;
    }
  }
  const sources = encyclopedicFirst((found.sources || []).filter(s => s.text && s.url)).slice(0, 3).map(s => ({ ...s, text: cleanText(s.text) }));
  if (!sources.length) { yield found.text; return; }


  let prompt, shown;
  for (const budget of [90, 60, 40, 25, 15, 8, 4, 2, 1]) {
    shown = selectSentences(sources.map(s => [s.title, s.text]), question.slice(0, 300), budget);
    prompt = readingPrompt(shown, question.slice(0, 300));
    if (fits(prompt)) break;
  }
  let answer = '';
  for await (const piece of generate([{ role: 'user', content: prompt }])) answer += piece;
  yield backed(`${question} ${query}`, answer.trim(), shown) ? answer : NO_ANSWER;
  yield '\n\nSources:\n' + sources.map((s, i) => `[${i + 1}] ${s.title} — ${s.url}`).join('\n');
}
