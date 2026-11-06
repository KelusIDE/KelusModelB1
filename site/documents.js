




const SENTENCE = /(?<=[.!?]["')\]]?)\s+(?=["'(\[]?[\p{Lu}\d])/u;
const STOP = new Set(('the and for are but not you all any can her was one our out day get has him his how man new now old see two way '
  + 'who its did yes let put say she too use that with have this will your from they know want been good much some time very when '
  + 'come here just like long make many more only over such take than them well were what also into most other their there these '
  + 'which would about after again being could every first great might never shall since still those under until where while '
  + 'should through because before between however within without another although each both then thus upon used using may per via '
  + 'page pages figure table et al').split(' '));

const REQUEST = new Set(('write me short long brief one two three paragraph paragraphs summary summarize summarise sum up overview '
  + 'about this that the pdf pdfs document documents doc file files paper report article text please can could you your own '
  + 'words based give make tldr essay explain describe say says said tell main idea ideas points point key kani simple '
  + 'simply sentences sentence what does whole entire').split(' '));

const LEANS = /^(?:they|it|its|she|he|her|his|their|this|these|those|that|but|and|so|also|however|then|there)\b/i;
const PARAGRAPH_WORDS = 130, PARAGRAPH_SENTENCES = 7, ANSWER_WORDS = 70;
export const MAX_PAGES = 300;

const words = text => (text.toLowerCase().match(/\p{L}[\p{L}'-]*\p{L}/gu) || []).filter(w => w.length >= 3 && !STOP.has(w));


export const wantsParagraph = prompt =>
  /\b(?:paragraphs?|summar(?:y|i[sz]e)|sum (?:it |this )?up|tl;?dr|overview|in your own words|essay|abstract|write (?:a|an|me|one|something))\b/i.test(prompt);


export const wantsAnalysis = prompt => !prompt.trim()
  || /^(?:please\s+)?(?:analy[sz]e|inspect|check|read)(?:\s+(?:these|the|my|this|attached))?(?:\s+(?:files?|attachments?|pdfs?|documents?|papers?|reports?))?[.!?\s]*$/i.test(prompt.trim())
  || /^what(?:'s| is) (?:this|it|the (?:pdf|document|file|paper|report))(?: \w+)? about\??$/i.test(prompt.trim());





export function sentences(pages) {
  const found = [], counts = new Map();
  let section = 0;
  pages.forEach((raw, index) => {
    const lines = String(raw).replace(/(\p{L})-\n(\p{Ll})/gu, '$1$2').split('\n').map(line => line.trim()).filter(Boolean);

    const lengths = lines.map(line => line.length).sort((a, b) => a - b), usual = lengths[Math.floor(lengths.length * 0.75)] || 0;
    const heading = (line, next) => usual > 40 && line.length < usual * 0.6 && !/[.!?:;,]$/.test(line) && /^[\p{Lu}\d]/u.test(next || '');
    const text = lines.map((line, i) => heading(line, lines[i + 1]) ? line + '\u2029' : line).join(' ').replace(/[ \t]+/g, ' ');
    text.split('\u2029').forEach((block, b) => {
      if (b) section++;
      for (let part of block.trim().split(SENTENCE)) {
        part = part.trim();

        const pieces = part.length > 600 ? part.split(' ').reduce((all, word, i) => (i % 40 ? all[all.length - 1] += ' ' + word : all.push(word), all), []) : [part];
        for (const piece of pieces) {
          const letters = (piece.match(/\p{L}/gu) || []).length;
          if (piece.length < 30 || letters < piece.length * 0.6 || piece.split(' ').length < 5) continue;
          found.push({ page: index + 1, section, text: piece });
          counts.set(piece.toLowerCase(), (counts.get(piece.toLowerCase()) || 0) + 1);
        }
      }
    });
  });
  const repeated = pages.length >= 4 ? Math.max(3, pages.length * 0.3) : Infinity;
  return found.filter(s => counts.get(s.text.toLowerCase()) < repeated);
}







function relevance(all, focus) {
  const bags = all.map(s => new Set(words(s.text))), df = new Map();
  for (const bag of bags) for (const w of bag) df.set(w, (df.get(w) || 0) + 1);
  const n = all.length, wanted = [...new Set(focus.map(w => w.toLowerCase()))];
  const match = (bag, w) => bag.has(w) ? 1 : [...bag].some(b => b.startsWith(w) || w.startsWith(b) && b.length >= 4) ? 0.7 : 0;
  const count = w => bags.filter(bag => match(bag, w)).length;
  const score = list => bags.map(bag => list.reduce((sum, w) => sum + match(bag, w) * Math.log(1 + n / (df.get(w) || 1)), 0));
  const rare = wanted.filter(w => count(w) <= n / 2);
  return { bags, df, hits: score(wanted), sharp: rare.length ? score(rare) : score(wanted) };
}






export function keySentences(all, count, focus = []) {
  const { bags, df, hits } = relevance(all, focus), n = all.length;
  const scores = bags.map((bag, i) => {
    let total = 0;
    for (const w of bag) total += df.get(w) > 1 ? df.get(w) * Math.log(1 + n / df.get(w)) : 0;
    return total / Math.sqrt(1 + all[i].text.split(' ').length) * (i < n * 0.1 ? 1.15 : 1) + hits[i] * 1000;
  });
  const chosen = [];
  for (const i of all.map((_, i) => i).sort((a, b) => scores[b] - scores[a] || a - b)) {
    const same = chosen.some(j => {
      const shared = [...bags[i]].filter(w => bags[j].has(w)).length;
      return shared / Math.max(1, Math.min(bags[i].size, bags[j].size)) > 0.6;
    });
    if (!same) chosen.push(i);
    if (chosen.length >= count) break;
  }
  return chosen.map(i => ({ ...all[i], index: i, hits: hits[i] }));
}


export function topics(all, count = 8) {
  const counts = new Map(), forms = new Map();
  for (const s of all) for (const raw of s.text.match(/\p{L}[\p{L}'-]*\p{L}/gu) || []) {
    const w = raw.toLowerCase();
    if (w.length < 4 || STOP.has(w)) continue;
    counts.set(w, (counts.get(w) || 0) + 1);
    const seen = forms.get(w) || new Map();
    seen.set(raw, (seen.get(raw) || 0) + 1);
    forms.set(w, seen);
  }
  return [...counts].filter(([, c]) => c >= 2).sort((a, b) => b[1] - a[1]).slice(0, count)
    .map(([w]) => [...forms.get(w)].sort((a, b) => b[1] - a[1])[0][0]);
}

const pageList = pages => {
  const sorted = [...new Set(pages)].sort((a, b) => a - b);
  return `${sorted.length > 1 ? 'pages' : 'page'} ${sorted.join(', ')}`;
};


// PDFs saved by Chrome, Google Docs and other Skia-based tools give word spaces as \u0001 and similar control
// characters; read as letters, they glue every line into one "word" and the page looks like a scan.
const CONTROL = /[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F]/g;

export function pdfAttachment(name, size, { title, author = null, pages, total }) {
  pages = pages.map(page => String(page).replace(CONTROL, ' '));
  const all = sentences(pages), count = pages.reduce((n, page) => n + (String(page).match(/\S+/g) || []).length, 0);
  const facts = [`${total.toLocaleString()} ${total === 1 ? 'page' : 'pages'} · ${count.toLocaleString()} words`];
  if (title) facts.push(`Title: ${title.slice(0, 160)}`);
  if (author) facts.push(`Author: ${author.slice(0, 160)}`);
  if (pages.length < total) facts.push(`Read the first ${pages.length} pages.`);
  if (!all.length) facts.push('No readable text: the pages may be scanned images.');
  return { name, kind: 'pdf', size, title: title || null, author: author || null, pages, total, words: count, sentences: all, text: pages.join('\n\n'), report: facts.join('\n') };
}

const NO_TEXT = "I couldn't find readable text in this PDF. It may be scanned pages, which are pictures of text, and I can't read text in pictures yet.";


export function analysis(documents) {
  return documents.map(doc => {
    const lines = [doc.title ? `${doc.title} (${doc.name})` : doc.name,
      `${doc.total.toLocaleString()} ${doc.total === 1 ? 'page' : 'pages'} · ${doc.words.toLocaleString()} words · about ${Math.max(1, Math.round(doc.words / 230))} min to read`];
    if (doc.author) lines.splice(1, 0, `By ${doc.author}`);
    if (doc.pages.length < doc.total) lines.push(`I read the first ${doc.pages.length} pages.`);
    if (!doc.sentences.length) return [...lines, '', NO_TEXT].join('\n');
    const subjects = topics(doc.sentences);
    if (subjects.length) lines.push('', `Main topics: ${subjects.join(', ')}`);
    const key = keySentences(doc.sentences.filter(s => !LEANS.test(s.text)), 5).sort((a, b) => a.index - b.index);
    lines.push('', 'Key points:', ...key.map(s => `• ${s.text} (p. ${s.page})`));
    return lines.join('\n');
  }).join('\n\n') + '\n\nThese are the document\'s own sentences. Ask me a question about it, or ask for a paragraph.';
}


function ranked(documents, prompt) {
  const all = documents.flatMap(doc => doc.sentences.map(s => ({ ...s, name: doc.name }))).map((s, index) => ({ ...s, index }));
  const focus = words(prompt).filter(w => !REQUEST.has(w)), { hits, sharp } = relevance(all, focus);
  return { all, hits, sharp, focus, ranking: all.length ? keySentences(all, 60, focus) : [] };
}

const size = s => s.text.split(' ').length;






function passage(all, sharp, i, limit, { share = 0, onePage = false } = {}) {
  const same = j => all[j] && all[j].name === all[i].name && all[j].section === all[i].section && (!onePage || all[j].page === all[i].page);
  let start = i, end = i, used = size(all[i]);
  while (LEANS.test(all[start].text) && same(start - 1)) used += size(all[--start]);
  const about = j => sharp[j] > 0 && sharp[j] >= sharp[i] * share || LEANS.test(all[j].text);
  while (same(end + 1) && about(end + 1) && used + size(all[end + 1]) <= limit) used += size(all[++end]);
  return all.slice(start, end + 1);
}


function section(all, i, limit) {
  const members = all.filter(s => s.name === all[i].name && s.section === all[i].section);
  let from = members.reduce((n, s) => n + size(s), 0) <= limit ? members[0].index : i;
  while (from > members[0].index && LEANS.test(all[from].text)) from--;
  const chosen = [];
  for (let used = 0, j = from; j <= members.at(-1).index && used + size(all[j]) <= limit; used += size(all[j++])) chosen.push(all[j]);
  return chosen.length ? chosen : [all[i]];
}


const best = (hits, taken = new Set(), share = 0) => {
  const top = Math.max(...hits);
  const i = hits.findIndex((h, i) => !taken.has(i) && h > 0 && h >= top * share && h === Math.max(...hits.filter((_, j) => !taken.has(j))));
  return i < 0 ? null : i;
};

const sources = chosen => [...new Set(chosen.map(s => s.name))]
  .map(name => `${name}, ${pageList(chosen.filter(s => s.name === name).map(s => s.page))}`).join('; ');





export function paragraph(documents, prompt) {
  const { all, hits, sharp, focus, ranking } = ranked(documents, prompt);
  if (!all.length) return NO_TEXT;
  let chosen;
  const first = best(hits);
  if (first !== null) {
    const sectioned = all.some(s => s.name === all[first].name && s.section !== all[first].section);
    chosen = sectioned ? section(all, first, PARAGRAPH_WORDS) : passage(all, sharp, first, PARAGRAPH_WORDS);
    const used = chosen.reduce((n, s) => n + size(s), 0), next = best(hits, new Set(chosen.map(s => s.index)), 0.6);
    if (used < PARAGRAPH_WORDS / 2 && next !== null) chosen = [...chosen, ...passage(all, sharp, next, PARAGRAPH_WORDS - used)];
  } else {
    chosen = [];

    for (const s of ranking.filter(s => !LEANS.test(s.text))) {
      if (chosen.reduce((n, c) => n + c.text.split(' ').length, 0) >= PARAGRAPH_WORDS || chosen.length >= PARAGRAPH_SENTENCES) break;
      chosen.push(s);
    }
  }
  const ordered = [...new Map(chosen.map(s => [s.index, s])).values()].sort((a, b) => a.index - b.index);
  const note = first === null && focus.length ? " Nothing in it matched your words, so this sums up its key points." : '';
  return `${ordered.map(s => s.text).join(' ')}\n\nFrom ${sources(ordered)}. These are the PDF's own sentences, so this says only what it says.${note}`;
}


export function answer(documents, question) {
  if (/^who (?:wrote|made|is the author of|authored) (?:this|it|the (?:pdf|document|paper|report))\b/i.test(question.trim())) {
    const authors = documents.filter(doc => doc.author);
    return authors.length ? authors.map(doc => `${doc.name} lists ${doc.author} as its author.`).join('\n')
      : `${documents.map(doc => doc.name).join(' and ')} doesn't list an author. Look for a name on its first page.`;
  }
  const { all, hits, sharp } = ranked(documents, question);
  if (!all.length) return NO_TEXT;
  const first = best(hits);
  if (first === null) {
    const names = [...new Set(all.map(s => s.name))].join(' and ');
    return `I couldn't find that in ${names}. It is mostly about ${topics(all, 5).join(', ')}. Try other words, or ask me to analyze it.`;
  }
  const quote = i => passage(all, sharp, i, ANSWER_WORDS, { share: 0.5, onePage: true });
  const passages = [quote(first)];
  const elsewhere = new Set(all.filter(s => s.page === all[first].page && s.name === all[first].name).map(s => s.index));
  const other = best(hits, elsewhere, 0.8);
  if (other !== null) passages.push(quote(other));
  return passages.map(parts => `Page ${parts[0].page} of ${parts[0].name}:\n“${parts.map(s => s.text).join(' ')}”`).join('\n\n')
    + '\n\nQuoted from the PDF. Ask for a paragraph if you want more about it.';
}


export function prepareDocuments(prompt, attached) {
  const documents = attached.filter((doc, i) => attached.findIndex(d => d.name === doc.name && d.size === doc.size) === i);
  const text = wantsAnalysis(prompt) ? analysis(documents) : wantsParagraph(prompt) ? paragraph(documents, prompt) : answer(documents, prompt);
  return { helper: true, source: 'document', text, receipt: 'Read on this device · no model inference' };
}
