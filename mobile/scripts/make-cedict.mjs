







import { readFile, writeFile } from 'node:fs/promises';

const MARKS = { a: 'āáǎà', e: 'ēéěè', i: 'īíǐì', o: 'ōóǒò', u: 'ūúǔù', ü: 'ǖǘǚǜ' };


export function pinyin(numbered) {
  return numbered.split(' ').map(syllable => {
    const match = /^([a-zü:]+)([1-5])$/i.exec(syllable);
    if (!match) return syllable;
    const text = match[1].replace(/u:/gi, 'ü'), tone = Number(match[2]);
    if (tone === 5) return text;
    const lower = text.toLowerCase();

    const at = /[ae]/.test(lower) ? lower.search(/[ae]/) : lower.includes('ou') ? lower.indexOf('o') : Math.max(...[...'iouü'].map(v => lower.lastIndexOf(v)));
    if (at < 0) return text;
    const marked = MARKS[lower[at]][tone - 1];
    return text.slice(0, at) + (text[at] === lower[at] ? marked : marked.toUpperCase()) + text.slice(at + 1);
  }).join(' ').replace(/ ?· ?/g, '·');
}

const POINTER = /^(?:old |archaic |Japanese |also written |erroneous )?(?:variant of|see|see also|abbr\. for|same as) /;

if (import.meta.url === `file://${process.argv[1]}`) {
  const source = process.argv[2];
  if (!source) throw new Error('usage: node scripts/make-cedict.mjs path/to/cedict.txt');
  const lines = [];
  for (const line of (await readFile(source, 'utf8')).split('\n')) {
    const match = /^(\S+) (\S+) \[([^\]]+)\] \/(.*)\/$/.exec(line.trim());
    if (!match) continue;
    const [, traditional, simplified, numbered, glosses] = match;
    if ([...traditional].length > 4) continue;
    const meanings = glosses.split('/').filter(g => g && !POINTER.test(g) && !/^CL:/.test(g)).slice(0, 4);
    if (!meanings.length) continue;
    lines.push([traditional, simplified, pinyin(numbered), meanings.join('; ')].join('\t'));
  }
  await writeFile(new URL('../src/learn/cedict.txt', import.meta.url), lines.join('\n') + '\n');
  console.log(`${lines.length} entries`);
}
