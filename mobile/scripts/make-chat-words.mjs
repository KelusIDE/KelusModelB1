// Writes src/learn/chat-words.ts: each language's 1,000 most common words, minus words that are also common in English,
// and the 1,000 most common English words, minus words that are also common in those languages ("no", "me", "a").
// The app uses them to notice a chat message in another language. Input: the text samples in data/multilingual-v1.
//   node scripts/make-chat-words.mjs
import { readFileSync, writeFileSync } from 'node:fs';

const common = (language, count) => {
  const text = readFileSync(new URL(`../../data/multilingual-v1/${language}.txt`, import.meta.url), 'utf8').toLowerCase();
  const counts = new Map();
  for (const [word] of text.matchAll(/\p{L}+(?:'\p{L}+)?/gu)) counts.set(word, (counts.get(word) ?? 0) + 1);
  return [...counts].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)).slice(0, count).map(([word]) => word);
};
const LANGUAGES = ['es', 'fr', 'de', 'pt'];
const english = new Set(common('en', 5000)), top = Object.fromEntries(LANGUAGES.map(language => [language, common(language, 1000)]));
const foreign = new Set(Object.values(top).flat());
const lines = LANGUAGES.map(language =>
  `  ${language}: ${JSON.stringify(top[language].filter(word => word.length > 1 && !english.has(word)).join(' '))},`);
writeFileSync(new URL('../src/learn/chat-words.ts', import.meta.url),
  `// Made by scripts/make-chat-words.mjs. Do not edit.\nexport const COMMON_WORDS: Record<string, string> = {\n${lines.join('\n')}\n};\nexport const COMMON_ENGLISH = ${JSON.stringify(common('en', 1000).filter(word => word.length > 1 && !foreign.has(word)).join(' '))};\n`);
