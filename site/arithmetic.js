
const SMALL = Object.fromEntries('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split(' ').map((word, i) => [word, i]));
const TENS = Object.fromEntries('twenty thirty forty fifty sixty seventy eighty ninety'.split(' ').map((word, i) => [word, 20 + i * 10]));
TENS.fourty = 40;

function englishNumber(text) {
  text = text.trim().replace(/(?<=[a-z])-(?=[a-z])/g, ' ');
  if (/^-?\d+(?:\.\d+)?$/.test(text)) return text;
  let words = text.split(/\s+/), sign = 1, hundreds = 0;
  if (words[0] === 'negative') { sign = -1; words = words.slice(1); }
  if (words.length >= 2 && words[1] === 'hundred' && SMALL[words[0]] >= 1 && SMALL[words[0]] <= 9) {
    hundreds = SMALL[words[0]] * 100;
    words = words.slice(2);
    if (!words.length) return String(sign * hundreds);
    if (words[0] === 'and') words = words.slice(1);
  }
  let rest;
  if (words.length === 1 && Object.hasOwn(SMALL, words[0])) rest = SMALL[words[0]];
  else if (words.length === 1 && Object.hasOwn(TENS, words[0])) rest = TENS[words[0]];
  else if (words.length === 2 && Object.hasOwn(TENS, words[0]) && SMALL[words[1]] >= 1 && SMALL[words[1]] <= 9) rest = TENS[words[0]] + SMALL[words[1]];
  else return null;
  return String(sign * (hundreds + rest));
}

export function arithmeticExpression(query) {
  if (query.length > 300) return null;
  let text = query.replace(/^(?:what is|what's|whats|how much is|(?:can|could) you (?:please )?calculate|calculate|compute|solve|evaluate)\s+/, '');
  text = text.replace(/(?:[,!? .]+|\s)+(?:please|heh|haha|lol)$/, '').replace(/[?! .]+$/, '');
  text = text.replaceAll('×', '*').replaceAll('÷', '/').replaceAll('−', '-').replace(/\s*=\s*$/, '');
  if (/^[0-9.\s+*/%()\-]+$/.test(text) && /[+*/%\-]/.test(text)) return text;
  const match = text.match(/^(.+?)\s+(plus|minus|times|multiplied by|divided by|over)\s+(.+)$/);
  if (!match) return null;
  const a = englishNumber(match[1]), b = englishNumber(match[3]);
  if (a === null || b === null) return null;
  const operation = { plus: '+', minus: '-', times: '*', 'multiplied by': '*', 'divided by': '/', over: '/' }[match[2]];
  return `${a} ${operation} ${b}`;
}
