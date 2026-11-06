

export function normalize(text: string, accents = true) {
  const plain = text.normalize('NFKC').toLowerCase().replace(/[\p{P}\p{S}\s]/gu, '');
  return accents ? plain : plain.normalize('NFD').replace(/\p{M}/gu, '').normalize('NFC');
}

export function distance(a: string, b: string) {
  const x = [...a], y = [...b];
  let row = Array.from({ length: y.length + 1 }, (_, j) => j);
  for (let i = 1; i <= x.length; i++) {
    const next = [i];
    for (let j = 1; j <= y.length; j++) next[j] = Math.min(row[j] + 1, next[j - 1] + 1, row[j - 1] + (x[i - 1] === y[j - 1] ? 0 : 1));
    row = next;
  }
  return row[y.length];
}

export function similarity(expected: string, answer: string, accents = true) {
  const a = normalize(expected, accents), b = normalize(answer, accents);
  return a.length || b.length ? 1 - distance(a, b) / Math.max([...a].length, [...b].length) : 1;
}


export function spoken(expected: string, heard: string[]) {
  const best = heard.map(h => ({ heard: h, score: similarity(expected, h) })).sort((a, b) => b.score - a.score)[0];
  if (!best) return { heard: '', score: 0, verdict: 'again' as const };
  return { ...best, verdict: best.score >= 0.85 ? 'great' as const : best.score >= 0.6 ? 'almost' as const : 'again' as const };
}


export function typed(expected: string, answer: string): 'right' | 'accents' | 'close' | 'wrong' {
  if (normalize(expected) === normalize(answer)) return 'right';
  if (normalize(expected, false) === normalize(answer, false)) return 'accents';
  return similarity(expected, answer, false) >= 0.8 ? 'close' : 'wrong';
}
