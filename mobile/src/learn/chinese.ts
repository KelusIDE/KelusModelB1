

export type Cedict = Map<string, string[]>;
export interface Hanzi { word: string; pinyin: string; meanings: string }

export function readCedict(text: string): Cedict {
  const map: Cedict = new Map();
  for (const line of text.split('\n')) {
    const tab = line.indexOf('\t'), tab2 = line.indexOf('\t', tab + 1);
    if (tab2 < 0) continue;
    for (const key of new Set([line.slice(0, tab), line.slice(tab + 1, tab2)])) {
      const list = map.get(key);
      if (list) list.push(line); else map.set(key, [line]);
    }
  }
  return map;
}


const USUAL: Record<string, string> = Object.fromEntries(([['了', 'le'], ['的', 'de'], ['得', 'de'], ['着著', 'zhe'], ['都', 'dōu'], ['还還', 'hái'],
  ['长長', 'cháng'], ['行', 'xíng'], ['好', 'hǎo'], ['和', 'hé'], ['为為', 'wèi'], ['只', 'zhǐ'], ['会會', 'huì'], ['要', 'yào'], ['觉覺', 'jué'],
  ['看', 'kàn'], ['少', 'shǎo'], ['发發', 'fā'], ['重', 'zhòng'], ['中', 'zhōng'], ['差', 'chà'], ['种種', 'zhǒng'], ['数數', 'shù'],
  ['难難', 'nán'], ['乐樂', 'lè'], ['说說', 'shuō'], ['什', 'shén'], ['么麼', 'me'], ['那', 'nà'], ['哪', 'nǎ'], ['吗嗎', 'ma'], ['呢', 'ne'],
  ['吧', 'ba'], ['啊', 'a'], ['地', 'de'], ['个個', 'gè'], ['一', 'yī'], ['不', 'bù'], ['给給', 'gěi'], ['没沒', 'méi'], ['相', 'xiāng'],
  ['便', 'biàn'], ['教', 'jiāo'], ['朝', 'cháo'], ['量', 'liàng'], ['几幾', 'jǐ'], ['干乾', 'gān'], ['只隻', 'zhī']] as [string, string][])
  .flatMap(([characters, reading]) => [...characters].map(c => [c, reading])));

const rank = (line: string) => {
  const [, , pinyin, meanings] = line.split('\t');
  return (/^[A-Z]/.test(pinyin) ? 2 : 0) + (/^(used in|surname)\b/.test(meanings) ? 1 : 0) - Math.min(meanings.length, 300) / 1000;
};


export function entries(map: Cedict, word: string, script: 'Hans' | 'Hant'): Hanzi[] {
  const lines = [...(map.get(word) ?? [])].sort((a, b) => rank(a) - rank(b));
  const usual = USUAL[word];
  if (usual) lines.sort((a, b) => Number(b.split('\t')[2] === usual) - Number(a.split('\t')[2] === usual));
  return lines.map(line => {
    const [traditional, simplified, pinyin, meanings] = line.split('\t');
    return { word: script === 'Hant' ? traditional : simplified, pinyin, meanings };
  });
}

const longest = (map: Cedict, chars: string[], backward: boolean) => {
  const out: string[] = [];
  for (let i = 0; i < chars.length;) {
    let n = Math.min(4, chars.length - i);
    const piece = () => (backward ? chars.slice(chars.length - i - n, chars.length - i) : chars.slice(i, i + n)).join('');
    while (n > 1 && !map.has(piece())) n--;
    out.push(piece());
    i += n;
  }
  return backward ? out.reverse() : out;
};



export function segment(map: Cedict, text: string): string[] {
  const chars = [...text], front = longest(map, chars, false), back = longest(map, chars, true);
  const alone = (pieces: string[]) => pieces.filter(p => [...p].length === 1).length;
  return front.length < back.length || (front.length === back.length && alone(front) < alone(back)) ? front : back;
}

const HAN = /\p{Script=Han}/u;
const PUNCTUATION: Record<string, string> = { '，': ',', '。': '.', '？': '?', '！': '!', '：': ':', '；': ';', '、': ',', '「': '"', '」': '"', '（': '(', '）': ')' };


export const joined = (pinyin: string) => pinyin.split(' ').reduce((word, syllable, i) =>
  word + (i && /^[aeoāáǎàēéěèōóǒò]/i.test(syllable) ? "'" : '') + syllable, '');


export function pinyinOf(map: Cedict, text: string): string {
  let out = '', latin = false;
  for (const piece of segment(map, text)) {
    if (PUNCTUATION[piece] || /^[\s.,!?;:]$/.test(piece)) { out += PUNCTUATION[piece] ?? piece; latin = false; continue; }
    const best = HAN.test(piece) ? entries(map, piece, 'Hans')[0] : undefined;
    const glued = latin && !best;
    out += (out && !glued && !out.endsWith(' ') && !/^["(]$/.test(out.slice(-1)) ? ' ' : '') + (best ? joined(best.pinyin) : piece);
    latin = !best;
  }
  return out.replace(/\s+/g, ' ').trim();
}


export function lookUp(map: Cedict, word: string, script: 'Hans' | 'Hant'): Hanzi[] {
  if (map.has(word)) return entries(map, word, script);
  return segment(map, word).filter(piece => map.has(piece)).flatMap(piece => entries(map, piece, script).slice(0, 2));
}
