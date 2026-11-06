



import { readFileSync, writeFileSync } from 'node:fs';

const SOURCE = new URL('../../data/everyday-sources/tatoeba/', import.meta.url);
const FILES = { ko: 'kor', ja: 'jpn', es: 'spa', fr: 'fra', de: 'deu', pt: 'por', zh: 'cmn' };
const PER_LANGUAGE = 300;

const PAIRS = '们們 这這 说說 时時 国國 会會 来來 对對 发發 后後 个個 为為 学學 经經 动動 现現 机機 开開 关關 长長 问問 见見 进進 实實 两兩 车車 东東 门門 应應 电電 气氣 头頭 从從 书書 难難 爱愛 让讓 认認 识識 读讀 写寫 听聽 买買 卖賣 钱錢 银銀 医醫 药藥 话話 语語 请請 谢謝 体體 题題 华華 业業 产產 网網 线線 结結 给給 红紅 绿綠 鸟鳥 鱼魚 马馬 龙龍 风風 飞飛 云雲 万萬 与與 区區 级級 历歷 战戰 农農 观觀 欢歡 乐樂 节節 质質 资資 务務 处處 员員 种種 总總 统統 记記 讲講 该該 报報 场場 还還 运運 远遠 选選 边邊 将將 岁歲 热熱 图圖 团團 园園 圆圓 张張 归歸 当當 么麼 吗嗎 饭飯 喝喝 妈媽 爷爺 们們 们們 们們'.split(' ');
const SIMPLIFIED = new Set(PAIRS.map(p => p[0])), TRADITIONAL = new Set(PAIRS.map(p => p[1]).filter((c, i) => c !== PAIRS[i][0]));

let seed = 2412;
const random = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
const out = {};
for (const [code, file] of Object.entries(FILES)) {
  const rows = readFileSync(new URL(`${file}.txt`, SOURCE), 'utf8').split('\n').map(line => line.split('\t')).filter(r => r.length >= 3)
    .map(([en, text, credit]) => ({ en: en.trim(), text: text.trim(), credit: credit.replace(/^.*Attribution:\s*/, '').trim() }))
    .filter(r => r.en.split(' ').length >= 3 && r.en.split(' ').length <= 7 && r.text.length <= 40 && !/\b(Tom|Mary|John)\b/.test(r.en));
  const unique = [...new Map(rows.map(r => [r.en, r])).values()].sort(() => random() - 0.5);
  if (code === 'zh') {

    const script = r => {
      const traditional = [...r.text].some(c => TRADITIONAL.has(c)), simplified = [...r.text].some(c => SIMPLIFIED.has(c));
      return traditional && simplified ? 'mixed' : traditional ? 'zh-Hant' : simplified ? 'zh-Hans' : 'both';
    };
    out['zh-Hans'] = unique.filter(r => ['zh-Hans', 'both'].includes(script(r))).slice(0, PER_LANGUAGE);
    out['zh-Hant'] = unique.filter(r => ['zh-Hant', 'both'].includes(script(r))).slice(0, PER_LANGUAGE);
  } else out[code] = unique.slice(0, PER_LANGUAGE);
}
writeFileSync(new URL('../src/learn/sentences.json', import.meta.url), JSON.stringify(out) + '\n');
console.log(Object.entries(out).map(([k, v]) => `${k}: ${v.length}`).join(', '));
