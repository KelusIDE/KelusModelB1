import { PHRASES, english, type LanguageId } from './content.ts';



export interface Item { id: string; text: string; say: string; en: string; romanization?: string; mine?: boolean }
export interface Mine { id: string; text: string; en: string; added: string; romanization?: string }

export function items(id: LanguageId, mine: Mine[] = []): Item[] {
  return [...PHRASES.map(p => ({ id: p.id, ...p.forms[id], en: english(p, id) })),
    ...mine.map(m => ({ id: m.id, text: m.text, say: m.text, en: m.en, ...(m.romanization ? { romanization: m.romanization } : {}), mine: true }))];
}

const same = (a: string, b: string) => a.trim().toLowerCase() === b.trim().toLowerCase();


export function own(text: string, en: string, today: string, mine: Mine[], stamp = Date.now(), romanization = ''): Mine | null {
  text = text.trim().replace(/\s+/g, ' '); en = en.trim().replace(/\s+/g, ' ');
  if (!text || !en || mine.some(m => same(m.text, text))) return null;
  return { id: `my.${stamp.toString(36)}.${mine.length}`, text, en, added: today, ...(romanization.trim() ? { romanization: romanization.trim() } : {}) };
}


export function wordOfTheDay(id: LanguageId, today: string): Item {
  const hash = [...today].reduce((h, c) => (h * 31 + c.charCodeAt(0)) % 1_000_003, 7);
  const phrase = PHRASES[hash % PHRASES.length];
  return { id: phrase.id, ...phrase.forms[id], en: english(phrase, id) };
}
