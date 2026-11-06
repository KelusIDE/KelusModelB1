import type { LanguageId } from './content.ts';
import type { Item } from './cards.ts';



export type Kind = 'meaning' | 'phrase' | 'listen' | 'type';
export const TYPED: LanguageId[] = ['es', 'fr', 'de', 'pt'];
export interface Question { kind: Kind; item: Item; options: string[]; answer: number }

export function question(item: Item, kind: Kind, all: Item[], random: () => number): Question {
  const show = (x: Item) => kind === 'phrase' ? x.text : x.en;
  const group = (x: Item) => x.id.split('.')[0];

  const others = [...all].sort(() => random() - 0.5)
    .sort((a, b) => Number(group(b) === group(item)) - Number(group(a) === group(item)))
    .filter(x => x.id !== item.id && show(x) !== show(item));
  const options = [show(item)];
  for (const x of others) {
    if (options.length === 4) break;
    if (!options.includes(show(x))) options.push(show(x));
  }
  options.sort(() => random() - 0.5);
  return { kind, item, options, answer: options.indexOf(show(item)) };
}



export function quiz(pool: Item[], all: Item[], id: LanguageId, count: number, random: () => number, listening = true): Question[] {
  const chosen = [...pool].sort(() => random() - 0.5).slice(0, count);
  const kinds: Kind[] = (['meaning', 'phrase', 'listen', 'type'] as Kind[])
    .filter(kind => (kind !== 'listen' || listening) && (kind !== 'type' || TYPED.includes(id)));
  return chosen.map((x, i) => question(x, kinds[i % kinds.length], all, random));
}
