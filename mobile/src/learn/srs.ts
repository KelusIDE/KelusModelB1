

export interface Card { box: number; due: string }
export type Grade = 'again' | 'good' | 'easy';
const WAIT = [0, 1, 3, 7, 14, 30];


export function day(date = new Date()) {
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
}
export function addDays(today: string, days: number) {
  const [y, m, d] = today.split('-').map(Number);
  return day(new Date(y, m - 1, d + days));
}

export function review(card: Card | undefined, grade: Grade, today: string): Card {
  if (grade === 'again') return { box: 0, due: today };
  const box = Math.min(WAIT.length - 1, (card?.box ?? 0) + (grade === 'easy' ? 2 : 1));
  return { box, due: addDays(today, WAIT[box]) };
}


export function due(cards: Record<string, Card>, today: string): string[] {
  return Object.entries(cards).filter(([, card]) => card.due <= today)
    .sort(([, a], [, b]) => a.due.localeCompare(b.due) || a.box - b.box).map(([id]) => id);
}


export const known = (cards: Record<string, Card>) => Object.values(cards).filter(card => card.box >= 3).length;

export interface Streak { last: string; days: number }

export function practiced(streak: Streak | undefined, today: string): Streak {
  if (!streak) return { last: today, days: 1 };
  if (streak.last === today) return streak;
  return { last: today, days: streak.last === addDays(today, -1) ? streak.days + 1 : 1 };
}
