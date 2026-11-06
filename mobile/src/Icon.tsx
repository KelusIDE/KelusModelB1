const paths = {
  chat: 'M21 11.5a8.5 8.5 0 0 1-8.5 8.5H4l-2 2v-9.5A8.5 8.5 0 1 1 21 11.5Z',
  history: 'M4 6v5h5M4.5 10a8 8 0 1 1 1 7M12 8v5l3 2',
  settings: 'M4 7h16M4 17h16M8 4v6M16 14v6',
  plus: 'M12 5v14M5 12h14', arrow: 'M12 19V5M5 12l7-7 7 7',
  stop: 'M6 6h12v12H6Z', code: 'm8 7-5 5 5 5m8-10 5 5-5 5m-3-12-2 14',
  spark: 'm12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5Z',
  write: 'm16 3 5 5-12 12-6 1 1-6L16 3ZM13 6l5 5',
  chevron: 'm9 5 7 7-7 7', check: 'm5 12 4 4L19 6',
  copy: 'M9 9h12v12H9ZM15 5V3H3v12h2',
  trash: 'M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7M14 10v7',
  share: 'M12 16V3M7 8l5-5 5 5M5 13v8h14v-8',
  shield: 'm12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6l8-3Zm-4 9 3 3 5-6',
  close: 'm6 6 12 12M6 18 18 6', moon: 'M20 14A8 8 0 0 1 10 4a8.5 8.5 0 1 0 10 10Z',
  search: 'M10.5 18a7.5 7.5 0 1 0 0-15 7.5 7.5 0 0 0 0 15ZM21 21l-5.2-5.2',
  bolt: 'M13 2 4 14h7l-1 8 9-12h-7l1-8Z',
  camera: 'M4 8h3l2-3h6l2 3h3v11H4ZM12 17a4 4 0 1 0 0-8 4 4 0 0 0 0 8Z',
  image: 'M4 4h16v16H4ZM4 16l5-5 4 4 2-2 5 5M15.5 9.5a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3Z',
  translate: 'M4 5h9M8.5 3v2M6 5c1 4 3.5 6.5 6 8M11 5c-1 4-3.5 7-7 9M13 21l4-9 4 9M14.5 18h5',
  cards: 'M8 3h12v14H8ZM4 7v14h12',
  user: 'M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8ZM4.5 20.5c1.2-3.6 4.1-5.5 7.5-5.5s6.3 1.9 7.5 5.5',
  mail: 'M3.5 6h17v12h-17ZM4 6.5l8 6.5 8-6.5',
  download: 'M12 4v11M7 10l5 5 5-5M5 20h14',
  book: 'M4 5.5C4 4.7 4.7 4 5.5 4H11v15H5.5A1.5 1.5 0 0 0 4 20.5v-15ZM20 5.5c0-.8-.7-1.5-1.5-1.5H13v15h5.5c.8 0 1.5.7 1.5 1.5v-15Z',
} as const;
export function Icon({ name, size = 22 }: { name: keyof typeof paths; size?: number }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name]} /></svg>;
}
