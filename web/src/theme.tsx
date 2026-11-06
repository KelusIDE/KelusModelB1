import { createContext, useContext, useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react';
import { useMedia } from './useMedia';

export type Mode = 'system' | 'light' | 'dark';
export type AccentName = 'cobalt' | 'mint' | 'ocean' | 'violet' | 'sunset' | 'rose';
interface Palette { accent: string; ink: string; aurora: string[]; gradient: string[]; shine: string }


export const ACCENTS: Record<AccentName, { label: string; dark: Palette; light: Palette }> = {
  cobalt: { label: 'Kani blue',
    dark: { accent: '#a2b5ff', ink: '#171d29', aurora: ['#2c4fe6', '#a2b5ff', '#e0f585'], gradient: ['#a2b5ff', '#e0f585', '#a2b5ff'], shine: '#ffffff' },
    light: { accent: '#2c4fe6', ink: '#ffffff', aurora: ['#2c4fe6', '#e0f585', '#6882f0'], gradient: ['#2c4fe6', '#6b81e8', '#2c4fe6'], shine: '#8299ef' } },
  mint: { label: 'Mint',
    dark: { accent: '#bce5d1', ink: '#11261b', aurora: ['#1d6a4b', '#bce5d1', '#2c6e5a'], gradient: ['#bce5d1', '#7fd1b0', '#d9f2e6', '#bce5d1'], shine: '#ffffff' },
    light: { accent: '#1d6a4b', ink: '#ffffff', aurora: ['#7fd1b0', '#bce5d1', '#4fa98a'], gradient: ['#1d6a4b', '#3f9c79', '#1d6a4b'], shine: '#7fd1b0' } },
  ocean: { label: 'Ocean',
    dark: { accent: '#a8d4ff', ink: '#0d2238', aurora: ['#1a4f8a', '#a8d4ff', '#2a6cb0'], gradient: ['#a8d4ff', '#6fb3f5', '#dcefff', '#a8d4ff'], shine: '#ffffff' },
    light: { accent: '#1a5fb4', ink: '#ffffff', aurora: ['#8cc4f5', '#cfe6ff', '#5a9de0'], gradient: ['#1a5fb4', '#3b82d6', '#1a5fb4'], shine: '#8cc4f5' } },
  violet: { label: 'Violet',
    dark: { accent: '#d4bcff', ink: '#24123d', aurora: ['#4b2a8a', '#d4bcff', '#6a45b8'], gradient: ['#d4bcff', '#b28cff', '#ece2ff', '#d4bcff'], shine: '#ffffff' },
    light: { accent: '#6c3fc4', ink: '#ffffff', aurora: ['#c4a8f5', '#e6dbff', '#9a74e0'], gradient: ['#6c3fc4', '#8b5fe0', '#6c3fc4'], shine: '#c4a8f5' } },
  sunset: { label: 'Sunset',
    dark: { accent: '#ffc9a3', ink: '#3a1c08', aurora: ['#8a3f1a', '#ffc9a3', '#b8602c'], gradient: ['#ffc9a3', '#ff9f6b', '#ffe6d4', '#ffc9a3'], shine: '#ffffff' },
    light: { accent: '#b4521a', ink: '#ffffff', aurora: ['#f5b48c', '#ffe0cc', '#e08a52'], gradient: ['#b4521a', '#d9733a', '#b4521a'], shine: '#f5b48c' } },
  rose: { label: 'Rose',
    dark: { accent: '#ffb8cf', ink: '#3b0f20', aurora: ['#8a2a4b', '#ffb8cf', '#b8456a'], gradient: ['#ffb8cf', '#ff8fb3', '#ffe1eb', '#ffb8cf'], shine: '#ffffff' },
    light: { accent: '#b8325e', ink: '#ffffff', aurora: ['#f5a8c0', '#ffd9e5', '#e0709a'], gradient: ['#b8325e', '#d9507e', '#b8325e'], shine: '#f5a8c0' } },
};
const MODES: { mode: Mode; label: string }[] = [{ mode: 'system', label: 'System' }, { mode: 'light', label: 'Light' }, { mode: 'dark', label: 'Dark' }];
const KEY = 'kani-theme';
interface Choice { mode: Mode; accent: AccentName }

function stored(): Choice {
  try {
    const value = JSON.parse(localStorage.getItem(KEY) || '{}');
    return { mode: MODES.some(m => m.mode === value.mode) ? value.mode : 'light', accent: Object.hasOwn(ACCENTS, value.accent) ? value.accent : 'cobalt' };
  } catch { return { mode: 'light', accent: 'cobalt' }; }
}


export function applyStoredMode() {
  const { mode } = stored();
  if (mode !== 'system') document.documentElement.dataset.theme = mode;
}

interface Theme extends Choice { resolved: 'light' | 'dark'; palette: Palette; setMode(mode: Mode): void; setAccent(accent: AccentName): void }
const ThemeContext = createContext<Theme | null>(null);
export const useTheme = () => useContext(ThemeContext)!;


export function ThemeProvider({ children }: { children: ReactNode }) {
  const [choice, setChoice] = useState<Choice>(stored);
  const systemLight = useMedia('(prefers-color-scheme: light)');
  const resolved = choice.mode === 'system' ? (systemLight ? 'light' : 'dark') : choice.mode;
  const palette = ACCENTS[choice.accent][resolved];

  useLayoutEffect(() => {
    const root = document.documentElement;
    if (choice.mode === 'system') delete root.dataset.theme; else root.dataset.theme = choice.mode;

    root.style.setProperty('--accent', palette.accent);
    root.style.setProperty('--accent-text', palette.accent);
    root.style.setProperty('--accent-ink', palette.ink);
    root.style.setProperty('--accent-soft', palette.accent + (resolved === 'light' ? '14' : '12'));
    root.style.setProperty('--accent-line', palette.accent + (resolved === 'light' ? '40' : '38'));
    for (const meta of document.querySelectorAll<HTMLMetaElement>('meta[name="theme-color"]')) meta.content = resolved === 'light' ? '#f5f3ed' : '#171d29';
    try { localStorage.setItem(KEY, JSON.stringify(choice)); } catch {  }
  }, [choice, resolved, palette]);

  const value: Theme = { ...choice, resolved, palette,
    setMode: mode => setChoice(previous => ({ ...previous, mode })),
    setAccent: accent => setChoice(previous => ({ ...previous, accent })) };
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}


export function ThemeMenu() {
  const theme = useTheme();
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const outside = (event: PointerEvent) => { if (!root.current?.contains(event.target as Node)) setOpen(false); };
    const escape = (event: KeyboardEvent) => { if (event.key === 'Escape') setOpen(false); };
    document.addEventListener('pointerdown', outside);
    document.addEventListener('keydown', escape);
    return () => { document.removeEventListener('pointerdown', outside); document.removeEventListener('keydown', escape); };
  }, [open]);

  return (
    <div className="theme-menu" ref={root}>
      <button id="theme-button" className="header-button" aria-haspopup="dialog" aria-expanded={open} aria-label="Theme" onClick={() => setOpen(!open)}>
        <span className="theme-icon" aria-hidden="true">◐</span><span className="theme-text">Theme</span>
      </button>
      {open && (
        <div className="theme-panel" role="dialog" aria-label="Theme">
          <div className="theme-label">Appearance</div>
          <div className="segmented" role="radiogroup" aria-label="Appearance">
            {MODES.map(({ mode, label }) => (
              <button key={mode} type="button" role="radio" aria-checked={theme.mode === mode} data-mode={mode} onClick={() => theme.setMode(mode)}>{label}</button>
            ))}
          </div>
          <div className="theme-label">Accent</div>
          <div className="swatches" role="radiogroup" aria-label="Accent colour">
            {(Object.keys(ACCENTS) as AccentName[]).map(name => (
              <button key={name} type="button" role="radio" aria-checked={theme.accent === name} aria-label={ACCENTS[name].label} title={ACCENTS[name].label}
                data-accent={name} className="swatch" style={{ '--swatch': ACCENTS[name][theme.resolved].accent } as CSSProperties}
                onClick={() => theme.setAccent(name)} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
