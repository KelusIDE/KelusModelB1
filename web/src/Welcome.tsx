import type { ReactNode } from 'react';
import DotGrid from '@/components/DotGrid';
import Magnet from '@/components/Magnet';
import { Icon } from './Brand';
import { Entrance } from './SiteMotion';
import { useMedia } from './useMedia';
import { useTheme } from './theme';

const SUGGESTIONS = [
  { label: 'Write some code', prompt: 'Write a Python function to reverse a string.' },
  { label: 'Explain an idea', prompt: 'Explain what a variable is with a simple example.' },
  { label: 'Search the web', prompt: 'Search the web for interesting facts about octopuses.' },
];

export function Welcome({ disabled, onPick, composer }: { disabled: boolean; onPick: (prompt: string) => void; composer: ReactNode }) {
  const quiet = useMedia('(prefers-reduced-motion: reduce)');
  const wide = useMedia('(min-width: 900px)');
  const pointer = useMedia('(hover: hover) and (pointer: fine)');
  const { resolved, palette } = useTheme();
  return (
    <section id="welcome" aria-labelledby="welcome-title">
      {wide && !quiet && <div className="welcome-ambient" aria-hidden="true"><DotGrid dotSize={1.8} gap={25}
        baseColor={resolved === 'light' ? '#9da9a0' : '#73809b'} activeColor={palette.accent}
        proximity={90} strength={0.65} glow={0} clickShock={false} intro dpr={1} /></div>}
      <Entrance><h1 id="welcome-title">What can I help with?</h1></Entrance>
      <Entrance delay={0.08}>{composer}</Entrance>
      <Entrance className="suggestions" delay={0.18} stagger={0.09}>
        {SUGGESTIONS.map(item => <Magnet key={item.label} wrapperClassName="suggestion-magnet" padding={16}
          disabled={disabled || quiet || !pointer} maxOffset={4} magnetStrength={9} tilt={2}>
          <button data-prompt={item.prompt} disabled={disabled} onClick={() => onPick(item.prompt)}>{item.label}<Icon name="diagonal" /></button>
        </Magnet>)}
      </Entrance>
    </section>
  );
}
