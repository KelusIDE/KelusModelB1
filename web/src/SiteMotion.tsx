import type { ReactNode } from 'react';
import AnimatedContent from '@/components/AnimatedContent';
import StarBorder from '@/components/StarBorder';
import { Icon } from './Brand';
import { useMedia } from './useMedia';
import { useTheme } from './theme';

export function Entrance({ children, className, delay = 0, stagger = 0 }: {
  children: ReactNode; className?: string; delay?: number; stagger?: number;
}) {
  const quiet = useMedia('(prefers-reduced-motion: reduce)');
  if (quiet) return <div className={className}>{children}</div>;
  return <AnimatedContent className={className} trigger="mount" distance={16} duration={0.6}
    initialOpacity={1} delay={delay} stagger={stagger}>{children}</AnimatedContent>;
}

export function SendButton({ disabled }: { disabled: boolean }) {
  const quiet = useMedia('(prefers-reduced-motion: reduce)');
  const { resolved, palette } = useTheme();
  if (disabled || quiet) return <button id="send" type="submit" aria-label="Send message" disabled={disabled}><Icon name="arrow" /></button>;
  return <StarBorder id="send" as="button" type="submit" aria-label="Send message" theme={resolved}
    color={palette.shine} trailColor={palette.accent} duration={5} radius={11} thickness={1.5}
    stars={1} sparkle={false} glow={0.3} hover="brighten" clickPulse={false}
    backgroundColor={palette.accent} textColor={palette.ink}><Icon name="arrow" /></StarBorder>;
}
