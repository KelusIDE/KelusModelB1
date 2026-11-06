import { useEffect, useRef, useState } from 'react';
import { useInView, useReducedMotion } from 'motion/react';


export function Typewriter({ text, label }: { text: string; label: string }) {
  const ref = useRef<HTMLPreElement>(null);
  const inView = useInView(ref, { once: true, margin: '-40px' });
  const reduce = useReducedMotion();
  const [shown, setShown] = useState(0);
  useEffect(() => {
    if (!inView) return;
    if (reduce) { setShown(text.length); return; }
    const timer = setInterval(() => setShown(count => {
      if (count >= text.length) { clearInterval(timer); return count; }
      return count + 1;
    }), 36);
    return () => clearInterval(timer);
  }, [inView, reduce, text]);
  return (
    <pre ref={ref} className="snippet">
      <span className="sr-only">{label}: {text}</span>
      <code className="ghost" aria-hidden="true">{text}</code>
      <code className="typed" aria-hidden="true">{text.slice(0, shown)}<span className={`caret${shown >= text.length ? ' done' : ''}`} /></code>
    </pre>
  );
}
