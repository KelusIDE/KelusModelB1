import { useEffect, useRef } from 'react';
import { animate, useInView, useReducedMotion } from 'motion/react';


export function CountUp({ to }: { to: number }) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true });
  const reduce = useReducedMotion();
  const decimals = Number.isInteger(to) ? 0 : 1;
  useEffect(() => {
    const element = ref.current;
    if (!inView || !element) return;
    if (reduce) { element.textContent = to.toFixed(decimals); return; }
    const controls = animate(0, to, { duration: 1.6, ease: [0.22, 1, 0.36, 1], onUpdate: value => { element.textContent = value.toFixed(decimals); } });
    return () => controls.stop();
  }, [inView, reduce, to, decimals]);
  return <span ref={ref}>{(0).toFixed(decimals)}</span>;
}
