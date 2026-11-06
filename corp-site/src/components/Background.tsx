import { useEffect } from 'react';


export function Background() {
  useEffect(() => {
    if (!matchMedia('(pointer: fine)').matches) return;
    const root = document.documentElement;
    let frame = 0, x = 0, y = 0;
    const move = (event: PointerEvent) => {
      x = event.clientX; y = event.clientY;
      if (frame) return;
      frame = requestAnimationFrame(() => {
        frame = 0;
        root.style.setProperty('--px', `${x}px`);
        root.style.setProperty('--py', `${y}px`);
      });
    };
    window.addEventListener('pointermove', move, { passive: true });
    return () => { window.removeEventListener('pointermove', move); cancelAnimationFrame(frame); };
  }, []);
  return (
    <div className="backdrop" aria-hidden="true">
      <div className="blob b1" /><div className="blob b2" /><div className="blob b3" />
      <div className="spotlight" />
      <div className="grain" />
    </div>
  );
}
