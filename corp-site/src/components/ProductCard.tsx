import type { PointerEvent } from 'react';
import { motion, useMotionValue, useReducedMotion, useSpring } from 'motion/react';
import type { Product } from '../products';
import { Typewriter } from './Typewriter';

const EASE = [0.22, 1, 0.36, 1] as const;


export function ProductCard({ product, index }: { product: Product; index: number }) {
  const reduce = useReducedMotion();
  const tiltX = useMotionValue(0), tiltY = useMotionValue(0);
  const rotateX = useSpring(tiltX, { stiffness: 180, damping: 18 }), rotateY = useSpring(tiltY, { stiffness: 180, damping: 18 });

  const follow = (event: PointerEvent<HTMLElement>) => {
    if (event.pointerType !== 'mouse') return;
    const box = event.currentTarget.getBoundingClientRect();
    const x = (event.clientX - box.left) / box.width, y = (event.clientY - box.top) / box.height;
    event.currentTarget.style.setProperty('--mx', `${x * 100}%`);
    event.currentTarget.style.setProperty('--my', `${y * 100}%`);
    if (!reduce) { tiltY.set((x - 0.5) * 7); tiltX.set((0.5 - y) * 7); }
  };
  const rest = () => { tiltX.set(0); tiltY.set(0); };

  return (
    <motion.article className={`card${product.featured ? ' featured' : ''}`} onPointerMove={follow} onPointerLeave={rest}
      style={{ rotateX, rotateY, transformPerspective: 1000 }}
      initial={{ opacity: 0, y: 48, scale: 0.97 }} whileInView={{ opacity: 1, y: 0, scale: 1 }}
      viewport={{ once: true, margin: '-60px' }} transition={{ duration: 0.7, delay: (index % 2) * 0.12, ease: EASE }}>
      <div className="card-top">
        {'src' in product.icon
          ? <img className="icon" src={product.icon.src} alt="" width={56} height={56} />
          : <span className="icon glyph" aria-hidden="true">{product.icon.glyph}</span>}
        <span className={`status ${product.status}`}>{product.statusLabel}</span>
      </div>
      <p className="kind">{product.kind}</p>
      <h3>{product.name}</h3>
      <p>{product.description}</p>
      {product.chips.length > 0 && <ul className="chips" aria-label="Platforms">{product.chips.map(chip => <li key={chip}>{chip}</li>)}</ul>}
      {product.snippet && <Typewriter text={product.snippet} label={`${product.name} example`} />}
      {product.links.length > 0 && (
        <div className="links">
          {product.links.map(link => (
            <a key={link.href} className={link.style === 'text' ? 'text-link' : `button${link.style === 'primary' ? ' primary' : ''}`} href={link.href}>{link.label}</a>
          ))}
        </div>
      )}
      {product.note && <p className="fine">{product.note}</p>}
    </motion.article>
  );
}
