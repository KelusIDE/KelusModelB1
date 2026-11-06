import { motion, type Variants } from 'motion/react';
import { LOGO } from '../products';

const EASE = [0.22, 1, 0.36, 1] as const;
const line: Variants = { hidden: {}, show: { transition: { staggerChildren: 0.08, delayChildren: 0.2 } } };
const word: Variants = {
  hidden: { opacity: 0, y: '0.55em', filter: 'blur(10px)' },
  show: { opacity: 1, y: 0, filter: 'blur(0px)', transition: { duration: 0.8, ease: EASE } },
};
const rise = (delay: number) => ({ initial: { opacity: 0, y: 18 }, animate: { opacity: 1, y: 0 }, transition: { duration: 0.7, delay, ease: EASE } });


const ORBIT = [{ src: '/img/kelus.png' }, { glyph: '>_' }, { src: '/img/kani-app.png' }, { glyph: '{ }' }, { src: '/img/kel.svg' }];

export function Hero() {
  return (
    <section className="hero">
      <div className="hero-copy">
        <motion.p className="eyebrow" {...rise(0.1)}>Kelus Corp.</motion.p>
        <motion.h1 variants={line} initial="hidden" animate="show">
          {['Tools', 'for', 'people', 'who'].map(text => <span key={text}><motion.span className="word" variants={word}>{text}</motion.span>{' '}</span>)}
          <motion.span className="word" variants={word}><span className="gradient">build</span>.</motion.span>
        </motion.h1>
        <motion.p className="lede" {...rise(0.75)}>
          A code editor where AI agents work as a team, language models we train from scratch, and a programming language made for beginners. All free.
        </motion.p>
        <motion.div className="actions" {...rise(0.9)}>
          <a className="button primary" href="#products">See all products</a>
          <a className="button" href="https://kelus-b1.pages.dev">Try Kani in your browser</a>
        </motion.div>
      </div>

      <motion.div className="orbit" aria-hidden="true" initial={{ opacity: 0, scale: 0.8, rotate: -20 }} animate={{ opacity: 1, scale: 1, rotate: 0 }}
        transition={{ duration: 1.2, delay: 0.35, ease: EASE }}>
        <div className="ring r1" /><div className="ring r2" />
        <div className="halo" />
        <motion.img className="orbit-logo" src={LOGO} alt="" animate={{ y: [0, -12, 0] }}
          transition={{ duration: 6, repeat: Infinity, ease: 'easeInOut' }} whileHover={{ scale: 1.06, rotate: 6 }} />
        <div className="track">
          {ORBIT.map((item, i) => (
            <div key={i} className={`orbiter o${i}`}>
              <div className="upright">
                {'src' in item ? <img src={item.src} alt="" /> : <span className="glyph">{item.glyph}</span>}
              </div>
            </div>
          ))}
        </div>
      </motion.div>
    </section>
  );
}
