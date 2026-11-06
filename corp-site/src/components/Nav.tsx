import { useState } from 'react';
import { motion, useMotionValueEvent, useScroll, useSpring } from 'motion/react';
import { LOGO } from '../products';


export function Nav() {
  const { scrollY, scrollYProgress } = useScroll();
  const progress = useSpring(scrollYProgress, { stiffness: 140, damping: 30, restDelta: 0.001 });
  const [scrolled, setScrolled] = useState(false);
  useMotionValueEvent(scrollY, 'change', value => setScrolled(value > 12));
  return (
    <motion.header className={`nav${scrolled ? ' scrolled' : ''}`} initial={{ y: -28, opacity: 0 }} animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}>
      <div className="nav-inner">
        <a className="brand" href="/" aria-label="Kelus Corp. home">
          <motion.img src={LOGO} alt="" width={34} height={34} whileHover={{ rotate: -12, scale: 1.08 }} transition={{ type: 'spring', stiffness: 300, damping: 14 }} />
          <span>Kelus Corp.</span>
        </a>
        <nav aria-label="Sections">
          <a href="#products">Products</a>
          <a href="#models">Models</a>
          <a href="https://github.com/KelusIDE">GitHub</a>
        </nav>
      </div>
      <motion.div className="progress" style={{ scaleX: progress }} />
    </motion.header>
  );
}
