import type { ReactNode } from 'react';
import { motion } from 'motion/react';

const EASE = [0.22, 1, 0.36, 1] as const;


export function SectionHead({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <motion.div className="section-head" initial="hidden" whileInView="show" viewport={{ once: true, margin: '-80px' }}
      variants={{ hidden: {}, show: { transition: { staggerChildren: 0.12 } } }}>
      <div>
        <motion.h2 id={id} variants={{ hidden: { opacity: 0, y: 24 }, show: { opacity: 1, y: 0, transition: { duration: 0.7, ease: EASE } } }}>{title}</motion.h2>
        <motion.span className="underline" variants={{ hidden: { scaleX: 0 }, show: { scaleX: 1, transition: { duration: 0.9, ease: EASE } } }} />
      </div>
      <motion.p variants={{ hidden: { opacity: 0, y: 16 }, show: { opacity: 1, y: 0, transition: { duration: 0.7, ease: EASE } } }}>{children}</motion.p>
    </motion.div>
  );
}
