import { motion } from 'motion/react';
import { MODELS } from '../products';
import { CountUp } from './CountUp';

const EASE = [0.22, 1, 0.36, 1] as const;


export function Models() {
  return (
    <motion.ul className="models" initial="hidden" whileInView="show" viewport={{ once: true, margin: '-60px' }}
      variants={{ hidden: {}, show: { transition: { staggerChildren: 0.1 } } }}>
      {MODELS.map(model => (
        <motion.li key={model.name} variants={{ hidden: { opacity: 0, x: -28 }, show: { opacity: 1, x: 0, transition: { duration: 0.6, ease: EASE } } }}>
          <div>
            <h3>{model.name}</h3>
            <p>{model.about}</p>
          </div>
          {'count' in model.stat
            ? <span className="stat" aria-label={`${model.stat.count}${model.stat.suffix}`}><CountUp to={model.stat.count} />{model.stat.suffix}</span>
            : <span className={`stat${model.training ? ' training' : ''}`}>{model.stat.label}</span>}
        </motion.li>
      ))}
    </motion.ul>
  );
}
