import { MotionConfig } from 'motion/react';
import { Background } from './components/Background';
import { Hero } from './components/Hero';
import { Marquee } from './components/Marquee';
import { Models } from './components/Models';
import { Nav } from './components/Nav';
import { ProductCard } from './components/ProductCard';
import { SectionHead } from './components/SectionHead';
import { LOGO, PRODUCTS } from './products';

export default function App() {
  return (

    <MotionConfig reducedMotion="user">
      <Background />
      <Nav />
      <main>
        <Hero />
        <Marquee />

        <section id="products" className="section" aria-labelledby="products-title">
          <SectionHead id="products-title" title="Products">Six ways to use what we make, from your desktop to your phone.</SectionHead>
          <div className="grid">
            {PRODUCTS.map((product, index) => <ProductCard key={product.id} product={product} index={index} />)}
          </div>
        </section>

        <section id="models" className="section" aria-labelledby="models-title">
          <SectionHead id="models-title" title="The Kani models">
            Trained from scratch by Kelus: our own tokenizer, our own transformer, no borrowed weights. They're small and still learning, so answers can be wrong.
          </SectionHead>
          <Models />
        </section>
      </main>

      <footer className="footer">
        <div className="footer-brand">
          <img src={LOGO} alt="" width={28} height={28} />
          <span>© {new Date().getFullYear()} Kelus Corp. All rights reserved.</span>
        </div>
        <nav aria-label="Elsewhere">
          <a href="https://kelus-ide.pages.dev">Kelus IDE</a>
          <a href="https://kelus-b1.pages.dev">Kani</a>
          <a href="https://github.com/KelusIDE">GitHub</a>
        </nav>
      </footer>
    </MotionConfig>
  );
}
