import { useState } from "react";
import { MotionConfig, motion } from "motion/react";
import { Background } from "./components/Background";
import { LusPlayground } from "./components/LusPlayground";
import { Hero } from "./components/Hero";
import { Marquee } from "./components/Marquee";
import { Models } from "./components/Models";
import { Nav } from "./components/Nav";
import { ProductCard } from "./components/ProductCard";
import { Arrow } from "./components/Arrow";
import { LOGO, PRODUCTS } from "./products";

const principles = [
  [
    "01",
    "Keep it personal.",
    "Model inference belongs close to you. Kani can generate replies on your device, with online features available when you choose them.",
  ],
  [
    "02",
    "Build in the open.",
    "Explore the source, inspect the training tools, and follow the experiments. We share the work so you can make something of it.",
  ],
  [
    "03",
    "Make the next version better.",
    "Test the idea, see what works, and improve it. We make room for new experiments while being clear about what each tool can do today.",
  ],
];

export default function App() {
  const [filter, setFilter] = useState<"all" | "build" | "kani">("all");
  const products = PRODUCTS.filter(
    (product) => filter === "all" || product.group === filter,
  );
  return (
    <MotionConfig reducedMotion="user">
      <Background />
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <Nav />
      <main id="main">
        <Hero />
        <Marquee />
        <section
          id="products"
          className="section products-section wrap"
          aria-labelledby="products-title"
        >
          <div className="section-heading">
            <div>
              <p className="eyebrow">01 / THE TOOLBOX</p>
              <h2 id="products-title">
                Good ideas deserve
                <br />
                <em>good tools.</em>
              </h2>
            </div>
            <p>
              For the first line of code, the next question, and everything you
              haven’t thought of yet. Find your way in.
            </p>
          </div>
          <div className="product-toolbar">
            <span className="collection-label">THE KELUS COLLECTION</span>
            <div className="filters" role="group" aria-label="Filter products">
              {(
                [
                  ["all", "All tools"],
                  ["build", "For building"],
                  ["kani", "With Kani"],
                ] as const
              ).map(([value, label]) => (
                <button
                  key={value}
                  aria-pressed={filter === value}
                  onClick={() => setFilter(value)}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
          <div className="product-grid">
            {products.map((product, index) => (
              <ProductCard key={product.id} product={product} index={index} />
            ))}
          </div>
          <div className="collection-footer">
            <span>Different tools. The same spark.</span>
            <a className="text-link" href="https://github.com/KelusIDE">
              Explore the source <Arrow diagonal />
            </a>
          </div>
        </section>
        <LusPlayground />
        <section
          id="models"
          className="model-section"
          aria-labelledby="models-title"
        >
          <div className="wrap model-layout">
            <div className="model-intro">
              <p className="eyebrow">03 / UNDER THE HOOD</p>
              <h2 id="models-title">
                Small models.
                <br />
                <em>Our own beginnings.</em>
              </h2>
              <p>
                Our own tokenizer. Our own transformer. Trained from scratch,
                with no borrowed model weights.
              </p>
              <a
                className="text-link"
                href="https://github.com/KelusIDE/KelusModelB1"
              >
                Look inside Kani <Arrow diagonal />
              </a>
              <p className="model-note">
                These are experimental models. They can make mistakes,
                misunderstand a request, or go off topic.
              </p>
            </div>
            <Models />
          </div>
        </section>
        <section
          id="approach"
          className="section approach-section wrap"
          aria-labelledby="approach-title"
        >
          <div className="section-heading">
            <div>
              <p className="eyebrow">04 / HOW WE THINK</p>
              <h2 id="approach-title">
                A few things
                <br />
                <em>we believe in.</em>
              </h2>
            </div>
            <p>
              We’re an independent software lab. We make tools we want to use,
              and leave the door open for you.
            </p>
          </div>
          <div className="principles">
            {principles.map(([number, title, body]) => (
              <motion.article
                key={number}
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5 }}
              >
                <span className="principle-number">
                  {number}
                  <span>↗</span>
                </span>
                <h3>{title}</h3>
                <p>{body}</p>
              </motion.article>
            ))}
          </div>
        </section>
        <section className="closing wrap" aria-labelledby="closing-title">
          <div className="closing-mark" aria-hidden="true">
            ✳
          </div>
          <p className="eyebrow">THERE’S SOMETHING IN THAT IDEA.</p>
          <h2 id="closing-title">
            Let’s see where
            <br />
            <em>it takes you.</em>
          </h2>
          <a
            className="button primary"
            href="https://kelus-ide.pages.dev/ide.html"
          >
            Start making something <Arrow diagonal />
          </a>
          <a className="closing-secondary" href="https://kelus-b1.pages.dev">
            Or start with a conversation ↗
          </a>
        </section>
      </main>
      <footer className="footer wrap">
        <div className="footer-top">
          <a className="brand" href="#top" aria-label="Kelus home">
            <img src={LOGO} alt="" width={34} height={34} />
            <span>
              kelus<span className="brand-dot">.</span>
            </span>
          </a>
          <p>
            Tools for the next thing.
            <br />
            Built by Kelus.
          </p>
          <nav aria-label="Elsewhere">
            <a href="https://kelus-ide.pages.dev">
              Kelus IDE <Arrow diagonal />
            </a>
            <a href="https://kelus-b1.pages.dev">
              Kani <Arrow diagonal />
            </a>
            <a href="https://github.com/KelusIDE">
              GitHub <Arrow diagonal />
            </a>
          </nav>
        </div>
        <div className="footer-bottom">
          <span>© {new Date().getFullYear()} Kelus Corp.</span>
          <span>THINK. BUILD. MAKE IT YOURS.</span>
          <a href="#top">Back to top ↑</a>
        </div>
      </footer>
    </MotionConfig>
  );
}
