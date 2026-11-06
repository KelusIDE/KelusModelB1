import { useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { KANI_ICON, LOGO } from "../products";
import { Arrow } from "./Arrow";

export function Hero() {
  const [active, setActive] = useState<"kani" | "ide">("kani");
  const reduce = useReducedMotion();
  return (
    <section className="hero wrap" aria-labelledby="hero-title">
      <div className="hero-copy">
        <motion.p
          className="eyebrow"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <span className="live-dot" /> An independent software lab
        </motion.p>
        <motion.h1
          id="hero-title"
          initial={{ opacity: 0, y: reduce ? 0 : 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
        >
          Built for your
          <br />
          <em>next big idea.</em>
        </motion.h1>
        <motion.p
          className="hero-description"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: reduce ? 0 : 0.15 }}
        >
          An editor for your next project. AI that runs on your device. A
          language you can learn by doing.
        </motion.p>
        <div className="hero-actions">
          <a className="button primary" href="#products">
            Find your next tool <Arrow />
          </a>
          <a className="text-link" href="https://kelus-b1.pages.dev">
            Say hello to Kani <Arrow diagonal />
          </a>
        </div>
        <p className="hero-footnote">
          <span>Start with an idea.</span> Make it your own.
        </p>
      </div>
      <motion.div
        className="showcase"
        initial={{ opacity: 0, y: reduce ? 0 : 25 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: reduce ? 0 : 0.1 }}
      >
        <div className="showcase-orbit" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
        <div
          className="showcase-tabs"
          role="tablist"
          aria-label="Product preview"
          onKeyDown={(event) => {
            if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key))
              return;
            event.preventDefault();
            const next =
              event.key === "Home"
                ? "kani"
                : event.key === "End"
                  ? "ide"
                  : active === "kani"
                    ? "ide"
                    : "kani";
            setActive(next);
            document.getElementById(`tab-${next}`)?.focus();
          }}
        >
          <button
            id="tab-kani"
            role="tab"
            tabIndex={active === "kani" ? 0 : -1}
            aria-selected={active === "kani"}
            aria-controls="product-preview"
            onClick={() => setActive("kani")}
          >
            Kani <span>01</span>
          </button>
          <button
            id="tab-ide"
            role="tab"
            tabIndex={active === "ide" ? 0 : -1}
            aria-selected={active === "ide"}
            aria-controls="product-preview"
            onClick={() => setActive("ide")}
          >
            Kelus IDE <span>02</span>
          </button>
        </div>
        <div
          className="preview-window"
          id="product-preview"
          role="tabpanel"
          aria-labelledby={`tab-${active}`}
        >
          <div className="window-bar">
            <div className="window-dots" aria-hidden="true">
              <i />
              <i />
              <i />
            </div>
            <span>
              {active === "kani"
                ? "kani / a little room to think"
                : "kelus / a little room to build"}
            </span>
            <span className="preview-label">Preview</span>
          </div>
          <AnimatePresence mode="wait" initial={false}>
            {active === "kani" ? (
              <motion.div
                className="kani-preview"
                key="kani"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: reduce ? 0 : 0.15 }}
              >
                <img
                  className="preview-otter"
                  src={KANI_ICON}
                  width={68}
                  height={68}
                  alt="Kani otter"
                />
                <p className="preview-greeting">
                  Where will your
                  <br />
                  <em>idea take you?</em>
                </p>
                <div className="preview-prompts">
                  <a href="https://kelus-b1.pages.dev">
                    Explore an idea <span>↗</span>
                  </a>
                  <a href="https://kelus-b1.pages.dev">
                    Make something <span>↗</span>
                  </a>
                </div>
                <a className="preview-input" href="https://kelus-b1.pages.dev">
                  <span>Start a conversation with Kani</span>
                  <span className="send-icon">
                    <Arrow />
                  </span>
                </a>
                <p className="preview-local">
                  <span className="live-dot" /> Model inference on your device
                </p>
              </motion.div>
            ) : (
              <motion.div
                className="ide-preview"
                key="ide"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: reduce ? 0 : 0.15 }}
              >
                <div className="editor-sidebar">
                  <img src={LOGO} width={24} height={24} alt="" />
                  <span>EXPLORER</span>
                  <p>⌄ next-big-idea</p>
                  <p className="selected-file">&nbsp; ◇ hello.js</p>
                  <p>&nbsp; ◇ README.md</p>
                  <div className="sidebar-bottom">
                    Your workspace.
                    <br />
                    Your next step.
                  </div>
                </div>
                <div className="editor-content">
                  <div className="editor-tab">
                    hello.js <span>×</span>
                  </div>
                  <pre>
                    <code>
                      <span className="line-number">1</span>{" "}
                      <span className="code-purple">const</span> idea = {"{"}
                      <br />
                      <span className="line-number">2</span> name:{" "}
                      <span className="code-green">'Something new'</span>,<br />
                      <span className="line-number">3</span> ready:{" "}
                      <span className="code-orange">true</span>
                      <br />
                      <span className="line-number">4</span> {"}"};<br />
                      <span className="line-number">5</span>
                      <br />
                      <span className="line-number">6</span>{" "}
                      <span className="code-comment">
                        // Every project starts somewhere.
                      </span>
                      <br />
                      <span className="line-number">7</span> build(idea);
                      <span className="editor-caret" />
                    </code>
                  </pre>
                  <a
                    className="editor-link"
                    href="https://kelus-ide.pages.dev/ide.html"
                  >
                    Open your workspace <Arrow diagonal />
                  </a>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
        <div className="showcase-caption">
          <span>01 — A LOOK INSIDE</span>
          <span>Small tools. Room for big things.</span>
        </div>
        <div className="floating-label">
          <span className="sparkle" aria-hidden="true">
            ✳
          </span>
          <div>
            Ready when you are.<small>Make the first move.</small>
          </div>
        </div>
      </motion.div>
    </section>
  );
}
