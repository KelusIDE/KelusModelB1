import { useEffect, useState } from "react";
import { LOGO } from "../products";
import { Arrow } from "./Arrow";

export function Nav() {
  const [open, setOpen] = useState(false);
  useEffect(() => {
    const close = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, []);
  return (
    <header className="nav">
      <div className="nav-inner wrap">
        <a
          className="brand"
          href="#top"
          aria-label="Kelus home"
          onClick={() => setOpen(false)}
        >
          <img src={LOGO} alt="" width={34} height={34} />
          <span>
            kelus<span className="brand-dot">.</span>
          </span>
        </a>
        <button
          className="menu-toggle"
          aria-label={open ? "Close navigation" : "Open navigation"}
          aria-expanded={open}
          aria-controls="main-nav"
          onClick={() => setOpen(!open)}
        >
          <span /> <span />
        </button>
        <nav
          id="main-nav"
          className={open ? "is-open" : ""}
          aria-label="Main navigation"
          onClick={() => setOpen(false)}
        >
          <a href="#products">Our tools</a>
          <a href="#models">Our models</a>
          <a href="#approach">Our approach</a>
          <a className="nav-github" href="https://github.com/KelusIDE">
            Find us on GitHub <Arrow diagonal />
          </a>
        </nav>
      </div>
    </header>
  );
}
