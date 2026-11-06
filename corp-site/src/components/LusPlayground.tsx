import "../lus/playground.css";
import { useEffect, useRef, useState } from "react";
import { LUS_EXAMPLES } from "../lus/examples";
import { Arrow } from "./Arrow";

type Phase = "ready" | "running" | "success" | "error" | "stopped";
type RunResult = {
  lines: string[];
  error?: string;
  bytes?: number;
  compileMs?: number;
  runMs?: number;
};

export function LusPlayground() {
  const [selected, setSelected] = useState("hello");
  const example = LUS_EXAMPLES.find((item) => item.id === selected)!;
  const [source, setSource] = useState(example.code);
  const [input, setInput] = useState(example.input);
  const [phase, setPhase] = useState<Phase>("ready");
  const [result, setResult] = useState<RunResult>({ lines: [] });
  const workerRef = useRef<Worker | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const editorRef = useRef<HTMLTextAreaElement>(null);
  const gutterRef = useRef<HTMLDivElement>(null);
  const busy = phase === "running";

  function cleanup() {
    workerRef.current?.terminate();
    workerRef.current = null;
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = null;
  }
  useEffect(() => cleanup, []);

  function stop() {
    cleanup();
    setPhase("stopped");
    setResult({
      lines: [],
      error: "Stopped. You can edit the program and run it again.",
    });
  }
  function load(id: string) {
    cleanup();
    const next = LUS_EXAMPLES.find((item) => item.id === id)!;
    setSelected(id);
    setSource(next.code);
    setInput(next.input);
    setPhase("ready");
    setResult({ lines: [] });
  }
  function execute() {
    cleanup();
    setResult({ lines: [] });
    setPhase("running");
    try {
      const worker = new Worker("/lus/runner.js", { type: "module" });
      workerRef.current = worker;
      worker.onmessage = ({ data }) => {
        if (workerRef.current !== worker) return;
        cleanup();
        setResult(data);
        setPhase(data.type === "result" ? "success" : "error");
      };
      worker.onerror = (event) => {
        event.preventDefault();
        if (workerRef.current !== worker) return;
        cleanup();
        setPhase("error");
        setResult({
          lines: [],
          error:
            "The playground could not start. Reload the page and try again. Your browser needs WebAssembly and Web Workers.",
        });
      };
      timerRef.current = setTimeout(() => {
        if (workerRef.current !== worker) return;
        cleanup();
        setPhase("error");
        setResult({
          lines: [],
          error:
            "Execution stopped after 3 seconds. Check for an endless loop or try a smaller input.",
        });
      }, 3000);
      worker.postMessage({
        source,
        inputs: input === "" ? [] : input.replace(/\r\n?/g, "\n").split("\n"),
      });
    } catch {
      cleanup();
      setPhase("error");
      setResult({
        lines: [],
        error:
          "This browser could not start the playground. Try a browser with WebAssembly and Web Worker support.",
      });
    }
  }
  function download() {
    const url = URL.createObjectURL(
      new Blob([source], { type: "text/plain;charset=utf-8" }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = "main.lus";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return (
    <section
      id="lus-playground"
      className="lus-section wrap"
      aria-labelledby="lus-title"
    >
      <div className="section-heading">
        <div>
          <p className="eyebrow">02 / WRITE. RUN. REPEAT.</p>
          <h2 id="lus-title">
            Meet Lus.
            <br />
            <em>Your first line starts here.</em>
          </h2>
        </div>
        <p>
          Friendly syntax. Real WebAssembly. Edit a program and run it right
          here, in your browser.
        </p>
      </div>
      <div className="lus-workbench">
        <div className="lus-toolbar">
          <div className="lus-identity">
            <img src="/img/lus.svg" alt="" width={30} height={30} />
            <span>
              Lus playground<small>main.lus</small>
            </span>
          </div>
          <label className="lus-example-label">
            EXAMPLE
            <select
              id="lus-example"
              value={selected}
              disabled={busy}
              onChange={(event) => load(event.target.value)}
            >
              {LUS_EXAMPLES.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.label}
                </option>
              ))}
            </select>
          </label>
          <div className="lus-actions">
            <button
              type="button"
              className="lus-reset"
              onClick={() => load(selected)}
            >
              Reset
            </button>
            <button
              type="button"
              id="lus-run"
              className={`button ${busy ? "lus-stop" : "primary"}`}
              onClick={busy ? stop : execute}
            >
              {busy ? "■ Stop" : "Run code"}
              {!busy && <Arrow />}
            </button>
          </div>
        </div>
        <div className="lus-panels">
          <div className="lus-editor-panel">
            <div className="lus-panel-label">
              <label htmlFor="lus-code">SOURCE</label>
              <span>Lus → WebAssembly</span>
            </div>
            <div className="lus-editor">
              <div ref={gutterRef} className="lus-gutter" aria-hidden="true">
                {source.split("\n").map((_, i) => (
                  <span key={i}>{i + 1}</span>
                ))}
              </div>
              <textarea
                id="lus-code"
                ref={editorRef}
                value={source}
                maxLength={32768}
                disabled={busy}
                onChange={(event) => {
                  setSource(event.target.value);
                  if (phase !== "ready") {
                    setPhase("ready");
                    setResult({ lines: [] });
                  }
                }}
                onScroll={(event) => {
                  if (gutterRef.current)
                    gutterRef.current.scrollTop = event.currentTarget.scrollTop;
                }}
                onKeyDown={(event) => {
                  if (
                    event.key === "Enter" &&
                    (event.metaKey || event.ctrlKey)
                  ) {
                    event.preventDefault();
                    execute();
                  }
                  if (event.key === "Tab" && !event.shiftKey) {
                    event.preventDefault();
                    const editor = event.currentTarget;
                    const start = editor.selectionStart;
                    const end = editor.selectionEnd;
                    if (source.length - (end - start) + 4 > 32768) return;
                    setSource(
                      source.slice(0, start) + "    " + source.slice(end),
                    );
                    requestAnimationFrame(() =>
                      editor.setSelectionRange(start + 4, start + 4),
                    );
                  }
                }}
                spellCheck={false}
                autoCapitalize="off"
                autoCorrect="off"
                aria-describedby="lus-editor-hint"
              />
            </div>
            <div className="lus-editor-footer">
              <span id="lus-editor-hint">
                Ctrl / ⌘ + Enter to run · Tab to indent
              </span>
              <button type="button" onClick={download}>
                Download .lus ↓
              </button>
            </div>
          </div>
          <div className="lus-output-panel">
            <div className="lus-panel-label">
              <span>OUTPUT</span>
              <span className={`lus-status ${phase}`} role="status">
                {
                  (
                    {
                      ready: "Ready",
                      running: "Running…",
                      success: "Completed",
                      error: "Check your code",
                      stopped: "Stopped",
                    } as const
                  )[phase]
                }
              </span>
            </div>
            <div
              className="lus-output"
              role="log"
              aria-label="Program output"
              aria-live="polite"
              tabIndex={0}
            >
              {phase === "ready" ? (
                <div className="lus-empty">
                  <span aria-hidden="true">↳</span>
                  <p>
                    Make a change.
                    <br />
                    See what happens.
                  </p>
                  <small>
                    Press Run code to compile and execute your program.
                  </small>
                </div>
              ) : phase === "running" ? (
                <p className="lus-running">
                  Compiling and running your program…
                </p>
              ) : (
                <>
                  <pre>
                    {result.lines?.join("\n") ||
                      (!result.error ? "Program finished without output." : "")}
                  </pre>
                  {result.error && (
                    <pre
                      className={
                        phase === "stopped"
                          ? "lus-stopped-message"
                          : "lus-error"
                      }
                    >
                      {result.error}
                    </pre>
                  )}
                  {phase === "success" && (
                    <p className="lus-run-meta">
                      {result.bytes} bytes of WebAssembly · compiled in{" "}
                      {result.compileMs?.toFixed(1)} ms · ran in{" "}
                      {result.runMs?.toFixed(1)} ms
                    </p>
                  )}
                </>
              )}
            </div>
            <div className="lus-input">
              <label htmlFor="lus-input">
                PROGRAM INPUT <span>For ask() · one answer per line</span>
              </label>
              <textarea
                id="lus-input"
                value={input}
                disabled={busy}
                onChange={(event) => setInput(event.target.value)}
                maxLength={8192}
                placeholder={"Mina\n12"}
                spellCheck={false}
              />
            </div>
          </div>
        </div>
        <div className="lus-bottom">
          <p>
            <span className="live-dot" />
            Runs on your device. No account needed.
          </p>
          <span>Experimental · 3-second execution limit</span>
        </div>
      </div>
      <div className="lus-under">
        <p>{example.hint}</p>
        <a className="text-link" href="https://github.com/KelusIDE/Lus">
          Explore Lus <Arrow diagonal />
        </a>
      </div>
      <details className="lus-reference">
        <summary>
          A little syntax to get you started <span>+</span>
        </summary>
        <div className="lus-syntax">
          <p>
            <code>let name = "Mina"</code>A value that stays the same.
          </p>
          <p>
            <code>var score = 0</code>A value you can change.
          </p>
          <p>
            <code>say "Hello, {"{name}"}"</code>Print text and values.
          </p>
          <p>
            <code>repeat 3 {'{ say "Hi" }'}</code>Run a block more than once.
          </p>
        </div>
        <p className="lus-source-notice">
          Compiler source and <a href="/lus/LICENSE.txt">GPL-3.0 license</a> are
          available <a href="/lus/NOTICE.txt">here</a>.
        </p>
      </details>
    </section>
  );
}
