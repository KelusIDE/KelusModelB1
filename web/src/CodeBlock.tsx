import { useEffect, useRef, useState } from 'react';
import { Highlight, Prism, type PrismTheme } from 'prism-react-renderer';

const ALIASES: Record<string, string> = { py: 'python', python3: 'python', js: 'javascript', node: 'javascript', ts: 'typescript',
  html: 'markup', xml: 'markup', svg: 'markup', htm: 'markup', yml: 'yaml', 'c++': 'cpp', rs: 'rust', golang: 'go', kt: 'kotlin' };



const THEME: PrismTheme = {
  plain: { color: 'var(--ink)', backgroundColor: 'transparent' },
  styles: [
    { types: ['comment', 'prolog', 'doctype', 'cdata'], style: { color: 'var(--syntax-comment)', fontStyle: 'italic' } },
    { types: ['keyword', 'boolean', 'atrule', 'important', 'selector'], style: { color: 'var(--syntax-keyword)' } },
    { types: ['string', 'char', 'template-string', 'attr-value', 'regex', 'url', 'inserted'], style: { color: 'var(--syntax-string)' } },
    { types: ['number', 'constant', 'symbol'], style: { color: 'var(--syntax-number)' } },
    { types: ['function', 'function-variable', 'method'], style: { color: 'var(--syntax-function)' } },
    { types: ['class-name', 'builtin', 'maybe-class-name', 'decorator'], style: { color: 'var(--syntax-class)' } },
    { types: ['tag', 'property', 'deleted', 'namespace'], style: { color: 'var(--syntax-tag)' } },
    { types: ['attr-name', 'variable', 'parameter'], style: { color: 'var(--syntax-attr)' } },
    { types: ['operator', 'entity', 'punctuation'], style: { color: 'var(--syntax-punctuation)' } },
  ],
};


// A whole HTML page (not a snippet of markup inside another language's answer).
const isPage = (code: string, lang?: string) => ['html', 'htm'].includes(lang?.toLowerCase() || '') || (!lang && /^\s*<(!doctype html|html)\b/i.test(code));

// Shows an HTML reply as a page while it is written. /preview.html runs in a sandbox without same-origin, so the page
// can't reach this site, its storage or the account, and its CSP blocks all network requests. Scripts run only once
// the code block is complete; until then the frame redraws a few times a second.
function HtmlPreview({ code, live }: { code: string; live: boolean }) {
  const frame = useRef<HTMLIFrameElement>(null);
  const [loaded, setLoaded] = useState(false);
  const [runs, setRuns] = useState(0);
  const sent = useRef(0);
  useEffect(() => {
    if (!loaded) return;
    const send = () => { sent.current = Date.now(); frame.current?.contentWindow?.postMessage({ html: code, run: !live }, '*'); };
    if (!live) { send(); return; }
    const timer = setTimeout(send, Math.max(0, 400 - (Date.now() - sent.current)));
    return () => clearTimeout(timer);
  }, [code, live, loaded, runs]);
  return (
    <div className="html-preview">
      <iframe ref={frame} src="/preview.html" title="Live preview of the generated page" sandbox="allow-scripts allow-forms"
        referrerPolicy="no-referrer" onLoad={() => setLoaded(true)} />
      <div className="preview-foot">
        <span>{live ? <><span className="live-dot" />Building live · scripts run when the page is done</> : 'Locked preview · no network, no access to this site'}</span>
        {!live && <button type="button" className="copy-button" onClick={() => setRuns(n => n + 1)}>Run again</button>}
      </div>
    </div>
  );
}

export function CodeBlock({ code, lang, live = false }: { code: string; lang?: string; live?: boolean }) {
  const language = ALIASES[lang?.toLowerCase() || ''] || lang?.toLowerCase() || '';
  const known = Boolean(language && Prism.languages[language]);
  const page = isPage(code, lang);
  const [view, setView] = useState<'preview' | 'code'>('preview');
  const [copied, setCopied] = useState<'idle' | 'done' | 'failed'>('idle');
  useEffect(() => {
    if (copied === 'idle') return;
    const timer = setTimeout(() => setCopied('idle'), 1600);
    return () => clearTimeout(timer);
  }, [copied]);
  const pre = useRef<HTMLPreElement>(null);

  const copy = () => navigator.clipboard.writeText(code).then(() => setCopied('done'), () => {
    const selection = getSelection(), range = document.createRange();
    if (!pre.current || !selection) { setCopied('failed'); return; }
    range.selectNodeContents(pre.current);
    selection.removeAllRanges(); selection.addRange(range);
    const done = document.execCommand('copy');
    selection.removeAllRanges();
    setCopied(done ? 'done' : 'failed');
  });

  return (
    <div className="code-block">
      <div className="code-head">
        <span>{lang || 'code'}</span>
        <span className="code-actions">
          {page && <span className="code-tabs" role="tablist" aria-label="Show">
            {(['preview', 'code'] as const).map(name => <button key={name} type="button" role="tab" aria-selected={view === name}
              className={view === name ? 'active' : ''} onClick={() => setView(name)}>{name === 'preview' ? 'Preview' : 'Code'}</button>)}
          </span>}
          <button type="button" className="copy-button" onClick={copy} aria-label="Copy code">
            {copied === 'done' ? '✓ Copied' : copied === 'failed' ? 'Copy failed' : 'Copy'}
          </button>
        </span>
      </div>
      {page && view === 'preview' ? <HtmlPreview code={code} live={live} /> : <Highlight code={code} language={known ? language : 'plain'} theme={THEME}>
        {({ tokens, getLineProps, getTokenProps }) => (
          <pre ref={pre}><code>
            {tokens.map((line, i) => (
              <span key={i} {...getLineProps({ line })}>
                {line.map((token, j) => <span key={j} {...getTokenProps({ token })} />)}
              </span>
            ))}
          </code></pre>
        )}
      </Highlight>}
      {(language === 'lus' || language === 'kel') && <p className="code-note">Cide is still learning Kel. This program may compile and still give the wrong answer.</p>}
    </div>
  );
}
