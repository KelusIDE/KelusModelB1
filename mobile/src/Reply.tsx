import { useState } from 'react';
import { Clipboard } from '@capacitor/clipboard';
import { Icon } from './Icon';
export function Copy({ text }: { text: string }) {
  const [state, setState] = useState('Copy');
  return <button className="copy" onClick={() => void Clipboard.write({ string: text }).then(() => {
    setState('Copied'); setTimeout(() => setState('Copy'), 1800);
  }).catch(() => setState('Copy failed'))}><Icon name={state === 'Copied' ? 'check' : 'copy'} size={15} />{state}</button>;
}

function linked(text: string) {
  return text.split(/(https?:\/\/[^\s<>"')\]]+)/g).map((piece, i) => i % 2
    ? <a key={i} href={piece} target="_blank" rel="noopener noreferrer">{piece.replace(/^https?:\/\//, '')}</a> : piece);
}
export function Reply({ text }: { text: string }) {
  const parts = text.split(/(```[\s\S]*?(?:```|$))/g);
  return <>{parts.map((part, i) => {
    if (!part.startsWith('```')) return <p className="reply-text" key={i}>{linked(part)}</p>;
    const newline = part.indexOf('\n'), language = newline < 0 ? 'code' : part.slice(3, newline).trim();
    const code = (newline < 0 ? '' : part.slice(newline + 1)).replace(/```$/, '').trimEnd();
    return <div className="code-block" key={i}><div className="code-head"><span>{language || 'code'}</span><Copy text={code} /></div><pre><code>{code}</code></pre></div>;
  })}</>;
}
