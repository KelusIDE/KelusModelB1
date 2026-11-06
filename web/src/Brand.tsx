import type { ReactNode } from 'react';

export function KaniMark({ className = '' }: { className?: string }) {
  return <svg className={className} viewBox="0 0 40 40" fill="none" aria-hidden="true">
    <path d="M10 8V32M14 20L28 8M14 20L28 32" stroke="currentColor" strokeWidth="7" strokeLinecap="round" strokeLinejoin="round" />
    <circle cx="32" cy="20" r="3.5" className="mark-dot" />
  </svg>;
}

const paths: Record<string, ReactNode> = {
  plus: <path d="M12 5v14M5 12h14" />,
  history: <><path d="M4 10a8 8 0 1 1 .8 6M4 4v6h6" /><path d="M12 7v5l3 2" /></>,
  arrow: <path d="M12 19V5m-6 6 6-6 6 6" />,
  diagonal: <path d="M6 18 18 6M6 6h12v12" />,
  download: <path d="M12 3v12m-5-5 5 5 5-5M5 17v4h14v-4" />,
  close: <path d="m6 6 12 12M18 6 6 18" />,
  attach: <path d="m8 13 6-6a3 3 0 0 1 4 4l-8 8a5 5 0 0 1-7-7l9-9a2 2 0 0 1 3 3l-9 9" />,
  info: <><circle cx="12" cy="12" r="9" /><path d="M12 11v6m0-10v.1" /></>,
  tune: <><path d="M4 7h16M4 17h16" /><circle cx="9" cy="7" r="2.5" /><circle cx="15" cy="17" r="2.5" /></>,
  check: <path d="m5 12 4 4L19 6" />,
  chat: <path d="M5 4h14a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H9l-6 4V6a2 2 0 0 1 2-2Z" />,
};

export function Icon({ name, className = '' }: { name: keyof typeof paths; className?: string }) {
  return <svg className={`icon ${className}`} width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
}
