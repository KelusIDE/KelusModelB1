import type { ReactNode } from 'react';

export interface Link { label: string; href: string; style: 'primary' | 'secondary' | 'text' }
export interface Product {
  id: string;
  name: string;
  kind: string;
  status: 'available' | 'soon';
  statusLabel: string;
  icon: { src: string } | { glyph: string };
  description: ReactNode;
  chips: string[];
  links: Link[];
  note?: string;
  snippet?: string;
  featured?: boolean;
}
export interface Model { name: string; about: string; stat: { count: number; suffix: string } | { label: string }; training?: boolean }

export const LOGO = '/img/kani.png';

export const PRODUCTS: Product[] = [
  {
    id: 'ide', name: 'Kelus IDE', kind: 'Desktop IDE', status: 'available', statusLabel: 'Available', featured: true,
    icon: { src: '/img/kelus.png' },
    description: 'Code with a team of AI agents. A Coder proposes, a Tester runs your tests, a Reviewer checks the requirements and a Judge decides on the evidence. Nothing touches your files until you approve it. A Coach explains your code while you learn.',
    chips: ['macOS', 'Windows', 'Web IDE', 'Free'],
    links: [
      { label: 'Download Kelus', href: 'https://kelus-ide.pages.dev/#download', style: 'primary' },
      { label: 'Open the Web IDE', href: 'https://kelus-ide.pages.dev/ide.html', style: 'secondary' },
      { label: 'Source on GitHub', href: 'https://github.com/KelusIDE/Kelus', style: 'text' },
    ],
  },
  {
    id: 'kani', name: 'Kani', kind: 'AI chat on the web', status: 'available', statusLabel: 'Available', featured: true,
    icon: { src: '/img/kani.png' },
    description: 'Chat with Kani Otter 4.2, a language model Kelus trained from scratch. It runs entirely in your browser, so your messages stay on your device. Fact questions search the web, sending only the search words.',
    chips: ['Any modern browser', 'No sign-up needed', 'Free'],
    links: [
      { label: 'Try Kani', href: 'https://kelus-b1.pages.dev', style: 'primary' },
      { label: "How it's built", href: 'https://github.com/KelusIDE/KelusModelB1', style: 'text' },
    ],
  },
  {
    id: 'android', name: 'Kani for Android', kind: 'Android app', status: 'available', statusLabel: 'Version 0.5.3',
    icon: { src: '/img/kani-app.png' },
    description: 'Kani in your pocket. Chats run on the phone, even offline. The Learn tab turns photos and words into flashcards for Korean, Japanese, Chinese, Spanish, French, German and Portuguese. Sign in to sync chats with your Kelus account.',
    chips: ['Android 7.0+', 'Updates itself'],
    links: [{ label: 'Download the APK', href: 'https://github.com/KelusIDE/KelusModelB1/releases/latest', style: 'primary' }],
    note: 'Installed from GitHub, not the Play Store, so Android asks you to allow the install once.',
  },
  {
    id: 'cli', name: 'Kani CLI', kind: 'Terminal', status: 'soon', statusLabel: 'Coming soon',
    icon: { glyph: '>_' },
    description: 'Chat with Kani in your terminal and let it read, search and draft changes in your project, with every change shown as a diff first. No API keys and no dependencies.',
    chips: ['Node.js 18+', 'macOS', 'Linux', 'Windows'],
    links: [],
    note: 'Not public yet. It will be published on GitHub.',
  },
  {
    id: 'vscode', name: 'Kani for VS Code', kind: 'Editor extension', status: 'soon', statusLabel: 'Coming soon',
    icon: { glyph: '{ }' },
    description: 'Kani and Kani Painter in a VS Code side panel. Ask about your code or have it draw a picture, all running on your own computer.',
    chips: ['VS Code 1.85+'],
    links: [],
    note: 'Not public yet. It will be published on GitHub.',
  },
  {
    id: 'kel', name: 'Kel', kind: 'Programming language', status: 'soon', statusLabel: 'Coming soon',
    icon: { src: '/img/kel.svg' },
    description: <>Clean <code>{'{ }'}</code> code with friendly words like <code>say</code> and <code>repeat</code>, and mistakes that explain themselves. Typed like Rust and compiled to WebAssembly, so it runs fast in any browser.</>,
    chips: [],
    snippet: 'fn main {\n    let name = "Kel"\n    repeat 3 { say "Hi {name}!" }\n}',
    links: [{ label: 'Follow on GitHub', href: 'https://github.com/KelusIDE/Lus', style: 'secondary' }],
  },
];

export const MODELS: Model[] = [
  { name: 'Kani Otter 4.2', about: 'Our newest chat model. Powers Kani on the web, on Android and in the terminal.', stat: { count: 88, suffix: 'M parameters' } },
  { name: 'Kani Seal 3', about: 'Smaller and quicker. Pick it in Kani when speed matters more than depth.', stat: { count: 8.7, suffix: 'M parameters' } },
  { name: 'Cide', about: 'A coding specialist that Kani can hand programming tasks to. Preview in the CLI.', stat: { label: 'Preview' } },
  { name: 'Kani Painter', about: 'Turns a description into a drawing. In the CLI and the VS Code extension.', stat: { label: 'Drawing' } },
  { name: 'Kani Advanced', about: 'A larger model learning nine languages: English, Korean, Japanese, Simplified and Traditional Chinese, Spanish, French, German and Portuguese.', stat: { label: 'In training' }, training: true },
];


export const TICKER = ['Kelus IDE', 'Kani', 'Kani Otter 4.2', 'Kani for Android', 'Kel', 'Kani Seal 3', 'Cide', 'Kani Painter', 'Kani CLI', 'Kani for VS Code'];
