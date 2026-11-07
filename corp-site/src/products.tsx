import type { ReactNode } from "react";

export interface Link {
  label: string;
  href: string;
  style: "primary" | "secondary" | "text";
}
export interface Product {
  id: string;
  name: string;
  kind: string;
  group: "build" | "kani";
  status: "available" | "soon";
  statusLabel: string;
  icon: { src: string } | { glyph: string };
  description: ReactNode;
  chips: string[];
  links: Link[];
  note?: string;
  snippet?: string;
  featured?: boolean;
}
export interface Model {
  name: string;
  about: string;
  stat: { count: number; suffix: string } | { label: string };
  training?: boolean;
}

export const LOGO = "/img/kelus.png";
export const KANI_ICON = "/img/kani-otter.svg";
export const PRODUCTS: Product[] = [
  {
    id: "ide",
    name: "Kelus IDE",
    kind: "A place to build",
    group: "build",
    status: "available",
    statusLabel: "Available",
    featured: true,
    icon: { src: LOGO },
    description:
      "Your editor, terminal, and AI workspace. Explore a project, work through an idea, and review proposed changes before making them yours.",
    chips: ["Desktop", "Web", "Open source"],
    links: [
      {
        label: "Explore Kelus IDE",
        href: "https://ide.kelus.org",
        style: "primary",
      },
      {
        label: "Open web editor",
        href: "https://ide.kelus.org/ide.html",
        style: "text",
      },
    ],
  },
  {
    id: "kani",
    name: "Kani",
    kind: "Your own AI, on your device",
    group: "kani",
    status: "available",
    statusLabel: "Otter 4.12",
    featured: true,
    icon: { src: KANI_ICON },
    description:
      "A language model we train from scratch. Ask a question, explore a thought, or try an idea in code. The model runs right in your browser.",
    chips: ["Local inference", "No sign-up needed", "Free to try"],
    links: [
      {
        label: "Meet Kani",
        href: "https://kani.kelus.org",
        style: "primary",
      },
      {
        label: "See how it works",
        href: "https://github.com/KelusIDE/KelusModelB1",
        style: "text",
      },
    ],
    note: "Experimental. Answers can be wrong. Optional search sends queries online; signed-in chats can sync.",
  },
  {
    id: "android",
    name: "Kani, on the go",
    kind: "Mobile",
    group: "kani",
    status: "available",
    statusLabel: "v0.5.11",
    icon: { src: KANI_ICON },
    description:
      "Keep a conversation in your pocket. Run the packaged model on your phone, revisit saved chats, and practice with the Learn tools.",
    chips: ["Android APK", "iOS source"],
    links: [
      {
        label: "Get the Android app",
        href: "https://github.com/KelusIDE/KelusModelB1/releases/latest",
        style: "text",
      },
      {
        label: "iOS build guide",
        href: "https://github.com/KelusIDE/KelusModelB1/tree/main/mobile#ios",
        style: "text",
      },
    ],
    note: "Android updates ask before installing. iOS requires signing or sideload re-signing.",
  },
  {
    id: "cli",
    name: "Kani CLI",
    kind: "Terminal",
    group: "build",
    status: "available",
    statusLabel: "Open source",
    icon: { glyph: ">_" },
    description:
      "Stay in your flow. Chat with Kani, inspect project files, and prepare changes from the terminal. Your model runs on your machine.",
    chips: ["Node.js 22+", "macOS · Linux · Windows"],
    links: [
      {
        label: "Set up the CLI",
        href: "https://github.com/KelusIDE/KelusModelB1/tree/main/cli",
        style: "text",
      },
    ],
  },
  {
    id: "vscode",
    name: "Kani for VS Code",
    kind: "Editor extension",
    group: "build",
    status: "available",
    statusLabel: "Open source",
    icon: { glyph: "{ }" },
    description:
      "A conversation beside your code. Stream local replies, switch models, and copy or insert a snippet when you’re ready.",
    chips: ["VS Code 1.85+", "Local Python runtime"],
    links: [
      {
        label: "Get the extension source",
        href: "https://github.com/KelusIDE/KelusModelB1/tree/main/vscode-extension",
        style: "text",
      },
    ],
  },
  {
    id: "lus",
    name: "Lus",
    kind: "A language you can try here",
    group: "build",
    status: "available",
    statusLabel: "Playground",
    icon: { src: "/img/lus.svg" },
    description:
      "Friendly words, familiar braces, and real WebAssembly. Write a program, work with lists, and see the results in the playground below.",
    chips: ["Typed", "WebAssembly", "Experimental"],
    links: [
      {
        label: "Try the Lus playground",
        href: "#lus-playground",
        style: "text",
      },
    ],
  },
];

export const MODELS: Model[] = [
  {
    name: "Kani Otter 4.12",
    about:
      "Our current chat release. Runs locally in Kani’s browser and terminal clients.",
    stat: { count: 205.6, suffix: "M parameters" },
  },
  {
    name: "Cide 4.5",
    about:
      "A separate coding specialist. Try programming tasks and inspect what it produces.",
    stat: { count: 205.6, suffix: "M parameters" },
  },
  {
    name: "Kani Seal 3",
    about:
      "An earlier, smaller model. Still available when you want a lighter download.",
    stat: { count: 8.7, suffix: "M parameters" },
  },
];
export const TICKER = [
  "Think it",
  "Make it",
  "Try it",
  "Break it",
  "Learn from it",
  "Make it better",
];
