

import * as pdfjs from 'pdfjs-dist';
import type { TextItem } from 'pdfjs-dist/types/src/display/api';
import workerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url';
import { MAX_PAGES, type PdfText } from './engine';

const MAX_CHARACTERS = 2_000_000;


export function pageText(items: TextItem[]): string {
  let text = '', last: TextItem | null = null;
  for (const item of items) {
    if (last) {
      const size = Math.max(2, Math.abs(last.height) || Math.abs(last.transform[3]) || 10);
      if (Math.abs(item.transform[5] - last.transform[5]) > size * 0.5) text += '\n';
      else if (item.transform[4] - (last.transform[4] + last.width) > size * 0.15 && !/\s$/.test(text) && !/^\s/.test(item.str)) text += ' ';
    }
    text += item.str;
    last = item.hasEOL ? null : item;
    if (item.hasEOL) text += '\n';
  }
  return text;
}

export async function readPdf(bytes: Uint8Array): Promise<PdfText> {
  pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;

  const task = pdfjs.getDocument({ data: bytes.slice(), disableFontFace: true, enableXfa: false, useWasm: false });
  let document;
  try { document = await task.promise; }
  catch (error) {
    if ((error as Error).name === 'PasswordException') throw new Error('This PDF is password-protected. Open it, save a copy without a password, and attach that.');
    throw new Error('This PDF could not be read. It may be damaged; try saving it again.');
  }
  try {
    const pages: string[] = [];
    let characters = 0;
    for (let n = 1; n <= Math.min(document.numPages, MAX_PAGES) && characters < MAX_CHARACTERS; n++) {
      const page = await document.getPage(n);
      const content = await page.getTextContent();
      const text = pageText(content.items.filter((item): item is TextItem => 'str' in item));
      pages.push(text);
      characters += text.length;
      page.cleanup();
    }
    const metadata = await document.getMetadata().catch(() => null);
    const info = metadata?.info as { Title?: unknown; Author?: unknown } | undefined;
    const text = (value: unknown) => typeof value === 'string' && value.trim() ? value.trim() : null;
    return { title: text(info?.Title), author: text(info?.Author), pages, total: document.numPages };
  } finally {
    await task.destroy();
  }
}
