import { useCallback, useRef, useState } from 'react';
import { readAttachment, validateBatch, type Attachment } from './engine';


const readPdf = async (bytes: Uint8Array) => (await import('./pdf')).readPdf(bytes);


export function useAttachments(onError: (message: string) => void) {
  const [files, setFiles] = useState<Attachment[]>([]);
  const [reading, setReading] = useState(false);
  const epoch = useRef(0);
  const current = useRef<Attachment[]>([]);
  current.current = files;

  const add = useCallback(async (selected: File[]) => {

    selected = selected.filter(file => !current.current.some(f => f.name === file.name && f.size === file.size));
    if (reading || !selected.length) return;
    const request = epoch.current;
    setReading(true);
    try {
      validateBatch([...current.current, ...selected]);
      const additions: Attachment[] = [];
      for (const file of selected) additions.push(await readAttachment(file, { readPdf }));
      if (request === epoch.current) setFiles(previous => [...previous, ...additions]);
    } catch (error) {
      if (request === epoch.current) onError((error as Error).message);
    } finally {
      setReading(false);
    }
  }, [reading, onError]);

  const remove = useCallback((index: number) => setFiles(previous => previous.filter((_, i) => i !== index)), []);
  const clear = useCallback(() => { epoch.current++; setFiles([]); }, []);
  return { files, reading, add, remove, clear };
}
