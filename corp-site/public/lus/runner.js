import { compile, run, formatError } from './compiler/index.js';

const MAX_SOURCE = 32768;
const MAX_OUTPUT = 65536;
const MAX_LINES = 1000;

self.onmessage = async ({ data }) => {
  const { source, inputs } = data;
  if (typeof source !== 'string' || source.length > MAX_SOURCE || !Array.isArray(inputs)) {
    self.postMessage({ type: 'error', error: 'Use a program of up to 32,768 characters.' });
    return;
  }
  const lines = [];
  let size = 0, next = 0;
  try {
    const start = performance.now();
    const bytes = compile(source);
    const compiled = performance.now();
    await run(bytes, line => {
      size += line.length + 1;
      if (lines.length >= MAX_LINES || size > MAX_OUTPUT) throw new Error('Output limit reached. Print fewer lines and run again.');
      lines.push(line);
    }, question => {
      if (next >= inputs.length) throw new Error(`Input needed for "${question}". Add an answer in Program input, one answer per line, then run again.`);
      return inputs[next++];
    });
    self.postMessage({ type: 'result', lines, bytes: bytes.length, compileMs: compiled - start, runMs: performance.now() - compiled });
  } catch (error) {
    self.postMessage({ type: 'error', lines, error: formatError(error, source, 'main.lus').replace(/\bKel\b/g, 'Lus') });
  }
};
