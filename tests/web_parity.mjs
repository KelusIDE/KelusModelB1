// Runs the browser engine in Node for tests/test_web.py.
// Usage: node tests/web_parity.mjs <model directory> <cases.json>
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { Model, prepareReply, localAnswer } from '../site/b1.js';

const [directory, casesFile] = process.argv.slice(2);
const manifest = JSON.parse(await readFile(join(directory, 'b1.json'), 'utf8'));
const weights = await readFile(join(directory, 'b1.bin'));
const model = new Model(manifest, weights.buffer.slice(weights.byteOffset, weights.byteOffset + weights.byteLength));
const cases = JSON.parse(await readFile(casesFile, 'utf8'));
const output = {
  encodings: cases.texts.map(text => model.tokenizer.encode(text)),
  replies: cases.conversations.map(messages => {
    try {
      const { retained, omitted, tool } = prepareReply(model, messages, cases.max_tokens);
      return { text: tool ? tool[0] : [...model.generate(retained, { maxTokens: cases.max_tokens })].join(''), source: tool ? tool[1] : 'model', omitted };
    } catch (error) {
      return { error: error.message };
    }
  }),
  helpers: cases.helpers.map(prompt => localAnswer(prompt)),
};
process.stdout.write(JSON.stringify(output));
