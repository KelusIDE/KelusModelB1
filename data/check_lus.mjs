// Checks Kel programs (.lus) with the real compiler in ~/Downloads/kel. Reads a JSON list of { id, code, answers } from the file in
// argv[2] and prints a JSON list of { id, ok, error, lines }. `answers` are what ask("...") gets, in order.
//   --run  also runs them, for programs written by data/make_lus.py. Without it nothing executes: kelus/lus_eval.py checks
//          the model's programs this way, because compile() only parses, type-checks and writes WebAssembly bytes.
import { readFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';

const { compile, run, formatError } = await import(join(homedir(), 'Downloads/kel/src/index.js'));
const programs = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const execute = process.argv.includes('--run');
const results = [];
for (const { id, code, answers = [] } of programs) {
  try {
    const bytes = compile(code);
    const lines = [];
    let next = 0;
    if (execute) await run(bytes, line => lines.push(line), () => answers[next++] ?? '');
    results.push({ id, ok: true, lines });
  } catch (error) {
    results.push({ id, ok: false, error: formatError(error, code, 'main.lus') });
  }
}
console.log(JSON.stringify(results));
