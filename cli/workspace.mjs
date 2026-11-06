
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
const exec = promisify(execFile);
const MAX_BYTES = 128 * 1024, MAX_FILES = 1500, MAX_SCAN = 8 * 1024 * 1024;
const SKIP = new Set(['.git', '.kani', '.env', '.ssh', '.aws', '.venv', 'venv', 'node_modules', '__pycache__', 'checkpoints', 'training-runs', 'dist', 'build', 'coverage', 'vendor']);
const TEXT = /\.(?:py|m?js|cjs|jsx|ts|tsx|java|c|h|cpp|cc|hpp|cs|php|go|rs|swift|kt|rb|sql|r|scala|pl|dart|lua|cob|cbl|ps1|sh|zsh|bash|html?|css|scss|svelte|vue|md|txt|json|ya?ml|toml|ini|cfg|xml)$/i;
const hash = text => createHash('sha256').update(text).digest('hex');
export const clean = text => String(text).replace(/[\x00-\x08\x0b-\x1f\x7f-\x9f]/g, '');
const privatePath = rel => rel.split(path.sep).some(part => SKIP.has(part.toLowerCase()) || /^(?:\.env(?:\.|$)|id_(?:rsa|ed25519)|credentials(?:\.|$)|secrets?(?:\.|$))/i.test(part)) || /\.(?:pem|key|p12|pfx)$/i.test(rel);
const textPath = rel => TEXT.test(rel) || /^(?:Dockerfile|Makefile|LICENSE|README|\.gitignore)$/i.test(path.basename(rel));

export class Workspace {
  static async open(root) {
    const absolute = await fs.realpath(path.resolve(root));
    if (!(await fs.stat(absolute)).isDirectory()) throw new Error('Workspace must be a directory.');
    return new Workspace(absolute);
  }
  constructor(root) { this.root = root; }
  async resolve(name, missing = false, internal = false) {
    if (!name || name.includes('\0') || /[\r\n]/.test(name)) throw new Error('Expected a file path.');
    const full = path.resolve(this.root, name), rel = path.relative(this.root, full);
    if (!rel || rel === '..' || rel.startsWith('..' + path.sep) || path.isAbsolute(rel)) throw new Error('Path must be inside the workspace.');
    if (!internal && privatePath(rel)) throw new Error('Private or generated paths are excluded.');

    let current = this.root;
    for (const part of rel.split(path.sep)) {
      current = path.join(current, part);
      try { if ((await fs.lstat(current)).isSymbolicLink()) throw new Error('Symlinks are excluded from workspace tools.'); }
      catch (error) { if (!(missing && error.code === 'ENOENT')) throw error; }
    }
    return { full, rel };
  }
  async read(name) {
    const { full, rel } = await this.resolve(name);
    const file = await fs.open(full, 'r');
    try {
      const info = await file.stat();
      if (!info.isFile() || info.size > MAX_BYTES) throw new Error(`Read requires a text file smaller than ${MAX_BYTES / 1024} KiB.`);
      const bytes = await file.readFile();
      if (bytes.length > MAX_BYTES || bytes.includes(0)) throw new Error('Binary or oversized file excluded.');
      const text = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(bytes);
      return { path: rel, text, hash: hash(bytes), mode: info.mode, lines: text.split('\n') };
    } finally { await file.close(); }
  }
  async files() {
    let names, truncated = false, source = 'git tracked + unignored files';
    try {
      const { stdout } = await exec('git', ['ls-files', '--cached', '--others', '--exclude-standard', '-z', '--', '.'], { cwd: this.root, timeout: 5000, maxBuffer: 2 * 1024 * 1024 });
      names = [...new Set(stdout.split('\0').filter(Boolean))];
    } catch {
      source = 'directory scan (Git unavailable; standard exclusions only)';
      names = [];
      let seen = 0;
      const walk = async (directory, depth = 0) => {
        if (depth > 12) { truncated = true; return; }
        const dir = await fs.opendir(path.join(this.root, directory));
        for await (const entry of dir) {
          if (++seen > 6000 || names.length >= MAX_FILES) { truncated = true; break; }
          const name = path.join(directory, entry.name);
          if (privatePath(name) || entry.isSymbolicLink()) continue;
          if (entry.isDirectory()) await walk(name, depth + 1);
          else if (entry.isFile() && textPath(name)) names.push(name);
        }
      };
      await walk('');
    }
    names = names.filter(name => !privatePath(name) && textPath(name)).sort();
    if (names.length > MAX_FILES) truncated = true;
    const files = [];
    for (const name of names.slice(0, MAX_FILES)) {
      try { await this.resolve(name); files.push(name); } catch {  }
    }
    return { files, truncated, source };
  }
  async search(query, limit = 30) {
    if (!query.trim()) throw new Error('Usage: /search literal text');
    const listing = await this.files(), hits = [];
    let bytes = 0, skipped = 0, truncated = listing.truncated;
    for (const name of listing.files) {
      let file;
      try { file = await this.read(name); } catch { skipped++; continue; }
      bytes += Buffer.byteLength(file.text);
      if (bytes > MAX_SCAN) { truncated = true; break; }
      for (let i = 0; i < file.lines.length; i++) {
        if (file.lines[i].toLowerCase().includes(query.toLowerCase())) {
          hits.push({ path: name, line: i + 1, text: file.lines[i].slice(0, 240) });
          if (hits.length >= limit) return { hits, truncated: true, skipped };
        }
      }
    }
    return { hits, truncated, skipped };
  }
  async analyze(name) {
    if (name) {
      const f = await this.read(name), symbols = [];
      f.lines.forEach((line, index) => {
        if (/^\s*(?:(?:export|async|public|private|static)\s+)*(?:def |class |function |interface |struct |enum |import |from |#include|func |fn |using )/.test(line)) symbols.push(`${f.path}:${index + 1} ${line.trim().slice(0, 180)}`);
      });
      return [`${f.path}: ${f.lines.length} lines, ${Buffer.byteLength(f.text)} bytes.`, 'Declarations and imports (text scan):', ...symbols.slice(0, 40), symbols.length > 40 ? 'Additional declarations omitted.' : '', !symbols.length ? 'No declarations recognized. Use /read to inspect the source.' : ''].filter(Boolean).join('\n');
    }
    const { files, truncated, source } = await this.files(), counts = new Map();
    for (const name of files) { const ext = path.extname(name) || '(no extension)'; counts.set(ext, (counts.get(ext) || 0) + 1); }
    const summary = [`Workspace: ${this.root}`, `${files.length}${truncated ? '+' : ''} source/config/document files · ${source}`, `File types: ${[...counts].sort((a,b) => b[1]-a[1]).slice(0, 12).map(([k,v]) => `${k} ${v}`).join(', ')}`, '', 'Entry points and project files:'];
    const entry = files.filter(f => /(^|\/)(?:package.json|pyproject.toml|Cargo.toml|go.mod|README.md|main\.[^/]+|index\.[^/]+|app\.[^/]+|cli\.[^/]+)$/.test(f));
    summary.push(...entry.slice(0, 30));
    for (const name of files.filter(f => /(^|\/)package.json$/.test(f)).slice(0, 4)) {
      try {
        const f = await this.read(name), pkg = JSON.parse(f.text);
        if (pkg.scripts) summary.push(`${name}: scripts ${Object.keys(pkg.scripts).join(', ')}`);
        if (pkg.bin) summary.push(`${name}: CLI ${JSON.stringify(pkg.bin)}`);
      } catch {  }
    }
    summary.push('', 'Use /read PATH, /analyze PATH or /search TEXT for source evidence.', 'This is a local file inventory; semantic reasoning depends on the loaded model.');
    return summary.join('\n');
  }
  async stage(name, content) {
    if (typeof content !== 'string' || !content.trim() || Buffer.byteLength(content) > MAX_BYTES) throw new Error('Proposal must contain a bounded, nonempty text file.');
    const target = await this.resolve(name, true);
    let before = null;
    try { before = await this.read(target.rel); } catch (e) { if (e.code !== 'ENOENT') throw e; }
    if (before?.text === content) throw new Error('The proposal makes no change.');
    const proposal = { id: randomUUID(), path: target.rel, baseHash: before?.hash ?? null, before: before?.text ?? null, content };
    const store = await this.resolve('.kani/proposals', true, true);
    await fs.mkdir(store.full, { recursive: true, mode: 0o700 });
    await fs.writeFile(path.join(store.full, proposal.id + '.json'), JSON.stringify(proposal), { flag: 'wx', mode: 0o600 });
    return proposal;
  }
  async proposal(id) {
    if (!/^[a-f0-9-]{36}$/.test(id)) throw new Error('Expected a proposal ID from /draft.');
    const store = await this.resolve(`.kani/proposals/${id}.json`, false, true);
    if ((await fs.stat(store.full)).size > 4 * MAX_BYTES) throw new Error('Oversized proposal.');
    const p = JSON.parse(await fs.readFile(store.full, 'utf8'));
    if (p.id !== id || typeof p.content !== 'string' || !p.content.trim() || Buffer.byteLength(p.content) > MAX_BYTES || (p.before !== null && typeof p.before !== 'string') || p.baseHash !== (p.before === null ? null : hash(p.before))) throw new Error('Invalid proposal.');
    await this.resolve(p.path, true);
    return p;
  }
  async apply(id) {
    const p = await this.proposal(id), target = await this.resolve(p.path, true);
    let current = null;
    try { current = await this.read(p.path); } catch (e) { if (e.code !== 'ENOENT') throw e; }
    if ((current?.hash ?? null) !== p.baseHash) throw new Error('File changed since the proposal. Generate a new draft; nothing was written.');
    await fs.mkdir(path.dirname(target.full), { recursive: true });
    const backup = await this.resolve(`.kani/backups/${id}`, true, true);
    await fs.mkdir(path.dirname(backup.full), { recursive: true, mode: 0o700 });
    if (current) await fs.writeFile(backup.full, current.text, { flag: 'wx', mode: 0o600 });
    const temporary = target.full + '.kani-' + randomUUID();
    try {
      await fs.writeFile(temporary, p.content, { flag: 'wx', mode: current ? current.mode & 0o777 : 0o644 });
      await this.resolve(p.path, true);
      if (current) {
        if ((await this.read(p.path)).hash !== p.baseHash) throw new Error('File changed during apply; original preserved.');
        await fs.rename(temporary, target.full);
      } else { await fs.link(temporary, target.full); }
    } finally { await fs.rm(temporary, { force: true }); }
    await this.discard(id);
    return `Wrote ${p.path}.${current ? ` Previous content: .kani/backups/${id}` : ''} Code has not been executed.`;
  }
  async discard(id) {
    await this.proposal(id);
    const store = await this.resolve(`.kani/proposals/${id}.json`, false, true);
    await fs.unlink(store.full);
  }
}


export function proposalDiff(p) {
  const before = (p.before || '').split('\n'), after = p.content.split('\n');
  let start = 0, end = 0;
  while (start < before.length && start < after.length && before[start] === after[start]) start++;
  while (end < before.length - start && end < after.length - start && before[before.length-1-end] === after[after.length-1-end]) end++;
  return clean([`--- ${p.before === null ? '/dev/null' : p.path}`, `+++ ${p.path}`, `@@ line ${start + 1} @@`, ...before.slice(start, before.length-end).map(x => '-' + x), ...after.slice(start, after.length-end).map(x => '+' + x)].join('\n'));
}

export function pathArgument(text) {
  const match = text.trim().match(/^(?:"([^"]+)"|'([^']+)'|(\S+))(?:\s+([\s\S]*))?$/);
  if (!match) throw new Error('Expected a file path (quote paths containing spaces).');
  return [match[1] || match[2] || match[3], match[4] || ''];
}

export function extractCode(text) {
  const blocks = [...text.matchAll(/^```[\w+#.-]*\s*\n([\s\S]*?)^```[ \t]*$/gm)];
  if (blocks.length !== 1 || !blocks[0][1].trim()) throw new Error('The model did not produce one complete code block. No file was staged. Try a smaller, precise request.');
  return blocks[0][1].replace(/\n?$/, '\n');
}
