import { readFile, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url)), root = resolve(here, '..');
const catalog = JSON.parse(await readFile(join(here, 'catalog.json'), 'utf8'));
const host = 'https://kani-models.chatgpt-ai-5-2o2.workers.dev';
const target = id => `${host}/v1/${id}/${catalog.models[id].sha256}/b1.bin`;
const redirects = Object.entries(catalog.models).map(([id, model]) => ({ source: `/${model.directory}/b1.bin`, destination: target(id), type: 302 }));
const file = join(root, 'firebase.json');
const firebase = JSON.parse(await readFile(file, 'utf8'));
firebase.hosting.ignore = [...new Set([...firebase.hosting.ignore, '**/b1.bin'])];
firebase.hosting.redirects = [
  ...(firebase.hosting.redirects || []).filter(route => !redirects.some(next => next.source === route.source)), ...redirects,
];
for (const rule of firebase.hosting.headers) for (const header of rule.headers) {
  if (header.key === 'Content-Security-Policy' && header.value.includes("connect-src 'self'")) {
    if (!header.value.includes(host)) header.value = header.value.replace("connect-src 'self'", `connect-src 'self' ${host}`);
  }
}
await writeFile(file, JSON.stringify(firebase, null, 2) + '\n');
const pagesHeaders = join(root, 'site/_headers');
let headers = await readFile(pagesHeaders, 'utf8');
if (!headers.includes(host)) headers = headers.replace("connect-src 'self'", `connect-src 'self' ${host}`);
await writeFile(pagesHeaders, headers);
await writeFile(join(root, 'site/_redirects'), '# Checksum-pinned model exports hosted in GitHub Releases.\n'
  + redirects.map(route => `${route.source} ${route.destination} ${route.type}`).join('\n') + '\n');

if (process.argv[2]) {
  const ide = resolve(process.argv[2]), id = 'otter-4-11', source = `/kani/${id}/b1.bin`;
  const ideFile = join(ide, 'firebase.json'), config = JSON.parse(await readFile(ideFile, 'utf8'));
  const hosting = config.hosting.find(site => site.site === 'kelus-ide');
  hosting.ignore = [...new Set([...hosting.ignore, '**/b1.bin', '_headers', '_redirects'])];
  hosting.redirects = [...(hosting.redirects || []).filter(route => route.source !== source), { source, destination: target(id), type: 302 }];
  await writeFile(ideFile, JSON.stringify(config, null, 2) + '\n');
  await writeFile(join(ide, 'public/_redirects'), `# Checksum-pinned model export hosted in GitHub Releases.\n${source} ${target(id)} 302\n`);
}
console.log('Configured GitHub-backed model routes.');
