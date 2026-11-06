#!/usr/bin/env node




import { copyFile, mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ENGINE = ['b1.js', 'web.js', 'support.js', 'kernels.js', 'arithmetic.js', 'reading.js', 'recipes.js', 'places.js', 'cide.js'];
const out = process.argv[2];
if (!out) { console.error('Usage: node api/make_space.mjs OUT'); process.exit(1); }

const MODELS = JSON.parse(await readFile(path.join(ROOT, 'api/models.json'), 'utf8'));
const copy = async file => { await mkdir(path.dirname(path.join(out, file)), { recursive: true }); await copyFile(path.join(ROOT, file), path.join(out, file)); };
for (const file of ['api/models.json', 'api/backend/server.mjs', 'api/backend/reply.mjs', ...ENGINE.map(f => 'site/' + f)]) await copy(file);
let bytes = 0;
for (const model of MODELS) {
  for (const file of ['b1.json', 'b1.bin']) await copy(`site/${model.directory}/${file}`);
  bytes += (await stat(path.join(ROOT, 'site', model.directory, 'b1.bin'))).size;
}
await writeFile(path.join(out, 'Dockerfile'), `FROM node:22-slim
WORKDIR /app
COPY --chown=node . .
USER node
ENV PORT=7860
EXPOSE 7860
CMD ["node", "api/backend/server.mjs"]
`);
await writeFile(path.join(out, 'README.md'), `---
title: Kani API backend
emoji: 🦦
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

The backend of the Kani API by Kelus Corp. It answers only the API gateway, which needs the KANI_BACKEND_SECRET
set as a secret of this Space. Developers: get an hpk- key and the docs at https://kelus-b1.pages.dev/developers/
`);
console.log(`${out}: ${MODELS.length} models, ${(bytes / 1e6).toFixed(0)} MB of weights`);
