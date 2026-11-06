

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { DatabaseSync } from 'node:sqlite';
import worker, { newKey, sha256 } from '../src/index.js';

const SCHEMA = readFileSync(new URL('../schema.sql', import.meta.url), 'utf8');
const BACKEND = 'https://backend.test';
const pair = await crypto.subtle.generateKey({ name: 'RSASSA-PKCS1-v1_5', modulusLength: 2048, publicExponent: new Uint8Array([1, 0, 1]), hash: 'SHA-256' }, true, ['sign', 'verify']);
const JWK = { ...await crypto.subtle.exportKey('jwk', pair.publicKey), kid: 'test-kid', alg: 'RS256', use: 'sig' };


function database() {
  const db = new DatabaseSync(':memory:');
  db.exec(SCHEMA);
  const statement = (sql, args = []) => ({
    bind: (...values) => statement(sql, values),
    first: async () => db.prepare(sql).get(...args) ?? null,
    run: async () => ({ meta: { changes: db.prepare(sql).run(...args).changes } }),
    all: async () => ({ results: db.prepare(sql).all(...args).map(row => ({ ...row })) }),
  });
  return { prepare: sql => statement(sql), db };
}

function limiter(limit) {
  const counts = new Map();
  return { limit: async ({ key }) => { counts.set(key, (counts.get(key) || 0) + 1); return { success: counts.get(key) <= limit }; } };
}

function environment(overrides = {}) {
  return { DB: database(), PER_KEY: limiter(10), PER_ACCOUNT: limiter(20), BACKEND_URL: BACKEND, BACKEND_SECRET: 'backend-secret',
    FIREBASE_PROJECT: 'kelus-ide', DAILY_REQUESTS: '300', KEYS_PER_ACCOUNT: '3', ...overrides };
}

const encode = value => Buffer.from(typeof value === 'string' ? value : JSON.stringify(value)).toString('base64url');
async function idToken(claims = {}, { kid = 'test-kid', key = pair.privateKey } = {}) {
  const now = Math.floor(Date.now() / 1000);
  const body = encode({ alg: 'RS256', kid, typ: 'JWT' }) + '.' + encode({ iss: 'https://securetoken.google.com/kelus-ide', aud: 'kelus-ide', sub: 'user-1',
    iat: now - 10, auth_time: now - 10, exp: now + 3600, email: 'dev@example.com', email_verified: true, firebase: { sign_in_provider: 'password' }, ...claims });
  const signature = await crypto.subtle.sign('RSASSA-PKCS1-v1_5', key, new TextEncoder().encode(body));
  return body + '.' + Buffer.from(signature).toString('base64url');
}


function network(t, reply = () => new Response(JSON.stringify({ object: 'chat.completion', choices: [{ message: { content: 'Hi!' } }] }), { headers: { 'Content-Type': 'application/json' } })) {
  const sent = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    url = String(url);
    if (url.includes('securetoken@system.gserviceaccount.com')) return new Response(JSON.stringify({ keys: [JWK] }), { headers: { 'Cache-Control': 'max-age=3600' } });
    sent.push({ url, options });
    return reply(url, options);
  });
  return sent;
}

const ctx = () => { const waiting = []; return { waitUntil: promise => waiting.push(promise), done: () => Promise.all(waiting) }; };
const call = async (env, method, path, { token, body, origin, raw } = {}) => {
  const context = ctx();
  const response = await worker.fetch(new Request('https://kani-api.test' + path, { method, body: raw ?? (body === undefined ? undefined : JSON.stringify(body)),
    headers: { ...(token && { Authorization: `Bearer ${token}` }), ...(origin && { Origin: origin }), 'Content-Type': 'application/json' } }), env, context);
  await context.done();
  return response;
};
const PAGE = 'https://kelus-b1.web.app';

async function makeKey(env, name = 'test') {
  const response = await call(env, 'POST', '/v1/keys', { token: await idToken(), body: { name }, origin: PAGE });
  assert.equal(response.status, 201);
  return response.json();
}

test('keys start with hpk- and 40 letters and digits, and differ every time', () => {
  const keys = new Set(Array.from({ length: 200 }, newKey));
  assert.equal(keys.size, 200);
  for (const key of keys) assert.match(key, /^hpk-[0-9A-Za-z]{40}$/);
});

test('a signed-in account makes a key, sees it listed by hint only, and deletes it', async t => {
  network(t);
  const env = environment();
  const made = await makeKey(env, 'My app');
  assert.match(made.key, /^hpk-[0-9A-Za-z]{40}$/);
  assert.equal(made.hint, made.key.slice(0, 8) + '…' + made.key.slice(-4));
  assert.equal(made.name, 'My app');
  const stored = env.DB.db.prepare('SELECT * FROM api_keys').all();
  assert.equal(stored.length, 1);
  assert.equal(stored[0].hash, await sha256(made.key));
  assert.ok(!JSON.stringify(stored).includes(made.key), 'the key itself is never stored');

  const listed = await call(env, 'GET', '/v1/keys', { token: await idToken(), origin: PAGE });
  assert.equal(listed.headers.get('Access-Control-Allow-Origin'), PAGE);
  const page = await listed.json();
  assert.deepEqual(page.keys.map(k => [k.id, k.name, k.hint]), [[made.id, 'My app', made.hint]]);
  assert.ok(!JSON.stringify(page).includes(made.key));
  assert.equal(page.daily_limit, 300);

  const other = await call(env, 'DELETE', '/v1/keys/' + made.id, { token: await idToken({ sub: 'someone-else' }), origin: PAGE });
  assert.equal(other.status, 404, "another account cannot delete it");
  assert.equal((await call(env, 'DELETE', '/v1/keys/' + made.id, { token: await idToken(), origin: PAGE })).status, 204);
  const gone = await call(env, 'POST', '/v1/chat/completions', { token: made.key, body: { messages: [{ role: 'user', content: 'hi' }] } });
  assert.equal(gone.status, 401);
});

test('key management needs a real, verified Kelus sign-in from a Kani page', async t => {
  network(t);
  const env = environment();
  const other = await crypto.subtle.generateKey({ name: 'RSASSA-PKCS1-v1_5', modulusLength: 2048, publicExponent: new Uint8Array([1, 0, 1]), hash: 'SHA-256' }, true, ['sign']);
  const cases = [
    [undefined, 401], ['not.a.token', 401], [await idToken({}, { key: other.privateKey }), 401], [await idToken({ aud: 'other-project', iss: 'https://securetoken.google.com/other-project' }), 401],
    [await idToken({ exp: Math.floor(Date.now() / 1000) - 5 }), 401], [await idToken({}, { kid: 'unknown' }), 401], [await idToken({ email_verified: false }), 403],
    [newKey(), 401],
  ];
  for (const [token, status] of cases) assert.equal((await call(env, 'GET', '/v1/keys', { token, origin: PAGE })).status, status, String(token).slice(0, 30));
  const unsigned = encode({ alg: 'none' }) + '.' + encode({ iss: 'https://securetoken.google.com/kelus-ide', aud: 'kelus-ide', sub: 'x', iat: 1, auth_time: 1, exp: 9e9 }) + '.';
  assert.equal((await call(environment({ FIREBASE_AUTH_EMULATOR: 'true' }), 'GET', '/v1/keys', { token: unsigned })).status, 401, 'emulator tokens only on localhost');
  assert.equal((await call(env, 'GET', '/v1/keys', { token: await idToken({ email_verified: false, firebase: { sign_in_provider: 'google.com' } }) })).status, 200);
  const foreign = await call(env, 'GET', '/v1/keys', { token: await idToken(), origin: 'https://evil.example' });
  assert.equal(foreign.headers.get('Access-Control-Allow-Origin'), null);
});

test('an account has at most KEYS_PER_ACCOUNT keys', async t => {
  network(t);
  const env = environment({ KEYS_PER_ACCOUNT: '2' });
  await makeKey(env);
  await makeKey(env);
  const third = await call(env, 'POST', '/v1/keys', { token: await idToken(), body: {}, origin: PAGE });
  assert.equal(third.status, 409);
  assert.equal((await third.json()).error.code, 'too_many_keys');
});

test('a valid key reaches the backend with the backend secret, the body unchanged, and the answer passed back', async t => {
  const sent = network(t);
  const env = environment();
  const { key, id } = await makeKey(env);
  const body = { model: 'kani-seal-3', messages: [{ role: 'user', content: 'hi' }], max_tokens: 20 };
  const response = await call(env, 'POST', '/v1/chat/completions', { token: key, body });
  assert.equal(response.status, 200);
  assert.equal(response.headers.get('Access-Control-Allow-Origin'), '*');
  assert.equal((await response.json()).choices[0].message.content, 'Hi!');
  assert.equal(sent.length, 1);
  assert.equal(sent[0].url, BACKEND + '/v1/chat/completions');
  assert.equal(sent[0].options.headers.Authorization, 'Bearer backend-secret');
  assert.deepEqual(JSON.parse(sent[0].options.body), body);
  const row = env.DB.db.prepare('SELECT requests, last_used FROM api_keys WHERE id = ?').get(id);
  assert.equal(row.requests, 1);
  assert.ok(row.last_used > 0);
});

test('streams pass straight through', async t => {
  network(t, () => new Response('data: {"choices":[{"delta":{"content":"Hi"}}]}\n\ndata: [DONE]\n\n', { headers: { 'Content-Type': 'text/event-stream' } }));
  const env = environment();
  const { key } = await makeKey(env);
  const response = await call(env, 'POST', '/v1/chat/completions', { token: key, body: { messages: [{ role: 'user', content: 'hi' }], stream: true } });
  assert.match(response.headers.get('Content-Type'), /event-stream/);
  assert.match(await response.text(), /\[DONE\]/);
});

test('bad keys, unknown models and broken bodies never reach the backend', async t => {
  const sent = network(t);
  const env = environment();
  const { key } = await makeKey(env);
  const messages = [{ role: 'user', content: 'hi' }];
  const cases = [
    [undefined, { messages }, 401, 'missing_api_key'], ['sk-123', { messages }, 401, 'invalid_api_key'], [newKey(), { messages }, 401, 'invalid_api_key'],
    [key, { model: 'gpt-4o', messages }, 404, 'model_not_found'],
  ];
  for (const [token, body, status, code] of cases) {
    const response = await call(env, 'POST', '/v1/chat/completions', { token, body });
    assert.equal(response.status, status);
    assert.equal((await response.json()).error.code, code);
  }
  assert.equal((await call(env, 'POST', '/v1/chat/completions', { token: key, raw: '{' })).status, 400);
  assert.equal((await call(env, 'POST', '/v1/chat/completions', { token: key, raw: JSON.stringify({ messages: [{ role: 'user', content: 'x'.repeat(300000) }] }) })).status, 413);
  assert.equal(sent.length, 0);
});

test('the per-minute and daily limits stop requests with 429', async t => {
  network(t);
  const env = environment({ DAILY_REQUESTS: '3' });
  const { key } = await makeKey(env);
  const ask = () => call(env, 'POST', '/v1/chat/completions', { token: key, body: { messages: [{ role: 'user', content: 'hi' }] } });
  for (let i = 0; i < 3; i++) assert.equal((await ask()).status, 200);
  const over = await ask();
  assert.equal(over.status, 429);
  assert.equal((await over.json()).error.code, 'daily_limit_reached');
  assert.ok(Number(over.headers.get('Retry-After')) > 0);
  const listed = await (await call(env, 'GET', '/v1/keys', { token: await idToken(), origin: PAGE })).json();
  assert.equal(listed.today, 3);

  const busy = environment();
  const second = await makeKey(busy);
  for (let i = 0; i < 10; i++) await call(busy, 'POST', '/v1/chat/completions', { token: second.key, body: { messages: [{ role: 'user', content: 'hi' }] } });
  const limited = await call(busy, 'POST', '/v1/chat/completions', { token: second.key, body: { messages: [{ role: 'user', content: 'hi' }] } });
  assert.equal(limited.status, 429);
  assert.equal((await limited.json()).error.code, 'rate_limit_exceeded');
});

test('a sleeping or unreachable backend gives a clear 503 or 502', async t => {
  const env = environment();
  network(t);
  const { key } = await makeKey(env);
  t.mock.restoreAll();
  network(t, () => new Response('<html>Space is building</html>', { status: 503, headers: { 'Content-Type': 'text/html' } }));
  const waking = await call(env, 'POST', '/v1/chat/completions', { token: key, body: { messages: [{ role: 'user', content: 'hi' }] } });
  assert.equal(waking.status, 503);
  assert.equal((await waking.json()).error.code, 'backend_starting');
  t.mock.restoreAll();
  network(t, () => { throw new TypeError('fetch failed'); });
  const down = await call(env, 'POST', '/v1/chat/completions', { token: key, body: { messages: [{ role: 'user', content: 'hi' }] } });
  assert.equal(down.status, 502);
});

test('the model list and preflight need no key', async () => {
  const env = environment();
  const models = await (await call(env, 'GET', '/v1/models')).json();
  assert.deepEqual(models.data.map(m => m.id), ['kani-otter-4.3', 'cide-2', 'kani-seal-3']);
  const preflight = await call(env, 'OPTIONS', '/v1/chat/completions', { origin: 'https://someone.example' });
  assert.equal(preflight.status, 204);
  assert.equal(preflight.headers.get('Access-Control-Allow-Origin'), '*');
  assert.match(preflight.headers.get('Access-Control-Allow-Headers'), /Authorization/);
});
