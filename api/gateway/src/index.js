



import MODELS from '../../models.json' with { type: 'json' };
import { AuthError, verifyIdToken } from './firebase.js';
import { authEmail, EmailError } from './email.js';

const KEY = /^hpk-[0-9A-Za-z]{40}$/;
const ALPHABET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';
const KEY_PAGE = 'https://kelus-b1.pages.dev/developers/';
const BODY_LIMIT = 256 * 1024;
const PAGES = ['https://kelus-b1.pages.dev', 'https://kelus-corp.pages.dev', 'https://kelus-b1.web.app', 'https://kelus-b1.firebaseapp.com', 'https://kelus-corp.web.app', 'https://kani.qd.je', 'https://www.kani.qd.je'];

class ApiError extends Error {
  constructor(status, message, code, headers = {}) { super(message); this.status = status; this.code = code; this.headers = headers; }
}


export function newKey() {
  let key = 'hpk-';
  while (key.length < 44) for (const byte of crypto.getRandomValues(new Uint8Array(48))) if (byte < 248 && key.length < 44) key += ALPHABET[byte % 62];
  return key;
}

export async function sha256(text) {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map(b => b.toString(16).padStart(2, '0')).join('');
}

const today = () => new Date().toISOString().slice(0, 10);
const numberVar = (value, fallback) => Number.isInteger(Number(value)) && Number(value) > 0 ? Number(value) : fallback;

function json(status, payload, headers = {}) {
  return new Response(status === 204 ? null : JSON.stringify(payload), { status, headers: { 'Content-Type': 'application/json; charset=utf-8', ...headers } });
}

function cors(request, env, path) {
  const origin = request.headers.get('Origin');

  if (!path.startsWith('/v1/keys') && !path.startsWith('/v1/auth/email')) return { 'Access-Control-Allow-Origin': '*' };
  return origin && (PAGES.includes(origin) || origin === env.DEV_ORIGIN) ? { 'Access-Control-Allow-Origin': origin, Vary: 'Origin' } : { Vary: 'Origin' };
}

async function readBody(request) {
  if (Number(request.headers.get('Content-Length')) > BODY_LIMIT) throw new ApiError(413, 'The request is too large.', 'request_too_large');
  const text = await request.text();
  if (text.length > BODY_LIMIT) throw new ApiError(413, 'The request is too large.', 'request_too_large');
  try { return { text, body: JSON.parse(text) }; } catch { throw new ApiError(400, 'The body is not valid JSON.', 'invalid_json'); }
}

function bearer(request) {
  const header = request.headers.get('Authorization') || '';
  return header.startsWith('Bearer ') ? header.slice(7).trim() : request.headers.get('x-api-key')?.trim() || '';
}

async function account(request, env) {
  const token = bearer(request);
  if (!token || token.startsWith('hpk-')) throw new ApiError(401, 'Sign in to your Kelus account to manage keys.', 'sign_in_required');
  const local = ['localhost', '127.0.0.1'].includes(new URL(request.url).hostname);
  let user;
  try { user = await verifyIdToken(token, env.FIREBASE_PROJECT || 'kelus-ide', { unsigned: local && env.FIREBASE_AUTH_EMULATOR === 'true' }); }
  catch (error) { throw error instanceof AuthError ? new ApiError(401, error.message, 'sign_in_required') : error; }
  if (!user.verified) throw new ApiError(403, 'Verify your email address first: open the link Kelus sent you, then reload this page.', 'email_not_verified');
  const { success } = await env.PER_ACCOUNT.limit({ key: user.uid });
  if (!success) throw new ApiError(429, 'Too many changes. Wait a minute.', 'rate_limit_exceeded', { 'Retry-After': '60' });
  return user;
}

async function keys(request, env, path) {
  const user = await account(request, env);
  const daily = numberVar(env.DAILY_REQUESTS, 300), most = numberVar(env.KEYS_PER_ACCOUNT, 3);
  if (request.method === 'GET' && path === '/v1/keys') {
    const { results } = await env.DB.prepare('SELECT id, name, hint, created, last_used, requests FROM api_keys WHERE owner = ? ORDER BY created DESC').bind(user.uid).all();
    const used = await env.DB.prepare('SELECT requests FROM api_usage WHERE owner = ? AND day = ?').bind(user.uid, today()).first();
    return json(200, { keys: results, today: used?.requests || 0, daily_limit: daily, keys_limit: most });
  }
  if (request.method === 'POST' && path === '/v1/keys') {
    const { body } = await readBody(request);
    const name = typeof body?.name === 'string' ? body.name.replace(/[\u0000-\u001f\u007f]/g, '').trim().slice(0, 40) : '';
    const { count } = await env.DB.prepare('SELECT COUNT(*) AS count FROM api_keys WHERE owner = ?').bind(user.uid).first();
    if (count >= most) throw new ApiError(409, `An account can have ${most} keys. Delete one first.`, 'too_many_keys');
    const key = newKey(), id = 'key_' + crypto.randomUUID().replaceAll('-', '').slice(0, 16), created = Date.now();
    const row = { id, name: name || 'My key', hint: key.slice(0, 8) + '…' + key.slice(-4), created, last_used: null, requests: 0 };
    await env.DB.prepare('INSERT INTO api_keys (id, hash, hint, owner, name, created, requests) VALUES (?, ?, ?, ?, ?, ?, 0)')
      .bind(id, await sha256(key), row.hint, user.uid, row.name, created).run();
    return json(201, { ...row, key });
  }
  const match = path.match(/^\/v1\/keys\/(key_[0-9a-f]{16})$/);
  if (request.method === 'DELETE' && match) {
    const { meta } = await env.DB.prepare('DELETE FROM api_keys WHERE id = ? AND owner = ?').bind(match[1], user.uid).run();
    if (!meta.changes) throw new ApiError(404, 'No such key.', 'not_found');
    return json(204);
  }
  throw new ApiError(404, 'Not found.', 'not_found');
}

async function complete(request, env, ctx) {
  const key = bearer(request);
  if (!key) throw new ApiError(401, `Send your API key as "Authorization: Bearer hpk-...". Make one at ${KEY_PAGE}`, 'missing_api_key');
  if (!KEY.test(key)) throw new ApiError(401, `That is not a Kani API key (they start with hpk-). Make one at ${KEY_PAGE}`, 'invalid_api_key');
  const found = await env.DB.prepare('SELECT id, owner FROM api_keys WHERE hash = ?').bind(await sha256(key)).first();
  if (!found) throw new ApiError(401, `Invalid or deleted API key. Make one at ${KEY_PAGE}`, 'invalid_api_key');
  const { text, body } = await readBody(request);
  const model = body?.model ?? MODELS[0].id;
  if (!MODELS.some(m => m.id === model)) throw new ApiError(404, `Unknown model "${String(model).slice(0, 80)}". Models: ${MODELS.map(m => m.id).join(', ')}.`, 'model_not_found');
  const { success } = await env.PER_KEY.limit({ key: found.id });
  if (!success) throw new ApiError(429, 'Rate limit: 10 requests a minute per key. Wait a moment.', 'rate_limit_exceeded', { 'Retry-After': '60' });

  const daily = numberVar(env.DAILY_REQUESTS, 300);
  const counted = await env.DB.prepare('INSERT INTO api_usage (owner, day, requests) VALUES (?, ?, 1) ON CONFLICT (owner, day) DO UPDATE SET requests = requests + 1 WHERE requests < ? RETURNING requests')
    .bind(found.owner, today(), daily).first();
  if (!counted) throw new ApiError(429, `Your account used today's ${daily} requests. More at 00:00 UTC.`, 'daily_limit_reached', { 'Retry-After': String(Math.ceil((Date.parse(today() + 'T00:00:00Z') + 86400000 - Date.now()) / 1000)) });
  ctx.waitUntil(env.DB.prepare('UPDATE api_keys SET requests = requests + 1, last_used = ? WHERE id = ?').bind(Date.now(), found.id).run());
  let upstream;
  try {
    upstream = await fetch(new URL('/v1/chat/completions', env.BACKEND_URL), { method: 'POST', body: text, signal: request.signal,
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${env.BACKEND_SECRET}` } });
  } catch {
    throw new ApiError(502, 'Kani is unavailable right now. Try again in a minute.', 'backend_unavailable');
  }
  const type = upstream.headers.get('Content-Type') || '';

  if (!/application\/json|text\/event-stream/.test(type)) throw new ApiError(503, 'Kani is waking up. Try again in a minute.', 'backend_starting', { 'Retry-After': '30' });
  if (upstream.status === 401) throw new ApiError(502, 'The Kani API is misconfigured. Kelus Corp. has been told.', 'backend_auth');
  return new Response(upstream.body, { status: upstream.status, headers: { 'Content-Type': type, 'Cache-Control': 'no-cache' } });
}

async function route(request, env, ctx, path) {
  if (request.method === 'OPTIONS') {
    return new Response(null, { status: 204, headers: { 'Access-Control-Allow-Methods': 'GET, POST, DELETE, OPTIONS',
      'Access-Control-Allow-Headers': 'Authorization, Content-Type, x-api-key', 'Access-Control-Max-Age': '86400' } });
  }
  if (request.method === 'GET' && path === '/') return json(200, { name: 'Kani API by Kelus Corp.', docs: KEY_PAGE, models: MODELS.map(m => m.id) });
  if (request.method === 'GET' && path === '/v1/models')
    return json(200, { object: 'list', data: MODELS.map(m => ({ id: m.id, object: 'model', owned_by: 'kelus-corp', name: m.name, about: m.about })) });
  if (request.method === 'POST' && path === '/v1/chat/completions') return complete(request, env, ctx);
  if (path.startsWith('/v1/auth/email/')) return authEmail(request, env, ctx, path);
  if (path === '/v1/keys' || path.startsWith('/v1/keys/')) return keys(request, env, path);
  throw new ApiError(404, 'Not found. Use POST /v1/chat/completions; docs at ' + KEY_PAGE, 'not_found');
}

export default {
  async fetch(request, env, ctx) {
    const path = new URL(request.url).pathname.replace(/\/+$/, '') || '/';
    let response;
    try {
      response = await route(request, env, ctx, path);
    } catch (error) {
      const known = error instanceof ApiError || error instanceof EmailError;
      if (!known) console.error(JSON.stringify({ path, error: path.startsWith('/v1/auth/email') ? 'Account email request failed' : error?.message }));
      response = json(known ? error.status : 500, { error: { message: known ? error.message : 'Something went wrong. Try again.',
        type: !known || error.status >= 500 ? 'server_error' : 'invalid_request_error', code: known ? error.code : 'server_error' } }, known ? error.headers : {});
    }
    for (const [name, value] of Object.entries(cors(request, env, path))) response.headers.set(name, value);
    return response;
  },
};
