import { importPKCS8, SignJWT } from 'jose';
import { AuthError, verifyIdToken } from './firebase.js';

const OAUTH = 'https://oauth2.googleapis.com/token';
const SCOPE = 'https://www.googleapis.com/auth/identitytoolkit';
const ORIGINS = new Set(['https://kelus-b1.pages.dev', 'https://kelus-b1.web.app', 'https://kelus-b1.firebaseapp.com', 'https://kani.qd.je', 'https://www.kani.qd.je']);
export class EmailError extends Error {
  constructor(status, message, code, headers = {}) { super(message); this.status = status; this.code = code; this.headers = headers; }
}
const reply = (status, data) => Response.json(data, { status, headers: { 'Cache-Control': 'no-store' } });
const fail = () => new EmailError(503, 'Account email is temporarily unavailable. Please try again later.', 'email_unavailable');
const accepted = () => reply(202, { message: 'If an account exists for that email, a reset link is on its way.' });
const hash = async value => [...new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(value)))].map(n => n.toString(16).padStart(2, '0')).join('');
const validEmail = email => typeof email === 'string' && email.length <= 254 && !/[\s\u0000-\u001f\u007f]/.test(email) && /^[^@]+@[^@]+\.[^@]+$/.test(email);
export const emailReady = env => env.EMAIL_ENABLED === 'true' && Boolean(env.BREVO_API_KEY && env.FIREBASE_SERVICE_ACCOUNT && validEmail(env.EMAIL_FROM_ADDRESS) && env.EMAIL_ACTION_URL);

export async function boundedJson(response, limit = 32768) {
  if (!response.body) throw new EmailError(400, 'A JSON body is required.', 'invalid_json');
  const reader = response.body.getReader(), chunks = [];
  let length = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    length += value.length;
    if (length > limit) { await reader.cancel(); throw new EmailError(413, 'The request is too large.', 'request_too_large'); }
    chunks.push(value);
  }
  const bytes = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  try { return JSON.parse(new TextDecoder().decode(bytes)); }
  catch { throw new EmailError(400, 'A valid JSON body is required.', 'invalid_json'); }
}

async function consume(env, key, limit, cooldown = 0) {
  const now = Date.now(), day = new Date(now).toISOString().slice(0, 10);
  return Boolean(await env.DB.prepare('INSERT INTO auth_email_limits (key, day, requests, last_sent) VALUES (?, ?, 1, ?) ON CONFLICT (key, day) DO UPDATE SET requests = requests + 1, last_sent = excluded.last_sent WHERE requests < ? AND last_sent <= ? RETURNING requests')
    .bind(key, day, now, limit, now - cooldown).first());
}

export async function actionLink(env, type, email, fetcher = fetch) {
  if (!['VERIFY_EMAIL', 'PASSWORD_RESET'].includes(type)) throw fail();
  let credentials;
  try { credentials = JSON.parse(env.FIREBASE_SERVICE_ACCOUNT); } catch { throw fail(); }
  const project = env.FIREBASE_PROJECT || 'kelus-ide';
  if (credentials.project_id !== project || !/^[^\s@]+@[^\s@]+\.gserviceaccount\.com$/.test(credentials.client_email || '') || typeof credentials.private_key !== 'string') throw fail();
  const key = await importPKCS8(credentials.private_key, 'RS256');
  const assertion = await new SignJWT({ scope: SCOPE }).setProtectedHeader({ alg: 'RS256', typ: 'JWT' })
    .setIssuer(credentials.client_email).setAudience(OAUTH).setIssuedAt().setExpirationTime('5m').sign(key);
  const tokenResponse = await fetcher(OAUTH, { method: 'POST', signal: AbortSignal.timeout(15000),
    body: new URLSearchParams({ grant_type: 'urn:ietf:params:oauth:grant-type:jwt-bearer', assertion }) });
  if (!tokenResponse.ok) { await tokenResponse.body?.cancel(); throw fail(); }
  const token = await boundedJson(tokenResponse);
  if (typeof token.access_token !== 'string' || !token.access_token) throw fail();
  const response = await fetcher(`https://identitytoolkit.googleapis.com/v1/projects/${encodeURIComponent(project)}/accounts:sendOobCode`, {
    method: 'POST', signal: AbortSignal.timeout(15000), headers: { Authorization: `Bearer ${token.access_token}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ requestType: type, email, returnOobLink: true, canHandleCodeInApp: false }),
  });
  if (!response.ok) { await response.body?.cancel(); throw fail(); }
  const data = await boundedJson(response);
  const mode = type === 'VERIFY_EMAIL' ? 'verifyEmail' : 'resetPassword';
  let generated, destination;
  try { generated = new URL(data.oobLink); destination = new URL(env.EMAIL_ACTION_URL); } catch { throw fail(); }
  if (generated.protocol !== 'https:' || generated.port || generated.username || generated.password || ![`${project}.firebaseapp.com`, `${project}.web.app`].includes(generated.hostname)
    || generated.pathname !== '/__/auth/action' || generated.searchParams.get('mode') !== mode || !generated.searchParams.get('oobCode')
    || !ORIGINS.has(destination.origin) || destination.pathname !== '/' || destination.search || destination.hash || destination.username || destination.password) throw fail();
  destination.searchParams.set('mode', mode);
  destination.searchParams.set('oobCode', generated.searchParams.get('oobCode'));
  return destination.href;
}

const escape = value => value.replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch]);
export function emailTemplate(kind, link) {
  const verify = kind === 'verify', title = verify ? 'Verify your email for Kani' : 'Reset your Kani password';
  const action = verify ? 'Verify email' : 'Reset password';
  const explanation = verify ? 'Confirm this email address to finish setting up your Kelus account.' : 'Use the link below to choose a new password for your Kelus account.';
  const ignore = verify ? 'If you did not create this account, you can ignore this email.' : 'If you did not request this reset, you can ignore this email. Your password will stay the same.';
  return { subject: title,
    text: `${title}\n\n${explanation}\n\n${action}: ${link}\n\n${ignore}\n\nKani by Kelus`,
    html: `<div style="font-family:Arial,sans-serif;max-width:560px;margin:32px auto;color:#172d31"><p>KANI · BY KELUS</p><h1>${title}</h1><p>${explanation}</p><p style="margin:32px 0"><a style="background:#164e48;color:white;padding:14px 22px;border-radius:10px;text-decoration:none" href="${escape(link)}">${action}</a></p><p>${ignore}</p><p>Or copy this link into your browser:<br><a href="${escape(link)}">${escape(link)}</a></p></div>` };
}

export async function deliverEmail(env, kind, email, fetcher = fetch) {
  const link = await actionLink(env, kind === 'verify' ? 'VERIFY_EMAIL' : 'PASSWORD_RESET', email, fetcher);
  const template = emailTemplate(kind, link);
  const response = await fetcher('https://api.brevo.com/v3/smtp/email', { method: 'POST', signal: AbortSignal.timeout(15000),
    headers: { 'api-key': env.BREVO_API_KEY, Accept: 'application/json', 'Content-Type': 'application/json' },
    body: JSON.stringify({ sender: { email: env.EMAIL_FROM_ADDRESS, name: env.EMAIL_FROM_NAME || 'Kani by Kelus' },
      to: [{ email }], subject: template.subject, htmlContent: template.html, textContent: template.text }),
  });
  if (!response.ok) { await response.body?.cancel(); throw fail(); }
  await response.body?.cancel();
}

export async function authEmail(request, env, ctx, path) {
  if (request.method === 'GET' && path === '/v1/auth/email/status') return reply(200, { enabled: emailReady(env) });
  if (request.method !== 'POST' || !['/v1/auth/email/verify', '/v1/auth/email/reset'].includes(path)) throw new EmailError(404, 'Not found.', 'not_found');
  const origin = request.headers.get('Origin');
  const local = ['localhost', '127.0.0.1'].includes(new URL(request.url).hostname);
  if (!origin || !(ORIGINS.has(origin) || (local && origin === env.DEV_ORIGIN))) throw new EmailError(403, 'This email request is not allowed.', 'origin_not_allowed');
  if (!emailReady(env)) throw fail();
  if (!request.headers.get('Content-Type')?.toLowerCase().startsWith('application/json')) throw new EmailError(415, 'Use a JSON request.', 'invalid_content_type');
  const body = await boundedJson(request, 4096);
  let email, user;
  const verify = path.endsWith('/verify');
  if (verify) {
    const header = request.headers.get('Authorization') || '';
    if (!header.startsWith('Bearer ')) throw new EmailError(401, 'Sign in before requesting a verification email.', 'sign_in_required');
    try { user = await verifyIdToken(header.slice(7), env.FIREBASE_PROJECT || 'kelus-ide', { unsigned: local && env.FIREBASE_AUTH_EMULATOR === 'true' }); }
    catch (error) { if (error instanceof AuthError) throw new EmailError(401, 'Sign in again to verify your email.', 'sign_in_required'); throw error; }
    email = user.email;
  } else { email = body?.email; }
  if (!validEmail(email)) throw new EmailError(400, 'Enter a valid email address.', 'invalid_email');
  email = email.toLowerCase();
  const ip = request.headers.get('CF-Connecting-IP') || 'unknown';
  if (!await consume(env, 'ip:' + await hash(ip), 20)) throw new EmailError(429, 'Too many email requests. Try again tomorrow.', 'email_rate_limited');
  const eligible = await consume(env, 'email:' + await hash(email), 8, 60000);
  if (!eligible) {
    if (!verify) return accepted();
    throw new EmailError(429, 'Wait a minute before requesting another verification email.', 'email_rate_limited', { 'Retry-After': '60' });
  }
  if (!await consume(env, 'global', 80)) throw new EmailError(429, 'Account email is busy. Please try again later.', 'email_rate_limited');
  ctx.waitUntil(env.DB.prepare('DELETE FROM auth_email_limits WHERE day < ?').bind(new Date().toISOString().slice(0, 10)).run());
  if (verify) {
    try { await deliverEmail(env, 'verify', email); } catch { throw fail(); }
    return reply(202, { message: 'Verification email requested. Check your inbox.' });
  }
  ctx.waitUntil(deliverEmail(env, 'reset', email).catch(() => console.error(JSON.stringify({ event: 'auth_email_failed', kind: 'reset' }))));
  return accepted();
}
