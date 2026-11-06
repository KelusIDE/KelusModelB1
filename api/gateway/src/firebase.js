

const KEYS_URL = 'https://www.googleapis.com/service_accounts/v1/jwk/securetoken@system.gserviceaccount.com';
const SKEW = 300;
let cache = { keys: [], until: 0 };

export class AuthError extends Error {}

const decode = part => Uint8Array.from(atob(part.replace(/-/g, '+').replace(/_/g, '/')), c => c.charCodeAt(0));
const parse = part => { try { return JSON.parse(new TextDecoder().decode(decode(part))); } catch { throw new AuthError('Malformed sign-in token.'); } };

async function signingKeys(fresh) {
  if (!fresh && Date.now() < cache.until) return cache.keys;
  const response = await fetch(KEYS_URL);
  if (!response.ok) throw new Error('Could not load Google sign-in keys.');
  const age = Number(/max-age=(\d+)/.exec(response.headers.get('Cache-Control') || '')?.[1]) || 3600;
  cache = { keys: (await response.json()).keys || [], until: Date.now() + age * 1000 };
  return cache.keys;
}






export async function verifyIdToken(token, project, { unsigned = false } = {}) {
  const parts = String(token).split('.');
  if (parts.length !== 3) throw new AuthError('Malformed sign-in token.');
  const header = parse(parts[0]), claims = parse(parts[1]);
  if (!(unsigned && header.alg === 'none')) {
    if (header.alg !== 'RS256' || typeof header.kid !== 'string') throw new AuthError('Unexpected sign-in token.');
    let jwk = (await signingKeys(false)).find(k => k.kid === header.kid);
    if (!jwk) jwk = (await signingKeys(true)).find(k => k.kid === header.kid);
    if (!jwk) throw new AuthError('Unknown signing key. Sign in again.');
    const key = await crypto.subtle.importKey('jwk', { kty: jwk.kty, n: jwk.n, e: jwk.e, alg: 'RS256', ext: true },
      { name: 'RSASSA-PKCS1-v1_5', hash: 'SHA-256' }, false, ['verify']);
    let signature;
    try { signature = decode(parts[2]); } catch { throw new AuthError('Malformed sign-in token.'); }
    if (!await crypto.subtle.verify('RSASSA-PKCS1-v1_5', key, signature, new TextEncoder().encode(parts[0] + '.' + parts[1])))
      throw new AuthError('The sign-in token signature is wrong.');
  }
  const now = Date.now() / 1000;
  if (claims.aud !== project || claims.iss !== `https://securetoken.google.com/${project}`) throw new AuthError('This sign-in token is for another app.');
  if (typeof claims.sub !== 'string' || !claims.sub || claims.sub.length > 128) throw new AuthError('The sign-in token has no user.');
  if (!(claims.exp > now)) throw new AuthError('Your sign-in expired. Refresh the page.');
  if (!(claims.iat <= now + SKEW) || !(claims.auth_time <= now + SKEW)) throw new AuthError('The sign-in token is not valid yet.');
  const provider = claims.firebase?.sign_in_provider;
  return { uid: claims.sub, email: typeof claims.email === 'string' ? claims.email : null,
    verified: claims.email_verified === true || (provider !== 'password' && provider !== 'anonymous' && Boolean(provider)) };
}
