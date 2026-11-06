import catalog from './catalog.json' with { type: 'json' };

const RELEASE = `https://github.com/${catalog.repository}/releases/download/${catalog.tag}/`;
const DOWNLOAD_HOSTS = new Set(['github.com', 'release-assets.githubusercontent.com', 'objects.githubusercontent.com']);
const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
  'Access-Control-Allow-Headers': 'Range, If-None-Match',
  'Access-Control-Expose-Headers': 'Content-Length, Content-Range, ETag, X-Weights-Sha256',
  'X-Content-Type-Options': 'nosniff',
};
const failure = (status: number, message: string, headers: Record<string, string> = {}) => new Response(message, {
  status, headers: { ...CORS, 'Cache-Control': 'no-store', ...headers },
});

export async function serve(request: Request, fetcher: typeof fetch = fetch): Promise<Response> {
  const url = new URL(request.url);
  const match = url.pathname.match(/^\/v1\/([a-z0-9-]+)\/([a-f0-9]{64})\/b1\.bin$/);
  if (!match) return failure(404, 'Unknown model export.');
  const models: Record<string, typeof catalog.models['seal-3']> = catalog.models;
  const model = Object.hasOwn(models, match[1]) ? models[match[1]] : null;
  if (!model || model.sha256 !== match[2] || url.search) return failure(404, 'Unknown model export.');
  if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers: { ...CORS, 'Access-Control-Max-Age': '86400' } });
  if (!['GET', 'HEAD'].includes(request.method)) return failure(405, 'Use GET or HEAD.', { Allow: 'GET, HEAD, OPTIONS' });
  const etag = `"${model.sha256}"`;
  const headers = new Headers({ ...CORS, 'Content-Type': 'application/octet-stream', 'Content-Length': String(model.bytes),
    'Cache-Control': 'public, max-age=31536000, immutable', ETag: etag, 'X-Weights-Sha256': model.sha256, 'Accept-Ranges': 'bytes' });
  if ((request.headers.get('If-None-Match') || '').split(',').some(value => value.trim().replace(/^W\//, '') === etag || value.trim() === '*')) {
    headers.delete('Content-Length');
    return new Response(null, { status: 304, headers });
  }
  if (request.method === 'HEAD') return new Response(null, { headers });
  const range = request.headers.get('Range');
  if (range) {
    const parsed = range.match(/^bytes=(\d+)-(\d*)$/);
    if (!parsed || Number(parsed[1]) >= model.bytes || (parsed[2] && (Number(parsed[2]) < Number(parsed[1]) || Number(parsed[2]) >= model.bytes))) {
      return failure(416, 'Invalid byte range.', { 'Content-Range': `bytes */${model.bytes}` });
    }
  }
  let upstreamUrl = RELEASE + model.asset;
  try {
    for (let attempt = 0; attempt < 4; attempt++) {
      const target = new URL(upstreamUrl);
      if (target.protocol !== 'https:' || !DOWNLOAD_HOSTS.has(target.hostname) || target.username || target.password) {
        return failure(502, 'Unexpected model download destination.');
      }
      const upstream = await fetcher(upstreamUrl, {
        redirect: 'manual', headers: { 'User-Agent': 'Kelus-Kani-Model-Downloads', ...(range ? { Range: range } : {}) },
      });
      if ([301, 302, 303, 307, 308].includes(upstream.status)) {
        const next = upstream.headers.get('Location');
        await upstream.body?.cancel();
        if (!next) return failure(502, 'Model download redirect is missing.');
        upstreamUrl = new URL(next, upstreamUrl).href;
        continue;
      }
      if (![200, 206].includes(upstream.status) || !upstream.body) {
        await upstream.body?.cancel();
        return failure(502, 'GitHub model download is temporarily unavailable.');
      }
      const length = upstream.headers.get('Content-Length');
      if (upstream.status === 200 && length && Number(length) !== model.bytes) {
        await upstream.body.cancel();
        return failure(502, 'GitHub model export has an unexpected size.');
      }
      if (upstream.status === 206) {
        const contentRange = upstream.headers.get('Content-Range');
        if (!range || !contentRange || !length) { await upstream.body.cancel(); return failure(502, 'Invalid partial model response.'); }
        headers.set('Content-Range', contentRange);
        headers.set('Content-Length', length);
      }
      return new Response(upstream.body, { status: upstream.status, headers });
    }
    return failure(502, 'Too many model download redirects.');
  } catch {
    return failure(502, 'Could not reach the model download host. Try again.');
  }
}

export default { fetch: request => serve(request) } satisfies ExportedHandler<Env>;
