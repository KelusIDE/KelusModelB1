















const UPSTREAM = 'https://api.langsearch.com/v1/web-search';
const PLACES_UPSTREAM = 'https://places.googleapis.com/v1/places:searchText';
const MAPBOX_UPSTREAM = 'https://api.mapbox.com/search/searchbox/v1/forward';
const MAPBOX_STATIC = 'https://api.mapbox.com/styles/v1/mapbox/streets-v12/static';
const WIKIPEDIA = 'https://en.wikipedia.org/w/api.php';

const WIKIPEDIA_AGENT = 'KaniPlaces/1.0 (https://kelus-b1.pages.dev; Kelus Corp.)';

const PLACES_FIELDS = 'places.displayName,places.formattedAddress,places.rating,places.userRatingCount,places.googleMapsUri,places.currentOpeningHours,places.priceLevel';
const PRICES = { PRICE_LEVEL_FREE: 'Free', PRICE_LEVEL_INEXPENSIVE: '$', PRICE_LEVEL_MODERATE: '$$', PRICE_LEVEL_EXPENSIVE: '$$$', PRICE_LEVEL_VERY_EXPENSIVE: '$$$$' };
const PLACES = 5;
const CANDIDATES = 10;

const ALLOWED_ORIGINS = new Set(['https://kelus-b1.pages.dev', 'https://kelus-b1.web.app', 'https://kelus-b1.firebaseapp.com', 'https://localhost', 'capacitor://localhost']);
const MAX_BODY = 2048;
const MAX_QUERY = 300;
const RESULTS = 5;
const MAX_TEXT = 3000;
const MAX_UPSTREAM = 512 * 1024;

function reply(body, status, origin, extra = {}) {
  const headers = { 'Content-Type': 'application/json', 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff', ...extra };
  if (origin) Object.assign(headers, { 'Access-Control-Allow-Origin': origin, Vary: 'Origin' });
  return new Response(JSON.stringify(body), { status, headers });
}


async function readLimited(body, limit) {
  const reader = body.getReader();
  const chunks = [];
  let size = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    size += value.byteLength;
    if (size > limit) {
      await reader.cancel();
      throw new RangeError('too large');
    }
    chunks.push(value);
  }
  return new Blob(chunks).text();
}

function clean(text, limit) {
  return String(text ?? '').replace(/[\u0000-\u0008\u000b-\u001f\u007f]/g, '').replace(/\s+/g, ' ').trim().slice(0, limit);
}


function nearPoint(near) {
  if (near === undefined || near === null) return null;
  const lat = Number(near?.lat), lng = Number(near?.lng);
  if (!(Number.isFinite(lat) && Number.isFinite(lng) && Math.abs(lat) <= 90 && Math.abs(lng) <= 180)) throw new RangeError('bad location');
  return { latitude: Math.round(lat * 100) / 100, longitude: Math.round(lng * 100) / 100 };
}


const mapsLink = (name, address) => `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent([name, address].filter(Boolean).join(', '))}`;




async function mapbox(query, near, env, origin, { what = '', where = '', self = '' } = {}) {
  const started = Date.now();
  try {
    let around = near, words = query;
    if (!near && what && where) {
      const [area] = await mapboxForward(where, null, env, { types: 'place,locality,neighborhood,district,region,country', limit: '1' });
      const point = area?.geometry?.coordinates;
      if (Array.isArray(point) && point.length === 2) { around = { longitude: point[0], latitude: point[1] }; words = what; }
    }
    let features = await mapboxForward(words, around, env);
    if (!features.length && words !== query) features = await mapboxForward(query, null, env);

    const candidates = features.slice(0, CANDIDATES).flatMap(feature => {
      const p = feature?.properties || {}, name = clean(p.name, 120);
      if (!name) return [];
      const address = clean(p.full_address || p.place_formatted, 200), rating = Number(p.metadata?.rating);
      const [lng, lat] = Array.isArray(feature.geometry?.coordinates) ? feature.geometry.coordinates.map(Number) : [];
      const website = clean(p.metadata?.website, 300);
      return [{ name, address, rating: rating >= 0 && rating <= 5 ? rating : null, ratings: null, open: null, price: null, url: mapsLink(name, address),
        category: clean(Array.isArray(p.poi_category) ? p.poi_category[0] : '', 60) || null,
        area: [p.context?.place?.name, p.context?.region?.name].map(part => clean(part, 60)).filter(Boolean).join(', ') || null,
        ...(Number.isFinite(lat) && Number.isFinite(lng) && { lat: round(lat, 5), lng: round(lng, 5) }),
        website: /^https?:\/\//.test(website) ? website : null }];
    });
    await Promise.all(candidates.map(place => describe(place, env, self)));
    const found = candidates.map((place, i) => [place, i]).sort(([a, i], [b, j]) => Boolean(b.wiki) - Boolean(a.wiki) || i - j).slice(0, PLACES).map(([place]) => place);
    console.log(JSON.stringify({ event: 'places', provider: 'mapbox', results: found.length, near: Boolean(near), area: around !== near, ms: Date.now() - started }));
    return reply({ source: 'Mapbox', places: found }, 200, origin);
  } catch (error) {
    console.error(JSON.stringify({ event: 'places_failed', provider: 'mapbox', error: error.name, status: error.status, ms: Date.now() - started }));
    if (error.status === 429) return reply({ error: "Kani's places allowance is used up for now. Try again later." }, 429, origin);
    return reply({ error: error.name === 'TimeoutError' ? 'Places search timed out.' : 'Places search is unavailable right now.' }, 502, origin);
  }
}


function brief(text, limit = 260) {
  const whole = clean(text, 2000);
  return whole.length <= limit ? whole : whole.slice(0, limit).replace(/[\s,;:(]+\S*$/, '') + '…';
}

const round = (value, digits) => Math.round(value * 10 ** digits) / 10 ** digits;

async function mapboxForward(q, around, env, { types = 'poi', limit = String(CANDIDATES) } = {}) {
  const url = new URL(MAPBOX_UPSTREAM);
  url.search = new URLSearchParams({ q: q.slice(0, 256), access_token: env.MAPBOX_TOKEN, limit, types, language: 'en',
    ...(around && { proximity: `${around.longitude},${around.latitude}` }) }).toString();
  const upstream = await fetch(url, { signal: AbortSignal.timeout(15000) });
  if (!upstream.ok) {
    await upstream.body?.cancel();
    throw Object.assign(new Error('mapbox'), { name: 'UpstreamError', status: upstream.status });
  }
  const data = JSON.parse(await readLimited(upstream.body, MAX_UPSTREAM));
  return Array.isArray(data?.features) ? data.features : [];
}


const SMALL_WORDS = new Set(['the', 'of', 'and', 'at', 'in', 'a', 'an', 'de', 'la', 'le']);
const nameWords = text => clean(text, 200).toLowerCase().normalize('NFKD').replace(/[̀-ͯ]/g, '')
  .replace(/[^a-z0-9]+/g, ' ').split(' ').filter(word => word && !SMALL_WORDS.has(word));
function sameName(a, b) {
  const first = new Set(nameWords(a)), second = new Set(nameWords(b));
  if (!first.size || !second.size) return false;
  const shared = [...second].filter(word => first.has(word)).length;
  return shared / Math.max(first.size, second.size) >= 0.7;
}



async function describe(place, env, self) {
  if (place.lat === undefined) return;
  try {
    const url = new URL(WIKIPEDIA);
    url.search = new URLSearchParams({ action: 'query', generator: 'geosearch', ggscoord: `${place.lat}|${place.lng}`, ggsradius: '2000',
      ggslimit: '10', prop: 'pageimages|extracts|info', exintro: '1', explaintext: '1', exsentences: '1', piprop: 'thumbnail',
      pithumbsize: '480', inprop: 'url', format: 'json', formatversion: '2' }).toString();
    const response = await fetch(url, { headers: { 'User-Agent': WIKIPEDIA_AGENT }, signal: AbortSignal.timeout(4000) });
    if (response.ok) {
      const pages = JSON.parse(await readLimited(response.body, MAX_UPSTREAM))?.query?.pages;
      const page = (Array.isArray(pages) ? pages : []).find(candidate => sameName(place.name, candidate?.title));
      const wiki = typeof page?.fullurl === 'string' && page.fullurl.startsWith('https://en.wikipedia.org/') ? page.fullurl : null;
      if (wiki) {
        place.about = brief(page.extract) || null;
        place.wiki = wiki;
        const photo = page.thumbnail?.source;
        if (typeof photo === 'string' && /^https:\/\/(upload|thumb)\.wikimedia\.org\//.test(photo)) place.image = photo;
      }
    } else await response.body?.cancel();
  } catch {  }
  if (!place.image && self) place.image = await mapLink(self, place.lat, place.lng, env);
}

const hex = bytes => [...new Uint8Array(bytes)].map(b => b.toString(16).padStart(2, '0')).join('');
async function signature(at, env) {
  const key = await crypto.subtle.importKey('raw', new TextEncoder().encode('kani-map:' + env.MAPBOX_TOKEN), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  return hex(await crypto.subtle.sign('HMAC', key, new TextEncoder().encode(at))).slice(0, 32);
}
async function mapLink(self, lat, lng, env) {
  const at = `${round(lat, 4)},${round(lng, 4)}`;
  return `${self}/map?at=${at}&sig=${await signature(at, env)}`;
}


async function staticMap(request, env) {
  const url = new URL(request.url), at = url.searchParams.get('at') || '', sig = url.searchParams.get('sig') || '';
  const match = at.match(/^(-?\d{1,2}(?:\.\d{1,4})?),(-?\d{1,3}(?:\.\d{1,4})?)$/);
  if (!env.MAPBOX_TOKEN || !match || Math.abs(match[1]) > 90 || Math.abs(match[2]) > 180 || sig !== await signature(at, env))
    return new Response('Not found.', { status: 404 });
  const cache = globalThis.caches?.default, key = new Request(`https://kani-map.cache/${at}`);
  const cached = await cache?.match(key);
  if (cached) return cached;
  const [lat, lng] = [match[1], match[2]];
  const upstream = await fetch(`${MAPBOX_STATIC}/pin-l+2f8f6a(${lng},${lat})/${lng},${lat},13,0/240x240@2x?access_token=${env.MAPBOX_TOKEN}`,
    { signal: AbortSignal.timeout(10000) });
  if (!upstream.ok || !(upstream.headers.get('Content-Type') || '').startsWith('image/')) {
    await upstream.body?.cancel();
    return new Response('Map unavailable.', { status: 502 });
  }
  const image = new Response(upstream.body, { headers: { 'Content-Type': upstream.headers.get('Content-Type'),
    'Cache-Control': 'public, max-age=86400', 'X-Content-Type-Options': 'nosniff', 'Cross-Origin-Resource-Policy': 'cross-origin' } });
  await cache?.put(key, image.clone());
  return image;
}

async function places(query, near, env, origin, extra = {}) {
  if (env.MAPBOX_TOKEN) return mapbox(query, near, env, origin, extra);
  if (!env.GOOGLE_MAPS_API_KEY) return reply({ error: 'Places search is not set up yet.' }, 503, origin);
  const started = Date.now();
  try {

    const body = { textQuery: query, pageSize: PLACES, languageCode: 'en', ...(near && { locationBias: { circle: { center: near, radius: 5000 } } }) };
    const upstream = await fetch(PLACES_UPSTREAM, {
      method: 'POST',
      headers: { 'X-Goog-Api-Key': env.GOOGLE_MAPS_API_KEY, 'X-Goog-FieldMask': PLACES_FIELDS, 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(15000),
    });
    if (!upstream.ok) {
      await upstream.body?.cancel();
      console.error(JSON.stringify({ event: 'places_upstream_error', status: upstream.status, ms: Date.now() - started }));
      if (upstream.status === 429) return reply({ error: "Kani's places allowance is used up for now. Try again later." }, 429, origin);
      return reply({ error: 'Places search is unavailable right now.' }, 502, origin);
    }
    const data = JSON.parse(await readLimited(upstream.body, MAX_UPSTREAM));
    const found = (Array.isArray(data?.places) ? data.places : []).slice(0, PLACES).flatMap(place => {
      const url = typeof place?.googleMapsUri === 'string' && place.googleMapsUri.startsWith('https://') ? place.googleMapsUri : null;
      const name = clean(place?.displayName?.text, 120);
      if (!url || !name) return [];
      const rating = Number(place.rating), ratings = Number(place.userRatingCount), open = place.currentOpeningHours?.openNow;
      return [{ name, address: clean(place.formattedAddress, 200), rating: rating >= 0 && rating <= 5 ? rating : null,
        ratings: Number.isInteger(ratings) && ratings >= 0 ? ratings : null, open: typeof open === 'boolean' ? open : null, price: PRICES[place.priceLevel] || null, url }];
    });
    console.log(JSON.stringify({ event: 'places', results: found.length, near: Boolean(near), ms: Date.now() - started }));
    return reply({ source: 'Google Maps', places: found }, 200, origin);
  } catch (error) {
    console.error(JSON.stringify({ event: 'places_failed', error: error.name, ms: Date.now() - started }));
    return reply({ error: error.name === 'TimeoutError' ? 'Places search timed out.' : 'Places search is unavailable right now.' }, 502, origin);
  }
}

export default {
  async fetch(request, env) {
    const requestOrigin = request.headers.get('Origin') || '';
    const origin = ALLOWED_ORIGINS.has(requestOrigin) || (env.DEV_ORIGIN && requestOrigin === env.DEV_ORIGIN) ? requestOrigin : null;
    const { pathname } = new URL(request.url);

    if (pathname === '/map' && request.method === 'GET') return staticMap(request, env);
    if (pathname !== '/search' && pathname !== '/places') return reply({ error: 'Not found.' }, 404, origin);
    if (request.method === 'OPTIONS') {
      if (!origin) return new Response(null, { status: 403 });
      return new Response(null, { status: 204, headers: { 'Access-Control-Allow-Origin': origin, 'Access-Control-Allow-Methods': 'POST',
        'Access-Control-Allow-Headers': 'Content-Type', 'Access-Control-Max-Age': '86400', Vary: 'Origin' } });
    }
    if (request.method !== 'POST') return reply({ error: 'Use POST.' }, 405, origin, { Allow: 'POST, OPTIONS' });

    if (!origin) return reply({ error: 'This search endpoint only serves the Kani website and app.' }, 403, null);



    if (!env.PER_VISITOR || !env.OVERALL) {
      console.warn(JSON.stringify({ event: 'rate_limits_missing' }));
    } else {
      const visitor = request.headers.get('CF-Connecting-IP') || 'unknown';
      const [perVisitor, overall] = await Promise.all([env.PER_VISITOR.limit({ key: visitor }), env.OVERALL.limit({ key: 'all' })]);
      if (!perVisitor.success || !overall.success) return reply({ error: 'Too many searches. Wait a minute and try again.' }, 429, origin, { 'Retry-After': '60' });
    }
    if (pathname === '/search' && !env.LANGSEARCH_API_KEY) return reply({ error: 'Web search is not set up yet.' }, 503, origin);

    const length = Number(request.headers.get('Content-Length'));
    if (!(length > 0 && length <= MAX_BODY)) return reply({ error: 'Send a small JSON body: {"query": "..."}.' }, 413, origin);
    let query, near, what, where;
    try {
      const body = JSON.parse(await readLimited(request.body, MAX_BODY));
      query = body.query;
      near = pathname === '/places' ? nearPoint(body.near) : null;
      what = clean(body.what, MAX_QUERY);
      where = near ? '' : clean(body.where, 100);
    } catch {
      return reply({ error: 'Send JSON: {"query": "...", "near": {"lat": ..., "lng": ...}} (near is optional).' }, 400, origin);
    }
    query = typeof query === 'string' ? query.trim() : '';
    if (!query || query.length > MAX_QUERY) return reply({ error: `Use a query of 1–${MAX_QUERY} characters.` }, 400, origin);
    if (pathname === '/places') return places(query, near, env, origin, { what, where, self: new URL(request.url).origin });

    const started = Date.now();
    try {

      const upstream = await fetch(UPSTREAM, {
        method: 'POST',
        headers: { Authorization: `Bearer ${env.LANGSEARCH_API_KEY}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, count: RESULTS, contents: { text: true } }),
        signal: AbortSignal.timeout(15000),
      });
      if (!upstream.ok) {
        await upstream.body?.cancel();
        console.error(JSON.stringify({ event: 'upstream_error', status: upstream.status, ms: Date.now() - started }));
        if (upstream.status === 429) return reply({ error: "Kani's free search allowance is used up for today. It resets at 00:00 UTC." }, 429, origin);
        return reply({ error: 'Web search is unavailable right now.' }, 502, origin);
      }
      const data = JSON.parse(await readLimited(upstream.body, MAX_UPSTREAM));
      const pages = Array.isArray(data?.data?.webPages?.value) ? data.data.webPages.value : [];
      const results = pages.slice(0, RESULTS).flatMap(page => {
        const url = typeof page?.url === 'string' && (page.url.startsWith('https://') || page.url.startsWith('http://')) ? page.url : null;
        return url ? [{ title: clean(page.name, 200), url, text: clean(page.text ?? page.snippet, MAX_TEXT), date: clean(page.datePublished, 40) || null }] : [];
      });
      console.log(JSON.stringify({ event: 'search', results: results.length, ms: Date.now() - started }));
      return reply({ source: 'LangSearch', results }, 200, origin);
    } catch (error) {
      console.error(JSON.stringify({ event: 'search_failed', error: error.name, ms: Date.now() - started }));
      return reply({ error: error.name === 'TimeoutError' ? 'Web search timed out.' : 'Web search is unavailable right now.' }, 502, origin);
    }
  },
};
