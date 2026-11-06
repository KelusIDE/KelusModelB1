




export const PLACES_CALL = /^PLACES: ?(\S.{0,199}?) @ (.{1,100})$/;
const PROXY = 'https://somewhat.chatgpt-ai-5-2o2.workers.dev/places';

const NODE = typeof process !== 'undefined' && Boolean(process.versions?.node) && typeof document === 'undefined';


export const maybePlaces = text => 'PLACES:'.startsWith(text) || text.startsWith('PLACES:');

const nearYou = where => where.toLowerCase() === 'here';
export const phrase = (what, where) => nearYou(where) ? `${what} near you` : `${what} in ${where}`;


export const mapsSearch = (what, where) =>
  `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(nearYou(where) ? what : `${what} in ${where}`)}`;


export const placesSummary = (what, where) => `I looked up ${phrase(what, where)}. Search again in Maps: ${mapsSearch(what, where)}`;


export function roughly(position) {
  const lat = Number(position?.lat ?? position?.coords?.latitude), lng = Number(position?.lng ?? position?.coords?.longitude);
  if (!(Number.isFinite(lat) && Number.isFinite(lng) && Math.abs(lat) <= 90 && Math.abs(lng) <= 180)) return null;
  return { lat: Math.round(lat * 100) / 100, lng: Math.round(lng * 100) / 100 };
}






let remembered = null;
export async function browserLocation(geolocation = globalThis.navigator?.geolocation, now = Date.now()) {
  if (!geolocation) return null;
  if (remembered && now - remembered.at < 600000) return remembered.near;
  const near = await locateOnce(geolocation);
  if (near) remembered = { near, at: now };
  return near;
}

async function locateOnce(geolocation) {
  const ask = (enableHighAccuracy, timeout) => new Promise(resolve => geolocation.getCurrentPosition(position => resolve(roughly(position)),
    error => resolve(error?.code === 1 ? 'denied' : null), { enableHighAccuracy, maximumAge: 600000, timeout }));
  const rough = await ask(false, 8000);
  if (rough === 'denied') return null;
  return rough || (await ask(true, 15000).then(near => near === 'denied' ? null : near));
}


export function placesText(what, where, places, source = 'the map service') {
  if (!places.length) return `${source === 'the map service' ? 'The map service' : source} found no ${phrase(what, where)}. Try other words, or ${mapsSearch(what, where)}`;
  const lines = places.map((place, i) => {
    const facts = [place.rating != null && `★ ${place.rating.toFixed(1)}${place.ratings ? ` (${place.ratings.toLocaleString('en-US')})` : ''}`,
      place.price, place.open === true ? 'Open now' : place.open === false ? 'Closed now' : ''].filter(Boolean);
    return [`${i + 1}. ${place.name}${facts.length ? ' · ' + facts.join(' · ') : ''}`, place.address && `   ${place.address}`, `   ${place.url}`].filter(Boolean).join('\n');
  });
  return `${places.length === 1 ? 'One place' : `${places.length} places`} for ${phrase(what, where)}:\n\n${lines.join('\n\n')}\n\nResults from ${source}`;
}


const WORKER = new URL(PROXY).origin;
const https = (value, test = /^https:\/\//) => typeof value === 'string' && test.test(value) && value.length <= 600 ? value : null;
const words = (value, limit) => typeof value === 'string' && value.trim() ? value.trim().slice(0, limit) : null;
function checked(place) {
  const name = words(place?.name, 120), url = https(place?.url);
  if (!name || !url) return [];
  return [{ name, url, address: words(place.address, 200), rating: Number.isFinite(place.rating) ? place.rating : null,
    ratings: Number.isInteger(place.ratings) ? place.ratings : null, open: typeof place.open === 'boolean' ? place.open : null,
    price: words(place.price, 8), category: words(place.category, 60), area: words(place.area, 120), about: words(place.about, 300),
    website: https(place.website, /^https?:\/\//), wiki: https(place.wiki, /^https:\/\/en\.wikipedia\.org\//),
    image: https(place.image, new RegExp(`^(https://(upload|thumb)\\.wikimedia\\.org/|${WORKER.replace(/\./g, '\\.')}/map\\?)`)) }];
}






export async function findPlaces(call, { locate, fetcher = fetch, signal }) {
  const what = call[1].trim(), where = call[2].trim(), result = { what, where, places: [], source: undefined };
  let near = null;
  if (nearYou(where)) {
    near = roughly(await Promise.resolve(locate?.()).catch(() => null));
    if (!near) return { ...result, status: 'location' };
  }
  let response;
  try {
    response = await fetcher(PROXY, { method: 'POST', headers: { 'Content-Type': 'application/json', ...(NODE && { Origin: 'https://kelus-b1.web.app' }) },
      credentials: 'omit', referrerPolicy: 'no-referrer',
      body: JSON.stringify({ query: nearYou(where) ? what : `${what} in ${where}`, what, ...(near ? { near } : { where }) }), signal });
  } catch (error) {
    if (error?.name === 'AbortError') throw error;
    return { ...result, status: 'error', message: 'Places search is unavailable right now. Try again in a moment.' };
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) return { ...result, status: 'error', message: typeof data.error === 'string' ? data.error.slice(0, 200) : 'Places search is unavailable right now.' };
  const source = typeof data.source === 'string' && data.source.trim() ? data.source.trim().slice(0, 40) : undefined;
  const places = (Array.isArray(data.places) ? data.places : []).slice(0, 10).flatMap(checked);
  return { ...result, status: places.length ? 'found' : 'none', places, source };
}


export function placesReply({ what, where, status, places, source, message }) {
  if (status === 'location') return `I need your location to find ${what} near you. Allow location for Kani, or name a place, like “${what} in Chicago”.`;
  if (status === 'error') return `${message} You can also try ${mapsSearch(what, where)}`;
  return placesText(what, where, places, source);
}


export async function* placesAnswer(call, options) {
  yield `Looking for ${phrase(call[1].trim(), call[2].trim())}…\n\n`;
  yield placesReply(await findPlaces(call, options));
}
