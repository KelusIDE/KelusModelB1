

import test from 'node:test';
import assert from 'node:assert/strict';
import worker from '../src/index.js';

const GOOGLE = { places: [
  { displayName: { text: 'Joe’s Pizza' }, formattedAddress: '1435 Broadway, New York', rating: 4.6, userRatingCount: 18234,
    googleMapsUri: 'https://maps.google.com/?cid=1', currentOpeningHours: { openNow: true, weekdayDescriptions: ['…'] }, priceLevel: 'PRICE_LEVEL_INEXPENSIVE' },
  { displayName: { text: 'No link' }, googleMapsUri: 'javascript:alert(1)' },
  { displayName: { text: 'Quiet Slice' }, formattedAddress: '2 Main St', googleMapsUri: 'https://maps.google.com/?cid=2' },
] };
const ask = (body, { origin = 'https://localhost', path = '/places', env = { GOOGLE_MAPS_API_KEY: 'test-key' } } = {}) => {
  const text = JSON.stringify(body);
  return worker.fetch(new Request(`https://kani-search.test${path}`, { method: 'POST', body: text,
    headers: { Origin: origin, 'Content-Type': 'application/json', 'Content-Length': String(new TextEncoder().encode(text).length) } }), env);
};

test('a near-me places search sends only the words and a rounded location, and returns what Kani shows', async t => {
  const sent = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => { sent.push({ url, options }); return new Response(JSON.stringify(GOOGLE)); });
  const response = await ask({ query: 'pizza places', near: { lat: 40.748441, lng: -73.985664 } });
  assert.equal(response.status, 200);
  assert.equal(response.headers.get('Access-Control-Allow-Origin'), 'https://localhost');
  const { source, places } = await response.json();
  assert.equal(source, 'Google Maps');
  assert.deepEqual(places, [
    { name: 'Joe’s Pizza', address: '1435 Broadway, New York', rating: 4.6, ratings: 18234, open: true, price: '$', url: 'https://maps.google.com/?cid=1' },
    { name: 'Quiet Slice', address: '2 Main St', rating: null, ratings: null, open: null, price: null, url: 'https://maps.google.com/?cid=2' },
  ]);
  assert.equal(sent.length, 1);
  assert.equal(sent[0].url, 'https://places.googleapis.com/v1/places:searchText');
  assert.equal(sent[0].options.headers['X-Goog-Api-Key'], 'test-key');
  assert.match(sent[0].options.headers['X-Goog-FieldMask'], /^places\.displayName,/);
  assert.deepEqual(JSON.parse(sent[0].options.body), { textQuery: 'pizza places', pageSize: 5, languageCode: 'en',
    locationBias: { circle: { center: { latitude: 40.75, longitude: -73.99 }, radius: 5000 } } });
});

test('a named-place search sends no location', async t => {
  const sent = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => { sent.push(JSON.parse(options.body)); return new Response(JSON.stringify({ places: [] })); });
  const response = await ask({ query: 'museums in Rome' }, { origin: 'https://kelus-b1.web.app' });
  assert.equal(response.status, 200);
  assert.deepEqual((await response.json()).places, []);
  assert.deepEqual(sent, [{ textQuery: 'museums in Rome', pageSize: 5, languageCode: 'en' }]);
});

test('places refuses other sites, bad locations and a missing key, without calling Google', async t => {
  const fetch = t.mock.method(globalThis, 'fetch', async () => new Response('{}'));
  assert.equal((await ask({ query: 'cafes' }, { origin: 'https://example.com' })).status, 403);
  assert.equal((await ask({ query: 'cafes', near: { lat: 123, lng: 0 } })).status, 400);
  assert.equal((await ask({ query: 'cafes', near: { lat: 'x' } })).status, 400);
  assert.equal((await ask({ query: '' })).status, 400);
  assert.equal((await ask({ query: 'cafes' }, { env: {} })).status, 503);
  assert.equal((await ask({ query: 'cafes' }, { path: '/search', env: { GOOGLE_MAPS_API_KEY: 'k' } })).status, 503);
  assert.equal(fetch.mock.callCount(), 0);
});

test('Google errors become short messages', async t => {
  t.mock.method(globalThis, 'fetch', async () => new Response('quota', { status: 429 }));
  const response = await ask({ query: 'cafes' });
  assert.equal(response.status, 429);
  assert.match((await response.json()).error, /allowance/);
});

const MAPBOX = { type: 'FeatureCollection', attribution: '© Mapbox', features: [
  { type: 'Feature', properties: { name: 'Joe’s Pizza', full_address: '1435 Broadway, New York, NY 10018, United States', metadata: { rating: 4.5 },
    coordinates: { longitude: -73.986, latitude: 40.754 } } },
  { type: 'Feature', properties: { name: 'Corner Slice', place_formatted: 'New York, NY, United States' } },
  { type: 'Feature', properties: {} },
] };

test('with a Mapbox token, places come from Mapbox: the words, the rounded location, and Maps links by name', async t => {
  const sent = [];
  t.mock.method(globalThis, 'fetch', async url => { sent.push(new URL(url)); return new Response(JSON.stringify(MAPBOX)); });
  const response = await ask({ query: 'pizza places', near: { lat: 40.748441, lng: -73.985664 } }, { env: { MAPBOX_TOKEN: 'pk.test', GOOGLE_MAPS_API_KEY: 'unused' } });
  assert.equal(response.status, 200);
  const { source, places } = await response.json();
  assert.equal(source, 'Mapbox');
  assert.deepEqual(places, [
    { name: 'Joe’s Pizza', address: '1435 Broadway, New York, NY 10018, United States', rating: 4.5, ratings: null, open: null, price: null,
      url: 'https://www.google.com/maps/search/?api=1&query=Joe%E2%80%99s%20Pizza%2C%201435%20Broadway%2C%20New%20York%2C%20NY%2010018%2C%20United%20States',
      category: null, area: null, website: null },
    { name: 'Corner Slice', address: 'New York, NY, United States', rating: null, ratings: null, open: null, price: null,
      url: 'https://www.google.com/maps/search/?api=1&query=Corner%20Slice%2C%20New%20York%2C%20NY%2C%20United%20States',
      category: null, area: null, website: null },
  ]);
  assert.equal(sent.length, 1);
  assert.equal(`${sent[0].origin}${sent[0].pathname}`, 'https://api.mapbox.com/search/searchbox/v1/forward');
  assert.deepEqual(Object.fromEntries(sent[0].searchParams), { q: 'pizza places', access_token: 'pk.test', limit: '10', types: 'poi', language: 'en', proximity: '-73.99,40.75' });
});

test('a named place on Mapbox sends no location', async t => {
  const sent = [];
  t.mock.method(globalThis, 'fetch', async url => { sent.push(new URL(url)); return new Response(JSON.stringify({ features: [] })); });
  const response = await ask({ query: 'museums in Rome' }, { env: { MAPBOX_TOKEN: 'pk.test' } });
  assert.deepEqual(await response.json(), { source: 'Mapbox', places: [] });
  assert.equal(sent[0].searchParams.get('q'), 'museums in Rome');
  assert.equal(sent[0].searchParams.has('proximity'), false);
});

const BUFORD = { features: [{ geometry: { coordinates: [-84.00471, 34.120935] }, properties: { name: 'Buford' } }] };
const PARKS = { features: [
  { geometry: { coordinates: [-83.74797635, 34.38753463] }, properties: { name: 'Don Carter State Park', full_address: '5000 N Browning Bridge Rd, Gainesville, Georgia',
    poi_category: ['state park', 'park'], context: { place: { name: 'Gainesville' }, region: { name: 'Georgia' } }, metadata: { website: 'https://gastateparks.org/DonCarter' } } },
  { geometry: { coordinates: [-84.1, 34.2] }, properties: { name: 'Shady Grove RV Park', poi_category: ['campground'], metadata: { website: 'javascript:alert(1)' } } },
] };
const WIKI = { query: { pages: [
  { title: 'Lake Lanier', fullurl: 'https://en.wikipedia.org/wiki/Lake_Lanier', extract: 'A lake.' },
  { title: 'Don Carter State Park', fullurl: 'https://en.wikipedia.org/wiki/Don_Carter_State_Park', extract: 'Don Carter State Park is a state park on Lake Lanier.',
    thumbnail: { source: 'https://upload.wikimedia.org/park.jpg' } },
] } };

test('a named area is looked up first, the search runs near it, and places get Wikipedia facts or a map', async t => {
  const sent = [];
  t.mock.method(globalThis, 'fetch', async url => {
    url = new URL(url); sent.push(url);
    if (url.hostname === 'en.wikipedia.org') return new Response(JSON.stringify(url.searchParams.get('ggscoord').startsWith('34.38') ? WIKI : { batchcomplete: true }));
    return new Response(JSON.stringify(url.searchParams.get('types') === 'poi' ? PARKS : BUFORD));
  });
  const response = await ask({ query: 'campgrounds in Buford, Georgia', what: 'campgrounds', where: 'Buford, Georgia' }, { env: { MAPBOX_TOKEN: 'pk.test' } });
  const { places } = await response.json();
  assert.equal(sent[0].searchParams.get('q'), 'Buford, Georgia');
  assert.match(sent[0].searchParams.get('types'), /^place,/);
  assert.equal(sent[1].searchParams.get('q'), 'campgrounds');
  assert.equal(sent[1].searchParams.get('proximity'), '-84.00471,34.120935');
  assert.deepEqual(places[0], { name: 'Don Carter State Park', address: '5000 N Browning Bridge Rd, Gainesville, Georgia', rating: null, ratings: null,
    open: null, price: null, url: 'https://www.google.com/maps/search/?api=1&query=Don%20Carter%20State%20Park%2C%205000%20N%20Browning%20Bridge%20Rd%2C%20Gainesville%2C%20Georgia',
    category: 'state park', area: 'Gainesville, Georgia', lat: 34.38753, lng: -83.74798, website: 'https://gastateparks.org/DonCarter',
    about: 'Don Carter State Park is a state park on Lake Lanier.', wiki: 'https://en.wikipedia.org/wiki/Don_Carter_State_Park', image: 'https://upload.wikimedia.org/park.jpg' });
  assert.equal(places[1].website, null);
  assert.deepEqual(places.map(p => p.name), ['Don Carter State Park', 'Shady Grove RV Park']);
  assert.equal(places[1].about, undefined);
  assert.match(places[1].image, /^https:\/\/kani-search\.test\/map\?at=34\.2,-84\.1&sig=[0-9a-f]{32}$/);
  assert.equal(sent.find(u => u.hostname === 'en.wikipedia.org').searchParams.get('ggsradius'), '2000');


  const long = { query: { pages: [{ ...WIKI.query.pages[1], extract: 'Don Carter State Park is ' + 'a very long description '.repeat(20) }] } };
  t.mock.method(globalThis, 'fetch', async url => new Response(JSON.stringify(new URL(url).hostname === 'en.wikipedia.org' ? long : new URL(url).searchParams.get('types') === 'poi' ? PARKS : BUFORD)));
  const [cut] = (await (await ask({ query: 'parks in Buford', what: 'parks', where: 'Buford' }, { env: { MAPBOX_TOKEN: 'pk.test' } })).json()).places;
  assert.ok(cut.about.length <= 261);
  assert.match(cut.about, / (a|very|long|description)…$/);


  const maps = [];
  t.mock.method(globalThis, 'fetch', async url => { maps.push(String(url)); return new Response('png', { headers: { 'Content-Type': 'image/png' } }); });
  const image = await worker.fetch(new Request(places[1].image), { MAPBOX_TOKEN: 'pk.test' });
  assert.equal(image.status, 200);
  assert.equal(image.headers.get('Content-Type'), 'image/png');
  assert.match(maps[0], /^https:\/\/api\.mapbox\.com\/styles\/v1\/mapbox\/streets-v12\/static\/pin-l\+2f8f6a\(-84\.1,34\.2\)\/-84\.1,34\.2,13,0\/240x240@2x\?access_token=pk\.test$/);
  for (const forged of [places[1].image.replace('34.2', '34.3'), places[1].image.replace(/sig=\w+/, 'sig=' + '0'.repeat(32)), 'https://kani-search.test/map?at=1,1'])
    assert.equal((await worker.fetch(new Request(forged), { MAPBOX_TOKEN: 'pk.test' })).status, 404);
  assert.equal(maps.length, 1);
});

test('when nothing is near the named area, the whole phrase is tried; Wikipedia failing only drops its facts', async t => {
  const sent = [];
  t.mock.method(globalThis, 'fetch', async url => {
    url = new URL(url); sent.push(url);
    if (url.hostname === 'en.wikipedia.org') throw new TypeError('network');
    if (url.searchParams.get('types') !== 'poi') return new Response(JSON.stringify(BUFORD));
    return new Response(JSON.stringify(url.searchParams.has('proximity') ? { features: [] } : PARKS));
  });
  const { places } = await (await ask({ query: 'campgrounds in Buford', what: 'campgrounds', where: 'Buford' }, { env: { MAPBOX_TOKEN: 'pk.test' } })).json();
  assert.equal(sent[2].searchParams.get('q'), 'campgrounds in Buford');
  assert.equal(sent[2].searchParams.has('proximity'), false);
  assert.equal(places.length, 2);
  assert.equal(places[0].about, undefined);
  assert.match(places[0].image, /\/map\?at=34\.3875,-83\.748&sig=/);
});

test('well-known places (with a Wikipedia article) come first, then Mapbox order', async t => {
  t.mock.method(globalThis, 'fetch', async url => {
    url = new URL(url);
    if (url.hostname === 'en.wikipedia.org') return new Response(JSON.stringify(url.searchParams.get('ggscoord').startsWith('34.38') ? WIKI : {}));
    return new Response(JSON.stringify({ features: [...PARKS.features].reverse() }));
  });
  const { places } = await (await ask({ query: 'parks', near: { lat: 34.1, lng: -84 } }, { env: { MAPBOX_TOKEN: 'pk.test' } })).json();
  assert.deepEqual(places.map(p => p.name), ['Don Carter State Park', 'Shady Grove RV Park']);
});
