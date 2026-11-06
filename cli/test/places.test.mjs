
import test from 'node:test';
import assert from 'node:assert/strict';
import { PLACES_CALL, browserLocation, maybePlaces, mapsSearch, placesAnswer, placesSummary, placesText, roughly, findPlaces, placesReply } from '../../site/places.js';

const PIZZA = [{ name: 'Joe’s Pizza', address: '1435 Broadway, New York', rating: 4.6, ratings: 18234, open: true, price: '$', url: 'https://maps.google.com/?cid=1' },
  { name: 'Quiet Slice', address: '', rating: null, ratings: null, open: false, price: null, url: 'https://maps.google.com/?cid=2' }];
const collect = async pieces => { let text = ''; for await (const piece of pieces) text += piece; return text; };

test('places calls are recognized, and held back while they could still be one', () => {
  assert.deepEqual('PLACES: cheap sushi @ here'.match(PLACES_CALL).slice(1), ['cheap sushi', 'here']);
  assert.deepEqual('PLACES: things to do @ Rome'.match(PLACES_CALL).slice(1), ['things to do', 'Rome']);
  assert.equal('PLACES: sushi near me'.match(PLACES_CALL), null);
  assert.ok(maybePlaces('PLA') && maybePlaces('PLACES: cafes @') && !maybePlaces('Places are fun'));
});

test('near me: the rounded location goes with the words, and the places come back with Maps links', async () => {
  const sent = [];
  const fetcher = async (url, options) => { sent.push({ url, ...options, body: JSON.parse(options.body) }); return new Response(JSON.stringify({ source: 'Mapbox', places: PIZZA })); };
  const text = await collect(placesAnswer('PLACES: pizza places @ here'.match(PLACES_CALL), { locate: async () => ({ lat: 40.748441, lng: -73.985664 }), fetcher }));
  assert.equal(sent.length, 1);
  assert.match(sent[0].url, /\/places$/);
  assert.deepEqual(sent[0].body, { query: 'pizza places', what: 'pizza places', near: { lat: 40.75, lng: -73.99 } });
  assert.equal(sent[0].credentials, 'omit');
  assert.equal(text, 'Looking for pizza places near you…\n\n2 places for pizza places near you:\n\n'
    + '1. Joe’s Pizza · ★ 4.6 (18,234) · $ · Open now\n   1435 Broadway, New York\n   https://maps.google.com/?cid=1\n\n'
    + '2. Quiet Slice · Closed now\n   https://maps.google.com/?cid=2\n\nResults from Mapbox');
});

test('a named place needs no location; no location means a clear request, not a guess', async () => {
  let located = false;
  const sent = [];
  const fetcher = async (url, options) => { sent.push(JSON.parse(options.body)); return new Response(JSON.stringify({ places: [] })); };
  const named = await collect(placesAnswer('PLACES: museums @ Rome'.match(PLACES_CALL), { locate: async () => { located = true; }, fetcher }));
  assert.equal(located, false);
  assert.deepEqual(sent, [{ query: 'museums in Rome', what: 'museums', where: 'Rome' }]);
  assert.match(named, /The map service found no museums in Rome\. .*https:\/\/www\.google\.com\/maps\/search\/\?api=1&query=museums%20in%20Rome/);
  const denied = await collect(placesAnswer('PLACES: cafes @ here'.match(PLACES_CALL), { locate: async () => null, fetcher }));
  assert.match(denied, /I need your location to find cafes near you/);
  assert.equal(sent.length, 1);
});

test('service errors are short and still offer a Maps search', async () => {
  const refused = await collect(placesAnswer('PLACES: bars @ Seoul'.match(PLACES_CALL),
    { fetcher: async () => new Response(JSON.stringify({ error: 'Places search is not set up yet.' }), { status: 503 }) }));
  assert.match(refused, /Places search is not set up yet\. You can also try https:\/\/www\.google\.com\/maps\/search/);
  const offline = await collect(placesAnswer('PLACES: bars @ Seoul'.match(PLACES_CALL), { fetcher: async () => { throw new TypeError('offline'); } }));
  assert.match(offline, /unavailable right now/);
});

test('saved chats keep a Maps search link, never the places', () => {
  assert.equal(placesSummary('vegan restaurants', 'here'), 'I looked up vegan restaurants near you. Search again in Maps: https://www.google.com/maps/search/?api=1&query=vegan%20restaurants');
  assert.equal(mapsSearch('hotels', 'Kyoto'), 'https://www.google.com/maps/search/?api=1&query=hotels%20in%20Kyoto');
  assert.deepEqual(roughly({ coords: { latitude: 37.774929, longitude: -122.419416 } }), { lat: 37.77, lng: -122.42 });
  assert.equal(roughly({ lat: 200, lng: 0 }), null);
  assert.match(placesText('parks', 'here', PIZZA.slice(0, 1)), /^One place for parks near you:/);
});

test('location: network first, GPS when there is no network fix, reused for ten minutes, nothing after a refusal', async () => {
  const asked = [];
  const geolocation = outcomes => ({ getCurrentPosition(ok, fail, options) {
    asked.push(options.enableHighAccuracy);
    const next = outcomes.shift();
    next.code ? fail(next) : ok({ coords: next });
  } });
  let clock = 0;
  const later = () => (clock += 3600000);
  assert.deepEqual(await browserLocation(geolocation([{ latitude: 51.5007, longitude: -0.1246 }]), later()), { lat: 51.5, lng: -0.12 });
  assert.deepEqual(asked.splice(0), [false]);
  assert.deepEqual(await browserLocation(geolocation([]), clock + 60000), { lat: 51.5, lng: -0.12 });
  assert.deepEqual(asked.splice(0), []);
  assert.deepEqual(await browserLocation(geolocation([{ code: 3 }, { latitude: 35.6586, longitude: 139.7454 }]), later()), { lat: 35.66, lng: 139.75 });
  assert.deepEqual(asked.splice(0), [false, true]);
  assert.equal(await browserLocation(geolocation([{ code: 1 }]), later()), null);
  assert.deepEqual(asked.splice(0), [false]);
  assert.equal(await browserLocation(undefined), null);
});

test('findPlaces gives the places as data, keeping only safe links and pictures', async () => {
  const place = { name: 'Don Carter State Park', url: 'https://www.google.com/maps/search/?api=1&query=Don', category: 'state park', area: 'Gainesville, Georgia',
    about: 'A state park on Lake Lanier.', wiki: 'https://en.wikipedia.org/wiki/Don_Carter_State_Park', website: 'http://gastateparks.org', lat: 34.38, lng: -83.74,
    image: 'https://somewhat.chatgpt-ai-5-2o2.workers.dev/map?at=34.38,-83.74&sig=ab' };
  const fetcher = async () => new Response(JSON.stringify({ source: 'Mapbox', places: [place,
    { ...place, name: 'Bad links', image: 'https://evil.example/x.png', website: 'javascript:alert(1)', wiki: 'https://evil.example/wiki' },
    { name: 'No link', url: 'javascript:alert(1)' }] }));
  const result = await findPlaces('PLACES: state parks @ Buford, Georgia'.match(PLACES_CALL), { fetcher });
  assert.equal(result.status, 'found');
  assert.equal(result.source, 'Mapbox');
  assert.deepEqual(result.places.map(p => p.name), ['Don Carter State Park', 'Bad links']);
  assert.equal(result.places[0].image, place.image);
  assert.equal(result.places[0].wiki, place.wiki);
  assert.equal(result.places[0].website, 'http://gastateparks.org');
  assert.deepEqual([result.places[1].image, result.places[1].website, result.places[1].wiki], [null, null, null]);
  assert.match(placesReply(result), /^2 places for state parks in Buford, Georgia:/);
  const none = await findPlaces('PLACES: cafes @ here'.match(PLACES_CALL), { fetcher, locate: async () => null });
  assert.equal(none.status, 'location');
});
