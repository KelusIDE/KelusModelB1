# kani-search: web search and places proxy for Kani

A Cloudflare Worker between Kani (the website kelus-b1.pages.dev and the Kani app) and two APIs:

- `/search`: the [LangSearch](https://langsearch.com) web search API. The browser sends only the search text;
  the Worker asks for five results and returns titles, links and up to 1,500 characters of text each.
- `/places`: [Mapbox Search Box](https://docs.mapbox.com/api/search/search-box/) when `MAPBOX_TOKEN` is set,
  else Google Places (Text Search) with `GOOGLE_MAPS_API_KEY`. Kani sends the words ("pizza places", "museums in
  Rome") and, for "near me" questions only, a location the app has already rounded to about a kilometre (the
  Worker rounds it again). The Worker asks for five places and returns name, address, rating when known (Google
  also gives review counts, price and open now) and a Maps link. Kani credits the service ("Results from Mapbox"),
  and saved chats keep only a Maps search link, because both services' terms allow only temporary use.

The API keys are Worker secrets and never reach the browser. Queries and locations are not logged.

Guards: only the Kani website's and app's origins get an answer (outside a browser the Origin header can be
forged), 10 lookups a minute per visitor and 60 a minute overall (per Cloudflare location, searches and places
together), a fixed request shape, and capped request and response sizes.

## Deploy

```sh
cd search-worker
npx wrangler login                          # once; approve in the browser
npx wrangler secret put LANGSEARCH_API_KEY  # paste the key from langsearch.com/dashboard
npx wrangler secret put MAPBOX_TOKEN        # places: a Mapbox access token (account.mapbox.com → Tokens)
npx wrangler deploy                         # updates https://somewhat.<account>.workers.dev
```

Places with Mapbox: create a free account at mapbox.com and copy the default public token (starts with `pk.`), or
make one with only the default public scopes. Mapbox lists about 50,000 free Search Box requests a month; check
your usage in the Mapbox dashboard after the first lookups. To use Google instead, set `GOOGLE_MAPS_API_KEY` (a
key with "Places API (New)" enabled, which needs billing; ratings and opening hours are Google's Enterprise
tier) and leave `MAPBOX_TOKEN` unset.

The website and app call `https://somewhat.chatgpt-ai-5-2o2.workers.dev`; `wrangler.jsonc` names that Worker
(`somewhat`) and keeps its dashboard settings (no preview URLs, no traces), so a deploy updates it in place.
A new Worker under another name would also need its host added to `connect-src` in `../firebase.json` and
`../mobile/index.html`, and to the URLs in `../site/web.js` and `../site/places.js`.

## Test

```sh
node --test                                 # the /places endpoint, with Google replaced by a stand-in
```

```sh
npx wrangler dev --var LANGSEARCH_API_KEY:YOUR_KEY --var DEV_ORIGIN:http://localhost:5000
curl -X POST http://localhost:8787/search -H 'Origin: http://localhost:5000' \
  -H 'Content-Type: application/json' -d '{"query": "when was Albert Einstein born"}'
```
