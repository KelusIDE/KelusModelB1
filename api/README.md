# Kani API

Developers use Kani in their own apps with an `hpk-` key, through an OpenAI-style chat completions endpoint:

```sh
curl https://kani-api.chatgpt-ai-5-2o2.workers.dev/v1/chat/completions \
  -H "Authorization: Bearer $KANI_API_KEY" -H "Content-Type: application/json" \
  -d '{"model": "kani-otter-4.3", "messages": [{"role": "user", "content": "hi"}]}'
```

Keys and the public docs: https://kelus-b1.pages.dev/developers/ (`site/developers/`).

```
developer ──hpk- key──▶ gateway (Cloudflare Worker kani-api, D1 kani-api) ──BACKEND_SECRET──▶ backend (Hugging Face Space, Node)
key page (site/developers/) ──Firebase ID token──▶ gateway /v1/keys
```

| Part | Where | What it does |
|---|---|---|
| `gateway/` | Cloudflare Worker `kani-api` + D1 `kani-api` | Checks keys (stored only as SHA-256), 10 requests a minute per key, 300 a day per account, 3 keys per account; key management for signed-in Kelus accounts; forwards to the backend |
| `backend/` | Hugging Face Docker Space (2 vCPU, free) | Runs the models in `models.json` with the website's engine (`site/b1.js`), one reply at a time, up to 8 waiting; local helpers, web search and places as in the app |
| `models.json` | both | The model list. Add a model here after exporting it to `site/models/<name>/` |
| `make_space.mjs` | | Copies the backend, the engine files and the int8 weights into a folder to upload as the Space |

Message text is never logged or stored; the backend logs model, source, token counts and time per request.
Web searches go from the backend through the search Worker (`search-worker/`), so the LangSearch key stays in
Cloudflare.

## Test

```sh
cd api && node --test          # backend with the real Kani Seal 3 weights; gateway with D1 in node:sqlite
```

End to end on this Mac (wrangler's local runtime may need an older compatibility date):

```sh
KANI_BACKEND_SECRET=dev PORT=7861 node api/backend/server.mjs &
cd api/gateway && npx wrangler d1 execute kani-api --local --file schema.sql
npx wrangler dev --port 8789 --compatibility-date 2026-07-08 --var BACKEND_URL:http://127.0.0.1:7861 \
  --var BACKEND_SECRET:dev --var FIREBASE_AUTH_EMULATOR:true
```

`FIREBASE_AUTH_EMULATOR` accepts unsigned Auth-emulator tokens, and only on localhost: never set it on the deployed Worker.
The key page talks to a local gateway at `http://127.0.0.1:<port>/developers/?emulators&api=http://localhost:8789`.

## Deploy

1. Backend secret, made once and kept outside the repo:
   `openssl rand -hex 32 > ~/.config/kani/backend-secret && chmod 600 ~/.config/kani/backend-secret`
2. Gateway:
   ```sh
   cd api/gateway
   npx wrangler d1 create kani-api                      # put the database_id in wrangler.jsonc
   npx wrangler d1 execute kani-api --remote --file schema.sql
   npx wrangler secret put BACKEND_SECRET < ~/.config/kani/backend-secret
   npx wrangler deploy
   ```
3. Backend: save a Hugging Face write token to `~/.config/kani/hf-token` (see `deploy_space.py`), then
   `.venv/bin/python api/deploy_space.py`. It builds the Space with `make_space.mjs`, uploads it and sets the Space
   secret `KANI_BACKEND_SECRET` to the same secret. Put the printed URL in `BACKEND_URL` in `gateway/wrangler.jsonc`
   and deploy the gateway again.
4. Key page: deploy the website (`firebase deploy --only hosting`); `firebase.json` allows the gateway in `connect-src`.

A free Space sleeps after two days without requests; the gateway then answers 503 "waking up" until it is back,
which takes a minute or two.
