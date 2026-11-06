# Model downloads

Model binaries live in the public `KelusIDE/KelusModelB1` GitHub release
`models-2026-11-05`. The Android release remains independent. `catalog.json`
records every binary's size, SHA-256, filename and local manifest directory.
The release includes each manifest, a checksum file and the project notices.

`kani-models` is a small Cloudflare Worker that streams these fixed GitHub
assets with browser CORS headers. It accepts only known model IDs and their
exact checksums. It cannot proxy arbitrary URLs, forwards no client cookies
or credentials, and does not buffer model binaries into Worker memory.
Versioned responses use immutable browser caching. Worker request quotas
still apply; this is not an unlimited service or a GPU inference endpoint.

The Kani browser runtime also stores verified weights in Cache Storage and
checks their SHA-256 before reuse. Storage eviction, private browsing or a
new model version can require another download. Inference remains local.
The separate voice models continue to use their original download hosts.

## Download weights after cloning

Use Node.js 22+ from the repository root:

```sh
node model-host/download.mjs latest cide
node model-host/download.mjs seal-3
```

The default command downloads Otter 4.12 and Cide 4.5 (about 394 MiB total).
Each file is streamed to a temporary path and checked against the pinned size
and SHA-256 before installation. Existing valid files are reused. A failed
download does not replace existing weights. Pass `--all` for every model in
the catalog (about 2.1 GiB). No Python checkpoints are included in this export.

Before packaging the mobile app, also download `seal-3`, then run
`npm run model --prefix mobile`. The model binaries belong in Releases, not
Git history; the matching manifests remain in source control.

## Release another snapshot

1. Change the release tag in `prepare-release.mjs` for the new snapshot.
2. Run `node model-host/prepare-release.mjs <temporary-staging-directory>`.
   This verifies each local export and stages immutable copies.
3. Create a draft GitHub release with `--latest=false`, then upload the staged
   files. Compare GitHub asset digests with `SHA256SUMS` before publishing.
4. Run `npm test --prefix model-host`, `wrangler types --config
   model-host/wrangler.jsonc --path model-host/worker-configuration.d.ts`,
   and `wrangler deploy --dry-run --config model-host/wrangler.jsonc`.
5. Deploy the Worker, verify a complete binary's SHA-256, then run
   `node model-host/configure-sites.mjs ../Kelus` to update model routes.
6. Build and deploy the affected website hosts. Firebase ignores `b1.bin`
   files; its redirects point to the Worker. Pages also excludes them.
7. Verify a cold browser load and a subsequent cache load. Remove the owned
   temporary staging directory after the release has been verified.

Code changes to the websites do not require reuploading unchanged weights.
Do not replace binaries under an existing checksum or reuse an old snapshot
tag for different contents. Keep the matching JSON manifest with each export.
