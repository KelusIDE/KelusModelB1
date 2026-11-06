#!/bin/zsh
# Publish site/ to Cloudflare Pages (https://kelus-b1.pages.dev). The weights stay in GitHub Releases (see model-host/ and site/_redirects),
# so every b1.bin is left out, along with the models the picker no longer offers. Publish and verify new model assets before running this.
set -eu
cd "${0:A:h}/.."
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
rsync -a --exclude='.*' --exclude='*.ipa' --exclude='developers/' --exclude='b1.bin' \
  --exclude='models/otter-4-8/' --exclude='models/otter-4-9/' --exclude='models/otter-4-11/' site/ "$stage/"
wrangler pages deploy "$stage" --project-name kelus-b1 --branch main --commit-dirty=true
