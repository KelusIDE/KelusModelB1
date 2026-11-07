#!/bin/sh
# Installs the Kani CLI for the current user. It needs no sudo and uses only ~/.kani and ~/.local/bin.
#
#   curl -fsSL https://kani.kelus.org/install.sh | sh
#   curl -fsSL https://kani.kelus.org/install.sh | sh -s -- --with-cide
#
# Options: --with-cide       also download Cide, the coding model (206 MB more)
#          --no-models       install the program only
#          --no-modify-path  never edit your shell startup file
# Run it again to update. Downloads are checked against SHA-256 sums before use.
set -eu

VERSION='0.18.1'
SHA256='77c6ae1598b6aa458a59aca1c286d24818cd9475844d7b6ef337850ed658254f'
REPOSITORY='https://github.com/KelusIDE/KelusModelB1'
URL="${KANI_CLI_URL:-$REPOSITORY/releases/download/cli-v$VERSION/kani-cli-$VERSION.tar.gz}"
KANI_HOME="${KANI_HOME:-$HOME/.kani}"
BIN_DIR="${KANI_BIN_DIR:-$HOME/.local/bin}"

say() { printf '%s\n' "$*"; }
die() { printf 'kani install: %s\n' "$*" >&2; exit 1; }

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1
  elif command -v openssl >/dev/null 2>&1; then openssl dgst -sha256 "$1" | sed 's/.* //'
  else die 'needs sha256sum, shasum or openssl to check the download.'; fi
}

fetch() {
  if command -v curl >/dev/null 2>&1; then curl -fsSL --retry 3 -o "$2" "$1"
  elif command -v wget >/dev/null 2>&1; then wget -q -O "$2" "$1"
  else die 'needs curl or wget.'; fi
}

add_to_path() {
  case ":$PATH:" in *":$BIN_DIR:"*) return 0 ;; esac
  line="export PATH=\"$BIN_DIR:\$PATH\""
  case "${SHELL:-}" in
    */zsh) profile="$HOME/.zshrc" ;;
    */bash) if [ "$(uname -s)" = Darwin ]; then profile="$HOME/.bash_profile"; else profile="$HOME/.bashrc"; fi ;;
    *) profile='' ;;
  esac
  if [ "$MODIFY_PATH" = no ] || [ -z "$profile" ]; then
    say "Add this line to your shell startup file, then open a new terminal:"
    say "  $line"
    return 0
  fi
  if ! grep -qsF "$line" "$profile"; then
    printf '\n# Kani CLI\n%s\n' "$line" >> "$profile"
    say "Added $BIN_DIR to your PATH in $profile."
  fi
  say 'Open a new terminal, then run: kani'
}

main() {
  MODELS='latest'
  MODIFY_PATH=yes
  for option in "$@"; do
    case "$option" in
      --with-cide) MODELS="$MODELS cide" ;;
      --no-models) MODELS='' ;;
      --no-modify-path) MODIFY_PATH=no ;;
      *) die "unknown option $option" ;;
    esac
  done

  command -v node >/dev/null 2>&1 || die 'needs Node.js 22 or newer. Get it from https://nodejs.org, then run this again.'
  major=$(node -p 'process.versions.node.split(".")[0]')
  [ "$major" -ge 22 ] || die "needs Node.js 22 or newer, and you have $(node --version). Update it from https://nodejs.org, then run this again."

  temporary=$(mktemp -d)
  trap 'rm -rf "$temporary"' EXIT
  say "Downloading Kani CLI $VERSION..."
  fetch "$URL" "$temporary/kani-cli.tar.gz" || die "could not download $URL"
  [ "$(sha256 "$temporary/kani-cli.tar.gz")" = "$SHA256" ] || die 'the download failed its checksum, so nothing was installed.'

  mkdir -p "$KANI_HOME" "$BIN_DIR"
  target="$KANI_HOME/cli-$VERSION"
  staged="$KANI_HOME/.cli-$VERSION.$$"
  mkdir "$staged"
  tar -xzf "$temporary/kani-cli.tar.gz" -C "$staged" || { rm -rf "$staged"; die 'could not unpack the download.'; }

  # Models are large and their names are pinned by checksum, so an update keeps the ones already here.
  previous="$KANI_HOME/current"
  if [ -d "$previous/site" ]; then
    find -L "$previous/site" -name b1.bin | while IFS= read -r weights; do
      keep="$staged/${weights#"$previous"/}"
      if [ -d "$(dirname "$keep")" ]; then ln -f "$weights" "$keep" 2>/dev/null || cp "$weights" "$keep"; fi
    done
  fi

  rm -rf "$target"
  mv "$staged" "$target"
  ln -sfn "cli-$VERSION" "$KANI_HOME/current"
  chmod +x "$target/cli/kani.mjs"
  ln -sf "$KANI_HOME/current/cli/kani.mjs" "$BIN_DIR/kani"
  for old in "$KANI_HOME"/cli-*; do
    if [ "$old" != "$target" ]; then rm -rf "$old"; fi
  done

  for model in $MODELS; do
    say "Downloading the $model model (about 206 MB). This only happens once..."
    node "$target/model-host/download.mjs" "$model" || die 'a model download failed. Run this installer again to continue.'
  done

  "$BIN_DIR/kani" --version >/dev/null || die 'the program was installed but will not start.'
  if [ -n "$MODELS" ]; then
    "$BIN_DIR/kani" 'what is 2 + 2' </dev/null >/dev/null || die 'the model did not load. Run this installer again.'
  fi
  say "Installed Kani CLI $VERSION."
  found=$(command -v kani 2>/dev/null || true)
  if [ -n "$found" ] && [ "$found" != "$BIN_DIR/kani" ]; then say "Note: another kani comes first on your PATH, at $found."; fi
  add_to_path
}

main "$@"
