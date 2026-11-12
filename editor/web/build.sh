#!/bin/sh
# Build the browser editor into editor/dist-web (index.html + editor.js +
# editor_bg.wasm), then serve it with `server --web editor/dist-web`.
# Requires the wasm32 target (rustup target add wasm32-unknown-unknown) and
# a wasm-bindgen CLI matching the crate version.
set -eu
cd "$(dirname "$0")/../.."

cargo build -p editor --release \
  --target wasm32-unknown-unknown \
  --no-default-features --features web

mkdir -p editor/dist-web
wasm-bindgen --target web \
  --out-dir editor/dist-web \
  target/wasm32-unknown-unknown/release/editor.wasm
cp editor/web/index.html editor/dist-web/index.html
echo "editor/dist-web ready — serve with: server --bind 127.0.0.1:3090 --web editor/dist-web"
