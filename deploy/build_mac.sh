#!/usr/bin/env bash
# Build the macOS .app + .dmg for Alpin Architect. Run from the repo root:  bash deploy/build_mac.sh
set -euo pipefail

cd "$(dirname "$0")"   # deploy/ — the spec resolves the repo root as its parent

# Use pyinstaller from the venv if present on PATH, else invoke it as a module.
PYINSTALLER=(pyinstaller)
if ! command -v pyinstaller >/dev/null 2>&1; then
  PYINSTALLER=(python -m PyInstaller)
fi

rm -rf build dist
"${PYINSTALLER[@]}" skiresort.spec --clean --noconfirm

# Ad-hoc (unsigned) signature: avoids the "app is damaged" quarantine error on Apple Silicon.
# NOT notarized — users still get the "unidentified developer" prompt on first launch (see README).
codesign --force --deep --sign - "dist/AlpinArchitect.app" || true

hdiutil create -volname "AlpinArchitect" \
  -srcfolder "dist/AlpinArchitect.app" \
  -ov -format UDZO "dist/AlpinArchitect.dmg"

echo "Built: deploy/dist/AlpinArchitect.app and deploy/dist/AlpinArchitect.dmg"
