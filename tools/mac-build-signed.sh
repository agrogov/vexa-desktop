#!/bin/sh
# Build the macOS app signed with a stable identity so macOS privacy grants
# (Microphone, Screen & System Audio Recording) survive rebuilds.
# Ad-hoc signed builds get a new identity (cdhash) every time, which makes
# macOS silently drop the previous permission grants.
set -e
ID="${APPLE_SIGNING_IDENTITY:-$(security find-identity -v -p codesigning | grep -E 'Developer ID Application|Apple Development|Mac Developer' | head -n1 | cut -d '"' -f2)}"
if [ -z "$ID" ]; then
  echo "No code-signing identity found. Set APPLE_SIGNING_IDENTITY or create one in Xcode > Settings > Accounts." >&2
  exit 1
fi
echo "Signing with: $ID"
APPLE_SIGNING_IDENTITY="$ID" exec npm run mac:build
