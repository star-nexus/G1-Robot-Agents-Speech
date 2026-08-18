#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROLE_PACKAGE="${TIFA_ROLE_PACKAGE:-$ROOT/roles/tifa-lockhart}"

[[ -f "$ROLE_PACKAGE/role.json" ]] || {
    echo "[FAIL] Tifa Role Package is missing: $ROLE_PACKAGE" >&2
    exit 1
}

exec "$ROOT/scripts/run-local-voice-agent.sh" \
    --role-package "$ROLE_PACKAGE" \
    "$@"
