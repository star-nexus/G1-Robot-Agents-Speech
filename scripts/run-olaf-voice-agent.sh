#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROLE_PACKAGE="${OLAF_ROLE_PACKAGE:-$ROOT/roles/olaf}"

[[ -f "$ROLE_PACKAGE/role.json" ]] || {
    echo "[FAIL] Olaf Role Package is missing: $ROLE_PACKAGE" >&2
    exit 1
}

exec "$ROOT/scripts/run-local-voice-agent.sh" \
    --role-package "$ROLE_PACKAGE" \
    "$@"
