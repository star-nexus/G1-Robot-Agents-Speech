#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "[DEPRECATED] install_orin.sh has been replaced by setup-orin.sh." >&2
exec bash "$ROOT/scripts/setup-orin.sh" "$@"
