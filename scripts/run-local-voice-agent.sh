#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${LOCAL_VOICE_AGENT_PYTHON:-$ROOT/.venv-gpu/bin/python}"
CONFIG="${LOCAL_VOICE_AGENT_CONFIG:-$ROOT/config.tts-demo.local.json}"
URL="${QWEN3_AGENT_URL:-http://127.0.0.1:8080/v1/chat/completions}"

fail() { echo "[FAIL] $*" >&2; exit 1; }

[[ -x "$PYTHON" ]] || fail "Python runtime is missing: $PYTHON"
[[ -f "$CONFIG" ]] || fail "Speech configuration is missing: $CONFIG"

# PYTHONPATH keeps this development launcher on the checked-out source without
# requiring an editable install in the Jetson runtime environment.
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON" -m star_runtime.cli.main agent \
    --config "$CONFIG" \
    --url "$URL" \
    "$@"
