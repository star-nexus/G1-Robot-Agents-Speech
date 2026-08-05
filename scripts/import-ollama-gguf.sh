#!/usr/bin/env bash
set -Eeuo pipefail

MODEL_PATH="${1:-}"
MODEL_NAME="${2:-qwen3-8b-q5}"

if [[ -z "$MODEL_PATH" ]]; then
    echo "Usage: $0 /absolute/path/to/model.gguf [ollama-model-name]" >&2
    exit 2
fi
command -v ollama >/dev/null 2>&1 || {
    echo "[FAIL] ollama is not installed." >&2
    exit 1
}
MODEL_PATH="$(realpath "$MODEL_PATH")"
[[ -f "$MODEL_PATH" ]] || {
    echo "[FAIL] GGUF model not found: $MODEL_PATH" >&2
    exit 1
}
curl -fsS http://127.0.0.1:11434/api/version >/dev/null || {
    echo "[FAIL] Ollama is not listening on http://127.0.0.1:11434" >&2
    exit 1
}

MODELFILE="$(mktemp)"
trap 'rm -f "$MODELFILE"' EXIT
printf '%s\n' \
    "FROM $MODEL_PATH" \
    "PARAMETER num_ctx 4096" \
    "PARAMETER temperature 0.6" \
    > "$MODELFILE"

ollama create "$MODEL_NAME" -f "$MODELFILE"
ollama show "$MODEL_NAME"
echo "[PASS] Ollama model ready: $MODEL_NAME"
