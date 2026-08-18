#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    printf '%s\n' \
        "Usage: scripts/run-qwen3-vl-llama.sh MODEL.gguf MMPROJ.gguf" \
        "" \
        "The defaults target one visual Agent on a 16 GB Jetson." \
        "Override QWEN_VL_HOST, QWEN_VL_PORT, QWEN_VL_CONTEXT," \
        "QWEN_VL_BATCH, QWEN_VL_UBATCH, QWEN_VL_IMAGE_TOKENS," \
        "QWEN_VL_KV_TYPE, QWEN_VL_GPU_LAYERS, or LLAMA_SERVER_BIN."
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

MODEL_FILE="${1:-${QWEN_VL_MODEL_FILE:-}}"
MMPROJ_FILE="${2:-${QWEN_VL_MMPROJ_FILE:-}}"
if [[ -z "$MODEL_FILE" || -z "$MMPROJ_FILE" ]]; then
    usage >&2
    exit 2
fi
if [[ ! -f "$MODEL_FILE" ]]; then
    printf '[FAIL] Model not found: %s\n' "$MODEL_FILE" >&2
    exit 1
fi
if [[ ! -f "$MMPROJ_FILE" ]]; then
    printf '[FAIL] Projector not found: %s\n' "$MMPROJ_FILE" >&2
    exit 1
fi

SERVER_BIN="${LLAMA_SERVER_BIN:-$(command -v llama-server || true)}"
if [[ -z "$SERVER_BIN" || ! -x "$SERVER_BIN" ]]; then
    printf '%s\n' '[FAIL] llama-server not found; set LLAMA_SERVER_BIN.' >&2
    exit 1
fi

exec "$SERVER_BIN" \
    -m "$MODEL_FILE" \
    --mmproj "$MMPROJ_FILE" \
    --alias "${QWEN_VL_ALIAS:-qwen3-vl-4b-q4}" \
    --host "${QWEN_VL_HOST:-127.0.0.1}" \
    --port "${QWEN_VL_PORT:-8080}" \
    -c "${QWEN_VL_CONTEXT:-1536}" \
    -n 128 \
    -ngl "${QWEN_VL_GPU_LAYERS:-99}" \
    -np 1 \
    -b "${QWEN_VL_BATCH:-512}" \
    -ub "${QWEN_VL_UBATCH:-512}" \
    -ctk "${QWEN_VL_KV_TYPE:-q8_0}" \
    -ctv "${QWEN_VL_KV_TYPE:-q8_0}" \
    --image-min-tokens "${QWEN_VL_IMAGE_TOKENS:-512}" \
    --image-max-tokens "${QWEN_VL_IMAGE_TOKENS:-512}" \
    --cache-ram 0 \
    --no-cache-prompt \
    --no-cache-idle-slots \
    --ctx-checkpoints 0 \
    --flash-attn on \
    --reasoning off \
    --no-webui
