#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER="${QWEN3_AGENT_LLAMA_SERVER:-$ROOT/.runtime/llama.cpp/bin/llama-server}"
MODEL="${QWEN3_AGENT_MODEL:-$ROOT/models/Qwen3-4B-Q5_K_M.gguf}"
HOST="${QWEN3_AGENT_HOST:-127.0.0.1}"
PORT="${QWEN3_AGENT_PORT:-8080}"
CTX="${QWEN3_AGENT_CONTEXT_SIZE:-4096}"
BATCH="${QWEN3_AGENT_BATCH_SIZE:-128}"
UBATCH="${QWEN3_AGENT_UBATCH_SIZE:-32}"

fail() { echo "[FAIL] $*" >&2; exit 1; }

[[ -x "$SERVER" ]] || fail "llama-server is missing or not executable: $SERVER"
[[ -f "$MODEL" ]] || fail "GGUF model is missing: $MODEL"

# This profile is intentionally concurrency-one. On Orin NX, all CPU and GPU
# allocations share the same physical RAM, so adding server slots would
# multiply KV/cache state and reduce the safety margin for ASR and TTS.
echo "[INFO] Qwen3 Agent profile: context=$CTX batch=$BATCH ubatch=$UBATCH parallel=1"
exec "$SERVER" \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --ctx-size "$CTX" \
    --parallel 1 \
    --batch-size "$BATCH" \
    --ubatch-size "$UBATCH" \
    --n-gpu-layers all \
    --flash-attn on \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --metrics \
    "$@"
