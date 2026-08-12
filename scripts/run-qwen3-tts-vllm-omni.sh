#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNTIME_ROOT="${VLLM_OMNI_RUNTIME_ROOT:-$ROOT/.runtime/vllm-omni-orin-jp6}"
PYTHON="${VLLM_OMNI_PYTHON:-$RUNTIME_ROOT/.venv/bin/python}"
MODEL="${QWEN3_TTS_MODEL:-$ROOT/models/Qwen3-TTS-12Hz-0.6B-CustomVoice}"
DEPLOY_CONFIG="${QWEN3_TTS_DEPLOY_CONFIG:-$ROOT/profiles/qwen3-tts-orin-nx.yaml}"
PORT="${QWEN3_TTS_PORT:-8091}"
CUDSS_DIR="${QWEN3_TTS_CUDSS_DIR:-}"
ALLOWED_MEDIA_PATH="${QWEN3_TTS_ALLOWED_MEDIA_PATH:-$(dirname "$MODEL")}"

fail() { echo "[FAIL] $*" >&2; exit 1; }

[[ -x "$PYTHON" ]] || fail "isolated vLLM-Omni runtime is missing: $PYTHON"
[[ -d "$MODEL" ]] || fail "Qwen3-TTS model directory is missing: $MODEL"
[[ -f "$DEPLOY_CONFIG" ]] || fail "deploy profile is missing: $DEPLOY_CONFIG"
if [[ -n "$CUDSS_DIR" && ! -d "$CUDSS_DIR" ]]; then
    fail "cuDSS runtime directory is missing: $CUDSS_DIR"
fi
[[ -d "$ALLOWED_MEDIA_PATH" ]] || fail "allowed media directory is missing: $ALLOWED_MEDIA_PATH"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export VLLM_OMNI_DISABLE_TORCHVISION="${VLLM_OMNI_DISABLE_TORCHVISION:-1}"
# FlashInfer does not publish a Jetson-native wheel for this runtime. vLLM has
# a supported native/Triton sampler fallback; the SM87 FA2 attention extension
# remains enabled independently.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
# Model architecture inspection runs in child Python processes. Prepending this
# directory makes the text/audio-only torchvision compatibility hook process-wide.
export PYTHONPATH="$ROOT/scripts/jetson_sitecustomize${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "$CUDSS_DIR" ]]; then
    export LD_LIBRARY_PATH="$CUDSS_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec "$PYTHON" "$ROOT/scripts/vllm-omni-orin-launcher.py" serve "$MODEL" \
    --omni \
    --port "$PORT" \
    --deploy-config "$DEPLOY_CONFIG" \
    --trust-remote-code \
    --allowed-local-media-path "$ALLOWED_MEDIA_PATH"
