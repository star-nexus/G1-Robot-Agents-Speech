#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${QWEN3_ASR_PYTHON:-$ROOT/.venv-gpu/bin/python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3.post1}"
MAX_JOBS="${MAX_JOBS:-1}"
BUILD_DIR="$(mktemp -d /tmp/g1-speech-fa2.XXXXXX)"
trap 'rm -rf "$BUILD_DIR"' EXIT

[[ -x "$PYTHON" ]] || {
    echo "[FAIL] Python runtime not found: $PYTHON" >&2
    echo "Run scripts/setup-jetson-gpu.sh and scripts/setup-qwen3-asr.sh first." >&2
    exit 1
}
[[ -x "$CUDA_HOME/bin/nvcc" ]] || {
    echo "[FAIL] CUDA compiler not found: $CUDA_HOME/bin/nvcc" >&2
    exit 1
}

# Orin is Ampere SM 8.7. The upstream source filenames say sm80, but the audited
# patch below makes nvcc emit native SM 8.7 code and skips unrelated architectures.
export CUDA_HOME MAX_JOBS
export PATH="$CUDA_HOME/bin:$PATH"
export FLASH_ATTENTION_FORCE_BUILD=TRUE
# Select the upstream Ampere block; the audited source patch retargets its
# generated code from server SM 8.0 to Orin SM 8.7.
export FLASH_ATTN_CUDA_ARCHS=80
# Both released Qwen3-ASR sizes use head-dim 64 in the audio tower and 128 in
# the decoder, with dropout 0 and no ALiBi, softcap, or local attention.
FA2_COMPILE_DEFINES="-DFLASHATTENTION_DISABLE_BACKWARD \
-DFLASHATTENTION_DISABLE_DROPOUT \
-DFLASHATTENTION_DISABLE_ALIBI \
-DFLASHATTENTION_DISABLE_SOFTCAP \
-DFLASHATTENTION_DISABLE_UNEVEN_K \
-DFLASHATTENTION_DISABLE_LOCAL"
export CFLAGS="${CFLAGS:+$CFLAGS }$FA2_COMPILE_DEFINES"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:+$NVCC_PREPEND_FLAGS }$FA2_COMPILE_DEFINES"

"$PYTHON" -m pip install "ninja>=1.11" packaging psutil
"$PYTHON" -m pip download \
    "flash-attn==$FLASH_ATTN_VERSION" \
    --no-deps --no-binary=:all: --no-build-isolation --dest "$BUILD_DIR"
tar -xf "$BUILD_DIR/flash_attn-$FLASH_ATTN_VERSION.tar.gz" -C "$BUILD_DIR"
SOURCE_DIR="$BUILD_DIR/flash_attn-$FLASH_ATTN_VERSION"
"$PYTHON" "$ROOT/scripts/prepare-flash-attention-2-qwen.py" "$SOURCE_DIR"
"$PYTHON" -m pip install "$SOURCE_DIR" \
    --no-build-isolation --no-deps --force-reinstall

"$PYTHON" - <<'PY'
import flash_attn
from flash_attn import flash_attn_func

print(f"[PASS] flash-attn {flash_attn.__version__} imports successfully")
print(f"[PASS] flash_attn_func: {flash_attn_func}")
PY

echo "Set qwen3_asr.attention_implementation to flash_attention_2 (or fa2)."
