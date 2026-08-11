#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${QWEN3_ASR_PYTHON:-$ROOT/.venv-gpu/bin/python}"

[[ -x "$PYTHON" ]] || {
    echo "[FAIL] Python runtime not found: $PYTHON" >&2
    echo "Run scripts/setup-jetson-gpu.sh first, or set QWEN3_ASR_PYTHON." >&2
    exit 1
}

"$PYTHON" -m pip install --upgrade \
    "numpy>=1.24,<1.25" \
    "accelerate>=1.0" \
    "pillow>=10" \
    "transformers==5.13.0"
"$PYTHON" -m pip install --editable "$ROOT" --no-deps

echo "[PASS] Qwen3-ASR Python runtime is ready."
echo "Copy config.qwen3-asr.example.json to config.qwen3-asr.local.json,"
echo "set model_dir, then validate it with g1-speech doctor --load-model."
echo "Keep benchmark variants with benchmark artifacts, not in the project root."
