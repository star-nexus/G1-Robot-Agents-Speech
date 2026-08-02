#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_FILE="$ROOT/deploy.env"

if [[ -f "$CONFIG_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$CONFIG_FILE"
    set +a
fi

: "${RUNTIME_PYTHON:=python3}"
: "${CYCLONEDDS_SOURCE_DIR:=$HOME/.cache/g1-speech/cyclonedds}"
: "${CYCLONEDDS_HOME:=$CYCLONEDDS_SOURCE_DIR/install}"
: "${UPGRADE_PIP:=1}"
: "${INSTALL_SYSTEMD:=1}"
: "${SYSTEMD_USER:=$(id -un)}"

log() { echo; echo "[GPU SETUP] $*"; }
fail() { echo "[FAIL] $*" >&2; exit 1; }

[[ "$(uname -m)" == "aarch64" ]] || fail "Jetson GPU setup requires aarch64"
command -v "$RUNTIME_PYTHON" >/dev/null || fail "Python not found: $RUNTIME_PYTHON"

log "Creating an isolated .venv-gpu environment"
"$RUNTIME_PYTHON" -m venv --system-site-packages "$ROOT/.venv-gpu"
PY="$ROOT/.venv-gpu/bin/python"
PIP="$ROOT/.venv-gpu/bin/pip"
if [[ "$UPGRADE_PIP" == "1" ]]; then
    "$PY" -m pip install --upgrade pip setuptools wheel
fi
"$PIP" install -e "$ROOT" sounddevice

if ! "$PY" -c 'import cyclonedds' 2>/dev/null; then
    CYCLONEDDS_SOURCE_DIR="$CYCLONEDDS_SOURCE_DIR" \
        CYCLONEDDS_HOME="$CYCLONEDDS_HOME" \
        bash "$ROOT/scripts/install-cyclonedds.sh" "$PY"
fi

log "Building and installing CUDA sherpa-onnx"
GPU_PYTHON="$PY" bash "$ROOT/scripts/build-sherpa-onnx-gpu.sh"
wheel="$(find "$ROOT/dist/gpu" -maxdepth 1 -type f \
    -name 'sherpa_onnx-*+cuda-*-linux_aarch64.whl' -print -quit)"
[[ -n "$wheel" ]] || fail "CUDA wheel not found"
"$PIP" install --force-reinstall --no-deps "$wheel"

"$PY" - <<'PY'
import pathlib
import sherpa_onnx

version = str(sherpa_onnx.__version__)
if "+cuda" not in version:
    raise SystemExit(f"[FAIL] sherpa-onnx is not a CUDA build: {version}")
lib_dir = pathlib.Path(sherpa_onnx.__file__).parent / "lib"
required = ["libonnxruntime_providers_cuda.so", "libonnxruntime_providers_shared.so"]
missing = [name for name in required if not (lib_dir / name).is_file()]
if missing:
    raise SystemExit(f"[FAIL] CUDA provider libraries are missing: {missing}")
print(f"[PASS] sherpa-onnx {version}")
PY

log "Downloading the SenseVoice FP32 model"
bash "$ROOT/scripts/download_models_gpu.sh" "$ROOT/models"

log "Generating config.gpu.json"
export G1_ROOT="$ROOT"
"$PY" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["G1_ROOT"])
source = root / "config.json"
if not source.exists():
    source = root / "config.example.json"
config = json.loads(source.read_text(encoding="utf-8"))
model_dir = "models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
config["sensevoice"].update(
    model_dir=model_dir,
    model_file=f"{model_dir}/model.onnx",
    device="cuda",
    num_threads=1,
)
target = root / "config.gpu.json"
target.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(target)
PY

TEST_WAV="$(find "$ROOT/models" -path '*/test_wavs/zh.wav' -print -quit)"
[[ -n "$TEST_WAV" ]] || fail "SenseVoice zh.wav is missing"

log "Running a real GPU model load and transcription"
"$ROOT/.venv-gpu/bin/g1-speech" doctor \
    --config "$ROOT/config.gpu.json" --load-model --skip-audio
"$ROOT/.venv-gpu/bin/g1-speech" transcribe \
    --config "$ROOT/config.gpu.json" "$TEST_WAV"

if [[ "$INSTALL_SYSTEMD" == "1" ]]; then
    log "Installing CPU/GPU systemd services"
    SYSTEMD_USER="$SYSTEMD_USER" \
        bash "$ROOT/scripts/install-speech-services.sh"
fi

echo
echo "GPU setup complete. The active service backend was not changed."
echo "Select CPU: sudo g1-speech-service cpu"
echo "Select GPU: sudo g1-speech-service gpu"
echo "Show status: g1-speech-service status"
