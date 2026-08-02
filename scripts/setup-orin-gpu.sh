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

: "${ORIN_PYTHON:=python3}"
: "${CYCLONEDDS_HOME:=/home/nvidia/cyclonedds/install}"
: "${UNITREE_SDK2_PYTHON_DIR:=/home/nvidia/unitree_sdk2_python}"
: "${UPGRADE_PIP:=1}"
: "${INSTALL_SYSTEMD:=1}"
: "${ORIN_SYSTEMD_USER:=$(id -un)}"

log() { echo; echo "[GPU SETUP] $*"; }
fail() { echo "[FAIL] $*" >&2; exit 1; }

[[ "$(uname -m)" == "aarch64" ]] || fail "只支持 Jetson aarch64"
command -v "$ORIN_PYTHON" >/dev/null || fail "Python 不存在: $ORIN_PYTHON"

log "创建独立 .venv-gpu（不会修改 .venv）"
"$ORIN_PYTHON" -m venv --system-site-packages "$ROOT/.venv-gpu"
PY="$ROOT/.venv-gpu/bin/python"
PIP="$ROOT/.venv-gpu/bin/pip"
if [[ "$UPGRADE_PIP" == "1" ]]; then
    "$PY" -m pip install --upgrade pip setuptools wheel
fi
"$PIP" install -e "$ROOT" sounddevice

if ! "$PY" -c 'import unitree_sdk2py, cyclonedds' 2>/dev/null; then
    [[ -d "$UNITREE_SDK2_PYTHON_DIR" ]] \
        || fail "Unitree SDK 源码目录不存在: $UNITREE_SDK2_PYTHON_DIR"
    export CYCLONEDDS_HOME
    "$PIP" install "cyclonedds==0.10.2"
    "$PIP" install -e "$UNITREE_SDK2_PYTHON_DIR"
fi

log "构建并安装 CUDA sherpa-onnx"
GPU_PYTHON="$PY" bash "$ROOT/scripts/build-sherpa-onnx-gpu.sh"
wheel="$(find "$ROOT/dist/gpu" -maxdepth 1 -type f \
    -name 'sherpa_onnx-*+cuda-*-linux_aarch64.whl' -print -quit)"
[[ -n "$wheel" ]] || fail "CUDA wheel 不存在"
"$PIP" install --force-reinstall --no-deps "$wheel"

"$PY" - <<'PY'
import pathlib
import sherpa_onnx

version = str(sherpa_onnx.__version__)
if "+cuda" not in version:
    raise SystemExit(f"[FAIL] 不是 CUDA sherpa-onnx: {version}")
lib_dir = pathlib.Path(sherpa_onnx.__file__).parent / "lib"
required = ["libonnxruntime_providers_cuda.so", "libonnxruntime_providers_shared.so"]
missing = [name for name in required if not (lib_dir / name).is_file()]
if missing:
    raise SystemExit(f"[FAIL] CUDA provider 动态库缺失: {missing}")
print(f"[PASS] sherpa-onnx {version}")
PY

log "下载 SenseVoice FP32 模型"
bash "$ROOT/scripts/download_models_gpu.sh" "$ROOT/models"

log "生成 config.gpu.json"
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
[[ -n "$TEST_WAV" ]] || fail "缺少 SenseVoice zh.wav"

log "GPU 实际加载与识别"
"$ROOT/.venv-gpu/bin/g1-speech" doctor \
    --config "$ROOT/config.gpu.json" --load-model --skip-audio
"$ROOT/.venv-gpu/bin/g1-speech" transcribe \
    --config "$ROOT/config.gpu.json" "$TEST_WAV"

if [[ "$INSTALL_SYSTEMD" == "1" ]]; then
    log "安装 CPU/GPU systemd 后台服务"
    ORIN_SYSTEMD_USER="$ORIN_SYSTEMD_USER" \
        bash "$ROOT/scripts/install-speech-services.sh"
fi

echo
echo "GPU setup complete. 当前服务后端未切换。"
echo "选择 CPU: sudo g1-speech-service cpu"
echo "选择 GPU: sudo g1-speech-service gpu"
echo "查看状态: g1-speech-service status"
