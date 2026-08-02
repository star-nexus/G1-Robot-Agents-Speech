#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${GPU_PYTHON:-$ROOT/.venv-gpu/bin/python}"
SHERPA_VERSION="${SHERPA_ONNX_GPU_VERSION:-1.13.4}"
ORT_VERSION="${SHERPA_ONNX_GPU_ORT_VERSION:-1.18.1}"
BUILD_ROOT="${SHERPA_ONNX_GPU_BUILD_DIR:-$ROOT/.build/sherpa-onnx-gpu}"
SOURCE_DIR="$BUILD_ROOT/sherpa-onnx-$SHERPA_VERSION"
SOURCE_ARCHIVE="$BUILD_ROOT/sherpa-onnx-v$SHERPA_VERSION.tar.gz"
WHEEL_DIR="${SHERPA_ONNX_GPU_WHEEL_DIR:-$ROOT/dist/gpu}"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-12.6}"
FORCE_REBUILD="${SHERPA_ONNX_GPU_FORCE_REBUILD:-0}"

log() { echo; echo "[GPU BUILD] $*"; }
fail() { echo "[FAIL] $*" >&2; exit 1; }

[[ "$(uname -s)" == "Linux" ]] || fail "只支持 Linux"
[[ "$(uname -m)" == "aarch64" ]] || fail "Jetson GPU wheel 要求 aarch64"
[[ -x "$PYTHON" ]] || fail "Python 环境不存在: $PYTHON"
[[ -x "$CUDA_ROOT/bin/nvcc" ]] || fail "未找到 CUDA nvcc: $CUDA_ROOT/bin/nvcc"
command -v cmake >/dev/null || fail "未安装 cmake"
command -v curl >/dev/null || fail "未安装 curl"

if [[ -r /etc/nv_tegra_release ]]; then
    log "Jetson: $(head -1 /etc/nv_tegra_release)"
fi
"$CUDA_ROOT/bin/nvcc" --version | tail -1

mkdir -p "$BUILD_ROOT" "$WHEEL_DIR"
wheel="$(find "$WHEEL_DIR" -maxdepth 1 -type f \
    -name "sherpa_onnx-${SHERPA_VERSION}+cuda-*-linux_aarch64.whl" -print -quit)"
if [[ -n "$wheel" && "$FORCE_REBUILD" != "1" ]]; then
    log "复用已有 CUDA wheel"
    echo "GPU wheel: $wheel"
    exit 0
fi

if [[ ! -f "$SOURCE_DIR/CMakeLists.txt" ]]; then
    log "下载 sherpa-onnx v$SHERPA_VERSION 源码归档"
    if ! tar tzf "$SOURCE_ARCHIVE" >/dev/null 2>&1; then
        curl -fL -C - --retry 10 --retry-all-errors \
            -o "$SOURCE_ARCHIVE" \
            "https://github.com/k2-fsa/sherpa-onnx/archive/refs/tags/v$SHERPA_VERSION.tar.gz"
    fi
    tar tzf "$SOURCE_ARCHIVE" >/dev/null
    tar xzf "$SOURCE_ARCHIVE" -C "$BUILD_ROOT"
fi

log "编译 CUDA sherpa-onnx wheel"
export PATH="$CUDA_ROOT/bin:$PATH"
export SHERPA_ONNX_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release \
-DSHERPA_ONNX_ENABLE_GPU=ON \
-DSHERPA_ONNX_LINUX_ARM64_GPU_ONNXRUNTIME_VERSION=$ORT_VERSION \
-DSHERPA_ONNX_ENABLE_BINARY=OFF \
-DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
-DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
-DSHERPA_ONNX_ENABLE_C_API=OFF"
export SHERPA_ONNX_MAKE_ARGS="${SHERPA_ONNX_MAKE_ARGS:--j4}"

cd "$SOURCE_DIR"
"$PYTHON" setup.py bdist_wheel --dist-dir "$WHEEL_DIR"

wheel="$(find "$WHEEL_DIR" -maxdepth 1 -type f \
    -name "sherpa_onnx-${SHERPA_VERSION}+cuda-*-linux_aarch64.whl" -print -quit)"
[[ -n "$wheel" ]] || fail "没有生成预期的 CUDA wheel"
echo
echo "GPU wheel: $wheel"
