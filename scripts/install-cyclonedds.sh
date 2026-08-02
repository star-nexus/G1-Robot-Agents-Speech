#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON="${1:-python3}"
: "${CYCLONEDDS_SOURCE_DIR:=$HOME/.cache/g1-speech/cyclonedds}"
: "${CYCLONEDDS_HOME:=$CYCLONEDDS_SOURCE_DIR/install}"

fail() { echo "[FAIL] $*" >&2; exit 1; }

[[ -x "$PYTHON" ]] || fail "Python environment not found: $PYTHON"
command -v git >/dev/null || fail "git is not installed"
command -v cmake >/dev/null || fail "cmake is not installed"

if [[ ! -d "$CYCLONEDDS_SOURCE_DIR/.git" ]]; then
    git clone --branch releases/0.10.x \
        https://github.com/eclipse-cyclonedds/cyclonedds.git \
        "$CYCLONEDDS_SOURCE_DIR"
fi

cmake -S "$CYCLONEDDS_SOURCE_DIR" -B "$CYCLONEDDS_SOURCE_DIR/build" \
    -DCMAKE_INSTALL_PREFIX="$CYCLONEDDS_HOME"
cmake --build "$CYCLONEDDS_SOURCE_DIR/build" --target install -j"$(nproc)"

export CYCLONEDDS_HOME
"$PYTHON" -m pip install "cyclonedds==0.10.2"
"$PYTHON" -c 'import cyclonedds; print("[PASS] Cyclone DDS Python is ready")'
