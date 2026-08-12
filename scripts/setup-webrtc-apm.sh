#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${1:-$ROOT/.venv/bin/python}"
VERSION="${WEBRTC_APM_VERSION:-1.0.1}"

fail() { echo "[FAIL] $*" >&2; exit 1; }
log() { echo; echo "[WebRTC APM] $*"; }

[[ -x "$PYTHON" ]] || fail "Python runtime is missing: $PYTHON"
[[ "$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == "3.10" ]] \
    || fail "This reproducible JetPack 6 build currently targets Python 3.10"
command -v g++ >/dev/null || fail "g++ is required"

BUILD_DIR="$(mktemp -d /tmp/g1-webrtc-apm.XXXXXX)"
trap 'rm -rf -- "$BUILD_DIR"' EXIT

log "Creating isolated build environment"
python3 -m venv "$BUILD_DIR/build-venv"
BUILD_PIP="$BUILD_DIR/build-venv/bin/pip"
BUILD_PYTHON="$BUILD_DIR/build-venv/bin/python"
"$BUILD_PIP" install --upgrade pip setuptools wheel meson ninja swig build

log "Downloading aec-audio-processing $VERSION source"
"$BUILD_PIP" download --no-deps --no-binary=:all: --no-build-isolation \
    "aec-audio-processing==$VERSION" --dest "$BUILD_DIR"
SDIST="$(find "$BUILD_DIR" -maxdepth 1 -name 'aec_audio_processing-*.tar.gz' -print -quit)"
[[ -n "$SDIST" ]] || fail "source archive was not downloaded"
mkdir "$BUILD_DIR/source"
tar -xzf "$SDIST" --strip-components=1 -C "$BUILD_DIR/source"

# JetPack 6 uses Python 3.10. The 1.0.1 sources use no Python 3.11-only API,
# although their package metadata is unnecessarily restrictive.
sed -i 's/requires-python = ">=3.11"/requires-python = ">=3.10"/' \
    "$BUILD_DIR/source/pyproject.toml"

log "Building WebRTC AudioProcessing 2.1 / AEC3 for ARM64"
env PATH="$BUILD_DIR/build-venv/bin:$PATH" \
    "$BUILD_PYTHON" -m build --wheel --no-isolation \
    --outdir "$BUILD_DIR/dist" "$BUILD_DIR/source"
WHEEL="$(find "$BUILD_DIR/dist" -name 'aec_audio_processing-*-linux_aarch64.whl' -print -quit)"
[[ -n "$WHEEL" ]] || fail "ARM64 wheel was not produced"

log "Installing into $PYTHON"
"$PYTHON" -m pip install --force-reinstall --no-deps "$WHEEL"
"$PYTHON" - <<'PY'
from aec_audio_processing import AudioProcessor

apm = AudioProcessor(
    enable_aec=True,
    enable_ns=True,
    ns_level=2,
    enable_agc=True,
    enable_vad=False,
)
apm.set_stream_format(16000, 1)
apm.set_reverse_stream_format(16000, 1)
apm.set_stream_delay(80)
frame = b"\0" * 320
assert len(apm.process_reverse_stream(frame)) == 320
assert len(apm.process_stream(frame)) == 320
print("[PASS] WebRTC APM reverse/capture 10 ms smoke test")
PY
