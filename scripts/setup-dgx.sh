#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_FILE="$ROOT/deploy-dgx.env"

usage() {
    echo "Usage: bash scripts/setup-dgx.sh [--config /path/to/deploy-dgx.env]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG_FILE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "[FAIL] Missing deployment config: $CONFIG_FILE" >&2
    echo "Run: cp '$ROOT/deploy-dgx.env.example' '$ROOT/deploy-dgx.env'" >&2
    exit 1
fi

set -a
# shellcheck disable=SC1090
source "$CONFIG_FILE"
set +a

log()  { echo; echo "[STEP] $*"; }
ok()   { echo "[PASS] $*"; }
warn() { echo "[WARN] $*"; }
fail() { echo "[FAIL] $*" >&2; exit 1; }

: "${DGX_PYTHON:=$HOME/.local/bin/python3.11}"
: "${DGX_DDS_IFACE:=}"
: "${DGX_MIC_DEVICE:=}"
: "${PROMPT_FOR_MIC_DEVICE:=1}"
: "${REQUIRE_MIC:=0}"
: "${SOURCE_NAME:=dgx_mic}"
: "${DDS_DOMAIN_ID:=0}"
: "${SPEECH_TOPIC:=rt/g1/hri/speech/final}"
: "${PLAYBACK_TOPIC:=rt/g1/hri/playback/state}"
: "${SENSEVOICE_DEVICE:=cpu}"
: "${SENSEVOICE_THREADS:=6}"
: "${SENSEVOICE_LANGUAGE:=zh}"
: "${SENSEVOICE_USE_ITN:=1}"
: "${VAD_THRESHOLD:=0.5}"
: "${VAD_PRE_ROLL_SECONDS:=0.3}"
: "${VAD_MIN_SILENCE_SECONDS:=0.35}"
: "${VAD_MIN_SPEECH_SECONDS:=0.25}"
: "${VAD_MAX_SPEECH_SECONDS:=10.0}"
: "${DDS_DELIVERY_TTL_SECONDS:=120.0}"
: "${DDS_OUTBOX_CAPACITY:=128}"
: "${PLAYBACK_RESUME_DELAY_MS:=250}"
: "${PLAYBACK_MAX_ACTIVE_SECONDS:=30.0}"
: "${AUTO_INSTALL_UNITREE_SDK:=1}"
: "${CYCLONEDDS_SOURCE_DIR:=$ROOT/.deps/cyclonedds}"
: "${CYCLONEDDS_HOME:=$CYCLONEDDS_SOURCE_DIR/install}"
: "${UNITREE_SDK2_PYTHON_DIR:=$ROOT/.deps/unitree_sdk2_python}"
: "${MODEL_SOURCE_DIR:=}"
: "${SKIP_MODEL_DOWNLOAD:=0}"
: "${UPGRADE_PIP:=1}"
: "${RUN_LATENCY_TEST:=1}"
: "${INSTALL_USER_SERVICE:=1}"
: "${START_SERVICE:=0}"

[[ "$(uname -s)" == "Linux" ]] || fail "setup-dgx.sh must run on Linux"
command -v "$DGX_PYTHON" >/dev/null || fail "Python not found: $DGX_PYTHON"
"$DGX_PYTHON" -c 'import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)' \
    || fail "Python 3.10..3.12 is required (3.11 recommended)"
[[ -n "$DGX_DDS_IFACE" ]] || fail "Set DGX_DDS_IFACE in deploy-dgx.env"
ip link show "$DGX_DDS_IFACE" >/dev/null 2>&1 \
    || fail "DGX DDS interface does not exist: $DGX_DDS_IFACE"
ok "DDS interface: $DGX_DDS_IFACE"

log "Preparing a private PortAudio runtime (no sudo required)"
PORTAUDIO_ROOT="$ROOT/.deps/portaudio"
PORTAUDIO_LIB_DIR=""
if ldconfig -p 2>/dev/null | grep -q 'libportaudio\.so'; then
    ok "System PortAudio"
else
    mkdir -p "$PORTAUDIO_ROOT/packages" "$PORTAUDIO_ROOT/root"
    if ! find "$PORTAUDIO_ROOT/root" -name 'libportaudio.so.2' -print -quit | grep -q .; then
        (cd "$PORTAUDIO_ROOT/packages" && apt-get download libportaudio2)
        PORTAUDIO_DEB="$(find "$PORTAUDIO_ROOT/packages" -maxdepth 1 -name 'libportaudio2_*.deb' -print -quit)"
        [[ -n "$PORTAUDIO_DEB" ]] || fail "Could not download libportaudio2"
        dpkg-deb -x "$PORTAUDIO_DEB" "$PORTAUDIO_ROOT/root"
    fi
    PORTAUDIO_SO="$(find "$PORTAUDIO_ROOT/root" -name 'libportaudio.so.2' -print -quit)"
    [[ -n "$PORTAUDIO_SO" ]] || fail "Private libportaudio.so.2 is missing"
    PORTAUDIO_LIB_DIR="$(dirname "$PORTAUDIO_SO")"
    ln -sfn "$(basename "$(readlink -f "$PORTAUDIO_SO")")" "$PORTAUDIO_LIB_DIR/libportaudio.so"
    ok "Private PortAudio: $PORTAUDIO_LIB_DIR"
fi

log "Creating the DGX speech Python environment"
cd "$ROOT"
"$DGX_PYTHON" -m venv .venv
PY="$ROOT/.venv/bin/python"
PIP="$ROOT/.venv/bin/pip"
if [[ "$UPGRADE_PIP" == "1" ]]; then
    "$PY" -m pip install --upgrade pip
fi
"$PIP" install --no-build-isolation -e "${ROOT}[dgx,dev]"
ok "g1-speech[dgx] installed"

install_unitree_sdk() {
    log "Installing private CycloneDDS 0.10.x and Unitree SDK2 Python"
    mkdir -p "$ROOT/.deps"
    if [[ ! -d "$CYCLONEDDS_SOURCE_DIR/.git" ]]; then
        git clone --depth 1 -b releases/0.10.x \
            https://github.com/eclipse-cyclonedds/cyclonedds.git \
            "$CYCLONEDDS_SOURCE_DIR"
    fi
    if [[ ! -f "$CYCLONEDDS_HOME/lib/libddsc.so" ]]; then
        cmake -S "$CYCLONEDDS_SOURCE_DIR" -B "$CYCLONEDDS_SOURCE_DIR/build" \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX="$CYCLONEDDS_HOME" \
            -DBUILD_EXAMPLES=OFF
        cmake --build "$CYCLONEDDS_SOURCE_DIR/build" --target install --parallel "$(nproc)"
    fi
    if [[ ! -d "$UNITREE_SDK2_PYTHON_DIR/.git" ]]; then
        git clone --depth 1 https://github.com/unitreerobotics/unitree_sdk2_python.git \
            "$UNITREE_SDK2_PYTHON_DIR"
    fi
    export CYCLONEDDS_HOME
    "$PIP" install 'cyclonedds==0.10.2'
    "$PIP" install --no-deps "$UNITREE_SDK2_PYTHON_DIR"
}

if ! "$PY" -c 'import unitree_sdk2py, cyclonedds' 2>/dev/null; then
    if [[ "$AUTO_INSTALL_UNITREE_SDK" == "1" ]]; then
        install_unitree_sdk
    else
        fail "unitree_sdk2py/cyclonedds missing; set AUTO_INSTALL_UNITREE_SDK=1"
    fi
fi
ok "Unitree SDK2 Python and CycloneDDS"

RUNTIME_LIBRARY_PATH="$CYCLONEDDS_HOME/lib"
if [[ -n "$PORTAUDIO_LIB_DIR" ]]; then
    RUNTIME_LIBRARY_PATH="$PORTAUDIO_LIB_DIR:$RUNTIME_LIBRARY_PATH"
fi
export LD_LIBRARY_PATH="$RUNTIME_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="$RUNTIME_LIBRARY_PATH${LIBRARY_PATH:+:$LIBRARY_PATH}"

if [[ -n "$MODEL_SOURCE_DIR" ]]; then
    log "Copying offline models from $MODEL_SOURCE_DIR"
    [[ -d "$MODEL_SOURCE_DIR" ]] || fail "MODEL_SOURCE_DIR not found: $MODEL_SOURCE_DIR"
    mkdir -p "$ROOT/models"
    cp -a "$MODEL_SOURCE_DIR"/. "$ROOT/models"/
fi
if [[ "$SKIP_MODEL_DOWNLOAD" != "1" ]]; then
    log "Ensuring SenseVoice and Silero VAD models"
    bash "$ROOT/scripts/download_models.sh" "$ROOT/models"
fi
MODEL_FILE="$(find "$ROOT/models" -maxdepth 2 -name model.int8.onnx -print -quit 2>/dev/null || true)"
[[ -n "$MODEL_FILE" ]] || fail "SenseVoice model.int8.onnx is missing under $ROOT/models"
[[ -f "$ROOT/models/silero_vad.onnx" ]] || fail "silero_vad.onnx is missing"
ok "Offline models"

log "Detecting DGX microphone inputs"
mapfile -t MIC_LINES < <("$PY" - <<'PY'
import sounddevice as sd
for index, device in enumerate(sd.query_devices()):
    if device["max_input_channels"] > 0:
        print(f"{index}\t{device['name']}")
PY
)
if [[ ${#MIC_LINES[@]} -eq 0 ]]; then
    if [[ "$REQUIRE_MIC" == "1" ]]; then
        fail "No microphone input found"
    fi
    warn "No microphone input found; install continues but the service will not be started"
    START_SERVICE=0
else
    printf '  %s\n' "${MIC_LINES[@]}"
    if [[ -z "$DGX_MIC_DEVICE" && "$PROMPT_FOR_MIC_DEVICE" == "1" && -t 0 ]]; then
        read -r -p "Microphone index/name (Enter keeps PortAudio default): " DGX_MIC_DEVICE
        export DGX_MIC_DEVICE
    fi
fi

if [[ -n "$DGX_MIC_DEVICE" ]]; then
    log "Validating the selected microphone at 16 kHz mono"
    "$PY" - "$DGX_MIC_DEVICE" <<'PY'
import sounddevice as sd
import sys

raw = sys.argv[1]
try:
    device = int(raw)
except ValueError:
    device = raw
sd.check_input_settings(device=device, channels=1, dtype="float32", samplerate=16000)
with sd.InputStream(
    device=device,
    channels=1,
    dtype="float32",
    samplerate=16000,
    blocksize=1600,
):
    sd.sleep(250)
print(f"[PASS] microphone {device!r}: 16 kHz mono")
PY
fi

log "Generating DGX config.json"
export G1_ROOT="$ROOT"
export G1_DGX_DDS_IFACE="$DGX_DDS_IFACE"
"$PY" <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["G1_ROOT"])
path = root / "config.json"
source = path if path.exists() else root / "config.example.json"
config = json.loads(source.read_text(encoding="utf-8"))
raw_device = os.environ.get("DGX_MIC_DEVICE", "").strip()
if not raw_device or raw_device.lower() == "null":
    device = None
else:
    try:
        device = int(raw_device)
    except ValueError:
        device = raw_device
config["source_name"] = os.environ.get("SOURCE_NAME", "dgx_mic")
config["audio"].update(sample_rate=16000, device=device)
config["sensevoice"].update(
    model_dir="models",
    device=os.environ.get("SENSEVOICE_DEVICE", "cpu"),
    language=os.environ.get("SENSEVOICE_LANGUAGE", "zh"),
    use_itn=os.environ.get("SENSEVOICE_USE_ITN", "1") == "1",
    num_threads=int(os.environ.get("SENSEVOICE_THREADS", "6")),
)
config["vad"].update(
    threshold=float(os.environ.get("VAD_THRESHOLD", "0.5")),
    speech_pre_roll_seconds=float(os.environ.get("VAD_PRE_ROLL_SECONDS", "0.3")),
    min_silence_seconds=float(os.environ.get("VAD_MIN_SILENCE_SECONDS", "0.35")),
    min_speech_seconds=float(os.environ.get("VAD_MIN_SPEECH_SECONDS", "0.25")),
    max_speech_seconds=float(os.environ.get("VAD_MAX_SPEECH_SECONDS", "10")),
)
config["dds"].update(
    domain_id=int(os.environ.get("DDS_DOMAIN_ID", "0")),
    network_interface=os.environ["G1_DGX_DDS_IFACE"],
    speech_topic=os.environ.get("SPEECH_TOPIC", "rt/g1/hri/speech/final"),
    playback_topic=os.environ.get("PLAYBACK_TOPIC", "rt/g1/hri/playback/state"),
    delivery_ttl_seconds=float(os.environ.get("DDS_DELIVERY_TTL_SECONDS", "120")),
    outbox_capacity=int(os.environ.get("DDS_OUTBOX_CAPACITY", "128")),
)
config["playback"].update(
    resume_delay_ms=int(os.environ.get("PLAYBACK_RESUME_DELAY_MS", "250")),
    max_active_seconds=float(os.environ.get("PLAYBACK_MAX_ACTIVE_SECONDS", "30")),
)
path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(path)
PY
"$PY" -m json.tool "$ROOT/config.json" >/dev/null
ok "config.json"

log "Running DGX doctor"
DOCTOR_ARGS=(doctor --config "$ROOT/config.json" --load-model)
if [[ ${#MIC_LINES[@]} -eq 0 ]]; then
    DOCTOR_ARGS+=(--skip-audio)
fi
"$ROOT/.venv/bin/g1-speech" "${DOCTOR_ARGS[@]}"

TEST_WAV="$(find "$ROOT/models" -path '*/test_wavs/zh.wav' -print -quit)"
[[ -n "$TEST_WAV" ]] || fail "SenseVoice zh.wav test file is missing"
log "Running SenseVoice file test"
"$ROOT/.venv/bin/g1-speech" transcribe --config "$ROOT/config.json" "$TEST_WAV"
if [[ "$RUN_LATENCY_TEST" == "1" ]]; then
    log "Running 800ms latency acceptance"
    "$PY" "$ROOT/acceptance/e2e_latency.py" \
        --config "$ROOT/config.json" --wav "$TEST_WAV" --target-ms 800
fi

if [[ "$INSTALL_USER_SERVICE" == "1" ]]; then
    log "Installing the DGX user service"
    USER_UNIT_DIR="$HOME/.config/systemd/user"
    USER_UNIT="$USER_UNIT_DIR/g1-speech.service"
    mkdir -p "$USER_UNIT_DIR"
    sed \
        -e "s|@ROOT@|$ROOT|g" \
        -e "s|@PYTHON@|$PY|g" \
        -e "s|@RUNTIME_LIBRARY_PATH@|$RUNTIME_LIBRARY_PATH|g" \
        "$ROOT/systemd/g1-speech.user.service.in" > "$USER_UNIT"
    systemctl --user daemon-reload
    if [[ "$START_SERVICE" == "1" ]]; then
        systemctl --user enable --now g1-speech.service
        systemctl --user --no-pager --full status g1-speech.service
    else
        systemctl --user disable --now g1-speech.service >/dev/null 2>&1 || true
        warn "User service installed but not started"
    fi
    ok "$USER_UNIT"
fi

echo
echo "DGX setup complete."
echo "Config: $ROOT/config.json"
echo "Local DDS verification: bash $ROOT/scripts/verify-local.sh --config $CONFIG_FILE"
echo "Manual service: $ROOT/.venv/bin/g1-speech serve --config $ROOT/config.json"
