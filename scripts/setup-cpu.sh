#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_FILE="$ROOT/deploy.env"

usage() {
    echo "Usage: bash scripts/setup-cpu.sh [--config /path/to/deploy.env]"
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
    echo "Run: cp '$ROOT/deploy.env.example' '$ROOT/deploy.env'" >&2
    echo "Then edit deploy.env before retrying." >&2
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

: "${RUNTIME_PYTHON:=python3}"
: "${DDS_NETWORK_INTERFACE:=}"
: "${AUDIO_INPUT_BACKEND:=alsa}"
: "${AUDIO_INPUT_FALLBACK:=}"
: "${ALSA_INPUT_CARD:=}"
: "${ALSA_INPUT_DEVICE:=0}"
: "${ALSA_INPUT_SAMPLE_RATE:=48000}"
: "${ALSA_INPUT_CHANNELS:=2}"
: "${ALSA_INPUT_DTYPE:=int16}"
: "${PULSE_INPUT_DEVICE:=pulse}"
: "${AUDIO_INPUT_BLOCK_MS:=20}"
: "${AUDIO_INPUT_LATENCY:=low}"
: "${PROMPT_FOR_MIC_DEVICE:=1}"
: "${AUDIO_PROCESSING_MODE:=off}"
: "${TTS_ENABLED:=0}"
: "${AUTO_INSTALL_CYCLONEDDS:=1}"
: "${CYCLONEDDS_SOURCE_DIR:=$HOME/.cache/g1-speech/cyclonedds}"
: "${CYCLONEDDS_HOME:=$CYCLONEDDS_SOURCE_DIR/install}"
: "${MODEL_SOURCE_DIR:=}"
: "${SKIP_MODEL_DOWNLOAD:=0}"
: "${SKIP_APT:=0}"
: "${UPGRADE_PIP:=1}"
: "${RUN_LATENCY_TEST:=1}"
: "${INSTALL_SYSTEMD:=1}"
: "${SYSTEMD_USER:=$(id -un)}"

[[ "$(uname -s)" == "Linux" ]] || fail "This setup script must run on Linux"
command -v "$RUNTIME_PYTHON" >/dev/null || fail "Python not found: $RUNTIME_PYTHON"
"$RUNTIME_PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' \
    || fail "Python >= 3.10 is required"

if [[ -n "$DDS_NETWORK_INTERFACE" ]]; then
    ip link show "$DDS_NETWORK_INTERFACE" >/dev/null 2>&1 \
        || fail "DDS network interface does not exist: $DDS_NETWORK_INTERFACE"
    ok "DDS interface: $DDS_NETWORK_INTERFACE"
else
    ok "DDS interface: automatic selection"
fi

if [[ "$SKIP_APT" != "1" ]]; then
    log "Installing system packages"
    sudo apt update
    sudo DEBIAN_FRONTEND=noninteractive apt install -y \
        python3-venv python3-dev build-essential cmake git curl bzip2 \
        libportaudio2 portaudio19-dev alsa-utils
    ok "System packages"
else
    warn "SKIP_APT=1; system package installation skipped"
fi

log "Creating Python environment"
cd "$ROOT"
"$RUNTIME_PYTHON" -m venv --system-site-packages .venv
PY="$ROOT/.venv/bin/python"
PIP="$ROOT/.venv/bin/pip"
if [[ "$UPGRADE_PIP" == "1" ]]; then
    "$PY" -m pip install --upgrade pip
fi
"$PY" -m pip install -e "$ROOT" sounddevice sherpa-onnx
if [[ "$TTS_ENABLED" == "1" ]]; then
    "$PY" -m pip install websocket-client
fi
if [[ "$AUDIO_PROCESSING_MODE" == "webrtc" ]]; then
    bash "$ROOT/scripts/setup-webrtc-apm.sh" "$PY"
fi
ok "CPU speech runtime installed"

if [[ "$AUDIO_INPUT_BACKEND" == "alsa" && -z "$ALSA_INPUT_CARD" \
      && "$PROMPT_FOR_MIC_DEVICE" == "1" && -t 0 ]]; then
    log "Selecting the stable ALSA microphone card ID"
    for card_id_path in /proc/asound/card*/id; do
        [[ -f "$card_id_path" ]] || continue
        printf '  %s\n' "$(<"$card_id_path")"
    done
    read -r -p "ALSA card ID (Enter uses unambiguous auto-detection): " ALSA_INPUT_CARD
    export ALSA_INPUT_CARD
fi

install_cyclonedds() {
    log "Installing Cyclone DDS 0.10.x"
    CYCLONEDDS_SOURCE_DIR="$CYCLONEDDS_SOURCE_DIR" \
        CYCLONEDDS_HOME="$CYCLONEDDS_HOME" \
        bash "$ROOT/scripts/install-cyclonedds.sh" "$PY"
}

if ! "$PY" -c 'import cyclonedds' 2>/dev/null; then
    if [[ "$AUTO_INSTALL_CYCLONEDDS" == "1" ]]; then
        install_cyclonedds
    else
        cat >&2 <<EOF
[FAIL] cyclonedds is unavailable in $PY

Install it into this environment manually:
  export CYCLONEDDS_HOME="$CYCLONEDDS_HOME"
  "$PIP" install "cyclonedds==0.10.2"

Or set AUTO_INSTALL_CYCLONEDDS=1 in deploy.env and rerun this script.
EOF
        exit 1
    fi
fi
"$PY" -c 'import cyclonedds; print("Cyclone DDS import OK")'
ok "Cyclone DDS Python"

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

log "Generating config.json from deploy.env"
CONFIG_ARGS=(config init --output "$ROOT/config.json" --from-env --force)
if [[ -f "$ROOT/config.json" ]]; then
    CONFIG_ARGS+=(--base "$ROOT/config.json")
fi
"$ROOT/.venv/bin/g1-speech" "${CONFIG_ARGS[@]}"
"$PY" -m json.tool "$ROOT/config.json" >/dev/null
ok "config.json"

log "Available microphone inputs"
"$PY" - <<'PY'
import sounddevice as sd
inputs = [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]
if not inputs:
    raise SystemExit("[FAIL] No microphone input found")
for index, name in inputs:
    print(f"  {index}: {name}")
print("Default:", sd.default.device)
PY

log "Running deployment checks"
"$ROOT/.venv/bin/g1-speech" doctor --config "$ROOT/config.json" --load-model

TEST_WAV="$(find "$ROOT/models" -path '*/test_wavs/zh.wav' -print -quit)"
[[ -n "$TEST_WAV" ]] || fail "SenseVoice zh.wav test file is missing"
log "Running SenseVoice file test"
"$ROOT/.venv/bin/g1-speech" transcribe --config "$ROOT/config.json" "$TEST_WAV"

if [[ "$RUN_LATENCY_TEST" == "1" ]]; then
    log "Running 800ms latency acceptance"
    "$PY" "$ROOT/tests/acceptance/e2e_latency.py" \
        --config "$ROOT/config.json" --wav "$TEST_WAV" --target-ms 800
fi

if [[ "$INSTALL_SYSTEMD" == "1" ]]; then
    log "Installing systemd service"
    SYSTEMD_USER="$SYSTEMD_USER" \
        bash "$ROOT/scripts/install-speech-services.sh"
    sudo /usr/local/bin/g1-speech-service cpu
    sudo systemctl --no-pager --full status g1-speech
    ok "systemd g1-speech"
fi

echo
echo "CPU setup complete."
echo "Config: $ROOT/config.json"
echo "Manual start: $ROOT/.venv/bin/g1-speech serve --config $ROOT/config.json"
echo "Service backend: sudo g1-speech-service cpu"
