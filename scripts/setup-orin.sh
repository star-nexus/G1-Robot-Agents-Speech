#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_FILE="$ROOT/deploy.env"

usage() {
    echo "Usage: bash scripts/setup-orin.sh [--config /path/to/deploy.env]"
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

: "${ORIN_PYTHON:=python3}"
: "${ORIN_DDS_IFACE:=}"
: "${SPARK_HOST:=}"
: "${DDS_DOMAIN_ID:=0}"
: "${SPEECH_TOPIC:=rt/g1/hri/speech/final}"
: "${PLAYBACK_TOPIC:=rt/g1/hri/playback/state}"
: "${ORIN_MIC_DEVICE:=}"
: "${PROMPT_FOR_MIC_DEVICE:=1}"
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
: "${AUTO_INSTALL_UNITREE_SDK:=0}"
: "${CYCLONEDDS_SOURCE_DIR:=$HOME/cyclonedds}"
: "${CYCLONEDDS_HOME:=$CYCLONEDDS_SOURCE_DIR/install}"
: "${UNITREE_SDK2_PYTHON_DIR:=$HOME/unitree_sdk2_python}"
: "${MODEL_SOURCE_DIR:=}"
: "${SKIP_MODEL_DOWNLOAD:=0}"
: "${SKIP_APT:=0}"
: "${UPGRADE_PIP:=1}"
: "${RUN_LATENCY_TEST:=1}"
: "${INSTALL_SYSTEMD:=1}"
: "${ORIN_SYSTEMD_USER:=$(id -un)}"

[[ "$(uname -s)" == "Linux" ]] || fail "setup-orin.sh must run on Linux/Orin"
if [[ "$(uname -m)" != "aarch64" ]]; then
    warn "Architecture is $(uname -m), not aarch64. Continue only for a test host."
fi
command -v "$ORIN_PYTHON" >/dev/null || fail "Python not found: $ORIN_PYTHON"
"$ORIN_PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' \
    || fail "Python >= 3.10 is required"

if [[ -z "$ORIN_DDS_IFACE" && -n "$SPARK_HOST" ]]; then
    ORIN_DDS_IFACE="$(ip route get "$SPARK_HOST" 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev") {print $(i+1); exit}}')"
fi
[[ -n "$ORIN_DDS_IFACE" ]] || fail "Set ORIN_DDS_IFACE in deploy.env"
ip link show "$ORIN_DDS_IFACE" >/dev/null 2>&1 \
    || fail "Orin DDS interface does not exist: $ORIN_DDS_IFACE"
ok "DDS interface: $ORIN_DDS_IFACE"

if [[ -n "$SPARK_HOST" ]]; then
    ping -I "$ORIN_DDS_IFACE" -c 2 -W 2 "$SPARK_HOST" >/dev/null \
        || fail "Cannot ping Spark $SPARK_HOST via $ORIN_DDS_IFACE"
    ok "Orin -> Spark network"
fi

if [[ "$SKIP_APT" != "1" ]]; then
    log "Installing Orin system packages"
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
"$ORIN_PYTHON" -m venv --system-site-packages .venv
PY="$ROOT/.venv/bin/python"
PIP="$ROOT/.venv/bin/pip"
if [[ "$UPGRADE_PIP" == "1" ]]; then
    "$PY" -m pip install --upgrade pip
fi
"$PIP" install -e "${ROOT}[orin]"
ok "g1-speech[orin] installed"

if [[ -z "$ORIN_MIC_DEVICE" && "$PROMPT_FOR_MIC_DEVICE" == "1" && -t 0 ]]; then
    log "Selecting the Orin microphone"
    "$PY" - <<'PY'
import sounddevice as sd
for index, device in enumerate(sd.query_devices()):
    if device["max_input_channels"] > 0:
        print(f"  {index}: {device['name']}")
print("Current default:", sd.default.device)
PY
    read -r -p "Microphone index/name (Enter keeps PortAudio default): " ORIN_MIC_DEVICE
    export ORIN_MIC_DEVICE
fi

install_unitree_sdk() {
    log "Installing CycloneDDS 0.10.x and Unitree SDK2 Python"
    if [[ ! -d "$CYCLONEDDS_SOURCE_DIR/.git" ]]; then
        git clone -b releases/0.10.x \
            https://github.com/eclipse-cyclonedds/cyclonedds.git \
            "$CYCLONEDDS_SOURCE_DIR"
    fi
    cmake -S "$CYCLONEDDS_SOURCE_DIR" -B "$CYCLONEDDS_SOURCE_DIR/build" \
        -DCMAKE_INSTALL_PREFIX="$CYCLONEDDS_HOME"
    cmake --build "$CYCLONEDDS_SOURCE_DIR/build" --target install -j"$(nproc)"

    if [[ ! -d "$UNITREE_SDK2_PYTHON_DIR/.git" ]]; then
        git clone https://github.com/unitreerobotics/unitree_sdk2_python.git \
            "$UNITREE_SDK2_PYTHON_DIR"
    fi
    export CYCLONEDDS_HOME
    "$PIP" install -e "$UNITREE_SDK2_PYTHON_DIR"
}

if ! "$PY" -c 'import unitree_sdk2py, cyclonedds' 2>/dev/null; then
    if [[ "$AUTO_INSTALL_UNITREE_SDK" == "1" ]]; then
        install_unitree_sdk
    else
        cat >&2 <<EOF
[FAIL] unitree_sdk2py/cyclonedds is unavailable in $PY

Either install it into this venv manually:
  export CYCLONEDDS_HOME="$CYCLONEDDS_HOME"
  "$PIP" install -e "$UNITREE_SDK2_PYTHON_DIR"

Or set AUTO_INSTALL_UNITREE_SDK=1 in deploy.env and rerun this script.
EOF
        exit 1
    fi
fi
"$PY" -c 'import unitree_sdk2py, cyclonedds; print("Unitree DDS imports OK")'
ok "Unitree SDK2 Python"

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
export G1_ROOT="$ROOT"
export G1_ORIN_DDS_IFACE="$ORIN_DDS_IFACE"
"$PY" <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["G1_ROOT"])
path = root / "config.json"
source = path if path.exists() else root / "config.example.json"
config = json.loads(source.read_text(encoding="utf-8"))

raw_device = os.environ.get("ORIN_MIC_DEVICE", "").strip()
if not raw_device or raw_device.lower() == "null":
    device = None
else:
    try:
        device = int(raw_device)
    except ValueError:
        device = raw_device

config["audio"].update(sample_rate=16000, device=device)
config["sensevoice"].update(
    model_dir="models",
    model_file=None,
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
    max_speech_seconds=float(os.environ.get("VAD_MAX_SPEECH_SECONDS", "10.0")),
)
config["dds"].update(
    domain_id=int(os.environ.get("DDS_DOMAIN_ID", "0")),
    network_interface=os.environ["G1_ORIN_DDS_IFACE"],
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

log "Running Orin doctor"
"$ROOT/.venv/bin/g1-speech" doctor --config "$ROOT/config.json" --load-model

TEST_WAV="$(find "$ROOT/models" -path '*/test_wavs/zh.wav' -print -quit)"
[[ -n "$TEST_WAV" ]] || fail "SenseVoice zh.wav test file is missing"
log "Running SenseVoice file test"
"$ROOT/.venv/bin/g1-speech" transcribe --config "$ROOT/config.json" "$TEST_WAV"

if [[ "$RUN_LATENCY_TEST" == "1" ]]; then
    log "Running 800ms latency acceptance"
    "$PY" "$ROOT/acceptance/e2e_latency.py" \
        --config "$ROOT/config.json" --wav "$TEST_WAV" --target-ms 800
fi

if [[ "$INSTALL_SYSTEMD" == "1" ]]; then
    log "Installing systemd service"
    ORIN_SYSTEMD_USER="$ORIN_SYSTEMD_USER" \
        bash "$ROOT/scripts/install-speech-services.sh"
    sudo /usr/local/bin/g1-speech-service cpu
    sudo systemctl --no-pager --full status g1-speech
    ok "systemd g1-speech"
fi

echo
echo "Orin setup complete."
echo "Config: $ROOT/config.json"
echo "Manual start: $ROOT/.venv/bin/g1-speech serve --config $ROOT/config.json"
echo "Service backend: sudo g1-speech-service cpu"
