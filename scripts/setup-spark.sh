#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_FILE="$ROOT/deploy.env"

usage() {
    echo "Usage: bash scripts/setup-spark.sh [--config /path/to/deploy.env]"
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
    echo "Copy deploy.env.example to deploy.env and edit it first." >&2
    exit 1
fi

set -a
# shellcheck disable=SC1090
source "$CONFIG_FILE"
set +a

log()  { echo; echo "[STEP] $*"; }
ok()   { echo "[PASS] $*"; }
fail() { echo "[FAIL] $*" >&2; exit 1; }

: "${SPARK_AGENT_PYTHON:=python3}"
: "${SPARK_DDS_IFACE:=}"
: "${ORIN_HOST:=}"
: "${DDS_DOMAIN_ID:=0}"
: "${SPARK_INSTALL_NO_DEPS:=1}"

if [[ "$SPARK_AGENT_PYTHON" == */* ]]; then
    [[ -x "$SPARK_AGENT_PYTHON" ]] || fail "Agent Python is not executable: $SPARK_AGENT_PYTHON"
    PY="$SPARK_AGENT_PYTHON"
else
    PY="$(command -v "$SPARK_AGENT_PYTHON" || true)"
    [[ -n "$PY" ]] || fail "Agent Python not found: $SPARK_AGENT_PYTHON"
fi

log "Inspecting the competition Agent Python environment"
echo "Python: $PY"
"$PY" --version
"$PY" -c 'import unitree_sdk2py, cyclonedds; print("Unitree DDS imports OK")' \
    || fail "Unitree SDK2 Python is unavailable in the Agent environment"
"$PY" -c 'import numpy; v=tuple(int(x) for x in numpy.__version__.split(".")[:2]); assert v >= (1,24), numpy.__version__; print("NumPy", numpy.__version__)' \
    || fail "Agent environment needs NumPy >= 1.24"

if [[ -z "$SPARK_DDS_IFACE" && -n "$ORIN_HOST" ]]; then
    SPARK_DDS_IFACE="$(ip route get "$ORIN_HOST" 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev") {print $(i+1); exit}}')"
fi
[[ -n "$SPARK_DDS_IFACE" ]] || fail "Set SPARK_DDS_IFACE in deploy.env"
ip link show "$SPARK_DDS_IFACE" >/dev/null 2>&1 \
    || fail "Spark DDS interface does not exist: $SPARK_DDS_IFACE"
if [[ -n "$ORIN_HOST" ]]; then
    ping -I "$SPARK_DDS_IFACE" -c 2 -W 2 "$ORIN_HOST" >/dev/null \
        || fail "Cannot ping Orin $ORIN_HOST via $SPARK_DDS_IFACE"
fi
ok "Spark network interface: $SPARK_DDS_IFACE"

log "Registering g1_speech in the Agent environment"
if [[ "$SPARK_INSTALL_NO_DEPS" == "1" ]]; then
    "$PY" -m pip install --no-deps -e "$ROOT"
else
    "$PY" -m pip install -e "$ROOT"
fi

"$PY" - <<'PY'
from g1_speech.dds import DdsPlaybackPublisher, DdsSpeechSubscriber
from g1_speech.dds_types import PlaybackStateMessage, SpeechEventMessage

speech = SpeechEventMessage(
    event_id="setup-probe",
    session_id="setup",
    sequence=1,
    created_unix_ns=1,
    source="setup-spark",
    text="测试",
    language="zh",
    audio_duration_ms=100,
    inference_ms=1.0,
    engine="sensevoice",
    is_final=True,
)
assert speech.text == "测试"
assert DdsSpeechSubscriber and DdsPlaybackPublisher and PlaybackStateMessage
print("g1_speech DDS protocol imports OK")
PY
ok "Spark g1_speech installation"

echo
echo "Spark setup complete. No Agent source files were modified."
echo "Standalone DDS test:"
echo "  $PY $ROOT/examples/agent_subscriber.py $SPARK_DDS_IFACE --domain $DDS_DOMAIN_ID"
echo "Manual Agent integration: $ROOT/PIPELINE.md"
