#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_FILE="$ROOT/deploy.env"

usage() {
    echo "Usage: bash scripts/verify-dual.sh [--config /path/to/deploy.env]"
    echo "Run this script on NVIDIA Spark after both setup scripts have passed."
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
    exit 1
fi

set -a
# shellcheck disable=SC1090
source "$CONFIG_FILE"
set +a

log()  { echo; echo "[STEP] $*"; }
ok()   { echo "[PASS] $*"; }
fail() { echo "[FAIL] $*" >&2; exit 1; }

: "${ORIN_HOST:?Set ORIN_HOST in deploy.env}"
: "${ORIN_USER:?Set ORIN_USER in deploy.env}"
: "${ORIN_DDS_IFACE:?Set ORIN_DDS_IFACE in deploy.env}"
: "${SPARK_HOST:?Set SPARK_HOST in deploy.env}"
: "${SPARK_DDS_IFACE:?Set SPARK_DDS_IFACE in deploy.env}"
: "${SPARK_AGENT_PYTHON:?Set SPARK_AGENT_PYTHON in deploy.env}"
: "${ORIN_INSTALL_DIR:=/home/$ORIN_USER/speech_service}"
: "${ORIN_SERVICE_NAME:=g1-speech}"
: "${DDS_DOMAIN_ID:=0}"
: "${VERIFY_SPEECH_SECONDS:=15}"

PY="$SPARK_AGENT_PYTHON"
[[ -x "$PY" ]] || fail "Spark Agent Python is not executable: $PY"
SSH_TARGET="$ORIN_USER@$ORIN_HOST"

log "Checking Spark -> Orin network"
ip link show "$SPARK_DDS_IFACE" >/dev/null 2>&1 \
    || fail "Spark interface missing: $SPARK_DDS_IFACE"
ping -I "$SPARK_DDS_IFACE" -c 3 -W 2 "$ORIN_HOST" >/dev/null \
    || fail "Spark cannot ping Orin $ORIN_HOST"
ok "Spark -> Orin ping"

log "Checking SSH and Orin -> Spark network"
ssh -o ConnectTimeout=5 "$SSH_TARGET" "true" \
    || fail "SSH to $SSH_TARGET failed"
ssh "$SSH_TARGET" "ip link show '$ORIN_DDS_IFACE' >/dev/null && ping -I '$ORIN_DDS_IFACE' -c 3 -W 2 '$SPARK_HOST' >/dev/null" \
    || fail "Orin cannot reach Spark via $ORIN_DDS_IFACE"
ok "Orin -> Spark ping"

log "Checking both Python environments"
"$PY" -c 'import g1_speech, unitree_sdk2py, cyclonedds; print("Spark imports OK")'
ssh "$SSH_TARGET" "'$ORIN_INSTALL_DIR/.venv/bin/python' -c 'import g1_speech, unitree_sdk2py, cyclonedds, sherpa_onnx, sounddevice; print(\"Orin imports OK\")'" \
    || fail "Orin speech environment check failed"
ok "Dual Python environments"

log "Checking Orin speech service"
if ssh "$SSH_TARGET" "systemctl is-active --quiet '$ORIN_SERVICE_NAME'"; then
    ok "systemd service $ORIN_SERVICE_NAME is active"
elif ssh "$SSH_TARGET" "pgrep -f '[g]1-speech serve' >/dev/null"; then
    ok "manual g1-speech serve process is active"
else
    fail "Orin speech service is not running. Start it with:
  $ORIN_INSTALL_DIR/.venv/bin/g1-speech serve --config $ORIN_INSTALL_DIR/config.json"
fi

log "Checking Orin runtime configuration"
REMOTE_CONFIG="$ORIN_INSTALL_DIR/config.json"
REMOTE_SUMMARY="$(ssh "$SSH_TARGET" "'$ORIN_INSTALL_DIR/.venv/bin/python' -c 'import json; c=json.load(open(\"$REMOTE_CONFIG\")); print(c[\"dds\"][\"domain_id\"], c[\"dds\"][\"network_interface\"], c[\"dds\"][\"speech_topic\"], c[\"dds\"][\"playback_topic\"])'")"
echo "Orin DDS: $REMOTE_SUMMARY"
[[ "$REMOTE_SUMMARY" == "$DDS_DOMAIN_ID $ORIN_DDS_IFACE ${SPEECH_TOPIC:-rt/g1/hri/speech/final} ${PLAYBACK_TOPIC:-rt/g1/hri/playback/state}" ]] \
    || fail "Orin DDS configuration does not match deploy.env"
ok "DDS domain/interface/topics"

log "Interactive speech/final verification"
LOG_FILE="$(mktemp)"
trap 'rm -f "$LOG_FILE"' EXIT
set +e
timeout "${VERIFY_SPEECH_SECONDS}s" \
    "$PY" -u "$ROOT/examples/agent_subscriber.py" "$SPARK_DDS_IFACE" --domain "$DDS_DOMAIN_ID" \
    >"$LOG_FILE" 2>&1 &
SUB_PID=$!
set -e
sleep 2
echo
echo "Speak ONE sentence into the G1 microphone now."
echo "Listening for ${VERIFY_SPEECH_SECONDS}s ..."
wait "$SUB_PID" || true
cat "$LOG_FILE"
grep -q "Agent perception <-" "$LOG_FILE" \
    || fail "No final SpeechEvent received during the verification window"
ok "speech/final Orin -> Spark"

log "Playback gate Spark -> Orin verification"
"$PY" - "$SPARK_DDS_IFACE" "$DDS_DOMAIN_ID" <<'PY'
import sys
import time
import uuid
from g1_speech.dds import DdsPlaybackPublisher, initialize_unitree_dds

initialize_unitree_dds(int(sys.argv[2]), sys.argv[1])
publisher = DdsPlaybackPublisher()
publisher.start()
request_id = "verify-" + str(uuid.uuid4())
try:
    publisher.set_active_reliably(True, request_id=request_id)
    time.sleep(0.5)
    publisher.set_active_reliably(False, request_id=request_id)
finally:
    publisher.close()
print("Playback gate DDS writes OK", request_id)
PY
ok "playback/state Spark -> Orin subscriber matched"

echo
echo "Dual-host automated verification PASS."
echo "Remaining manual checks: actual G1 TTS self-trigger, Agent Observation integration, and one safe action."
