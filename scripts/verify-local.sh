#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_FILE="$ROOT/deploy-dgx.env"
LIVE=0

usage() {
    echo "Usage: bash scripts/verify-local.sh [--config FILE] [--live]"
    echo "Without --live, verifies local DDS publish/subscribe using synthetic events."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG_FILE="$2"; shift 2 ;;
        --live) LIVE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

[[ -f "$CONFIG_FILE" ]] || { echo "[FAIL] Missing $CONFIG_FILE" >&2; exit 1; }
set -a
# shellcheck disable=SC1090
source "$CONFIG_FILE"
set +a

PY="$ROOT/.venv/bin/python"
[[ -x "$PY" ]] || { echo "[FAIL] Run setup-dgx.sh first" >&2; exit 1; }
: "${DGX_DDS_IFACE:?Set DGX_DDS_IFACE}"
: "${DDS_DOMAIN_ID:=0}"
: "${CYCLONEDDS_HOME:=$ROOT/.deps/cyclonedds/install}"

PORTAUDIO_LIB_DIR="$(find "$ROOT/.deps/portaudio/root" -name 'libportaudio.so.2' -printf '%h\n' -quit 2>/dev/null || true)"
RUNTIME_LIBRARY_PATH="$CYCLONEDDS_HOME/lib"
[[ -z "$PORTAUDIO_LIB_DIR" ]] || RUNTIME_LIBRARY_PATH="$PORTAUDIO_LIB_DIR:$RUNTIME_LIBRARY_PATH"
export LD_LIBRARY_PATH="$RUNTIME_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="$RUNTIME_LIBRARY_PATH${LIBRARY_PATH:+:$LIBRARY_PATH}"

ip link show "$DGX_DDS_IFACE" >/dev/null
"$PY" -c 'import g1_speech, unitree_sdk2py, cyclonedds, sherpa_onnx, sounddevice; print("[PASS] Python runtime imports")'
"$PY" -m pytest "$ROOT/tests"

echo "[STEP] Local DDS speech publish -> subscribe"
"$PY" - "$DGX_DDS_IFACE" "$DDS_DOMAIN_ID" <<'PY'
import sys
import threading
import time
import uuid

from g1_speech.contracts import SpeechEvent
from g1_speech.dds import DdsSpeechSubscriber, UnitreeDdsEventWriter, initialize_unitree_dds

initialize_unitree_dds(int(sys.argv[2]), sys.argv[1])
received = []
ready = threading.Event()
event_id = "dgx-local-verify-" + str(uuid.uuid4())
subscriber = DdsSpeechSubscriber(lambda event: (received.append(event), ready.set()))
writer = UnitreeDdsEventWriter()
subscriber.start()
writer.start()
try:
    event = SpeechEvent(
        event_id=event_id,
        session_id="verify-local",
        sequence=1,
        created_unix_ns=time.time_ns(),
        source="dgx_verify",
        text="DGX本机DDS验证",
        language="zh",
        audio_duration_ms=100,
        inference_ms=1.0,
    )
    deadline = time.monotonic() + 8
    delivered = False
    while time.monotonic() < deadline and not delivered:
        delivered = writer.write(event, 0.5)
        if not delivered:
            time.sleep(0.2)
    if not delivered or not ready.wait(3):
        raise SystemExit("[FAIL] Local DDS SpeechEvent was not delivered")
    assert received[0].event_id == event_id
    assert received[0].text == "DGX本机DDS验证"
    print("[PASS] Local DDS SpeechEvent", event_id)
finally:
    writer.close()
    subscriber.close()
PY

if [[ "$LIVE" == "1" ]]; then
    if ! systemctl --user is-active --quiet g1-speech.service && ! pgrep -f '[g]1-speech serve' >/dev/null; then
        echo "[FAIL] Start g1-speech before --live verification" >&2
        exit 1
    fi
    echo "[STEP] Speak one sentence into the DGX microphone within 20 seconds"
    if ! "$ROOT/.venv/bin/g1-speech" listen --config "$ROOT/config.json" --timeout 20 --once; then
        echo "[FAIL] No SpeechEvent received. Speak during the window and check that the microphone/transmitter is on and unmuted." >&2
        exit 1
    fi
    echo "[PASS] Live microphone -> ASR -> DDS -> local subscriber"
fi

echo "Local DGX verification PASS."
