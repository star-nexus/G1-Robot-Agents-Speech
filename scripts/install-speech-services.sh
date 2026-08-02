#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SYSTEMD_USER="${ORIN_SYSTEMD_USER:-$(stat -c '%U' "$ROOT")}"
CPU_TEMPLATE="$ROOT/systemd/g1-speech.service"
GPU_TEMPLATE="$ROOT/systemd/g1-speech-gpu.service"

fail() { echo "[FAIL] $*" >&2; exit 1; }
render_unit() {
    local source="$1" target="$2" venv="$3" config="$4" tmp
    tmp="$(mktemp)"
    sed \
        -e "s|^User=.*|User=$SYSTEMD_USER|" \
        -e "s|^WorkingDirectory=.*|WorkingDirectory=$ROOT|" \
        -e "s|^ExecStart=.*|ExecStart=$ROOT/$venv/bin/g1-speech serve --config $ROOT/$config|" \
        "$source" > "$tmp"
    sudo install -m 0644 "$tmp" "/etc/systemd/system/$target"
    rm -f "$tmp"
}

[[ -x "$ROOT/.venv/bin/g1-speech" ]] || fail "CPU environment is missing: $ROOT/.venv"
[[ -f "$ROOT/config.json" ]] || fail "CPU config is missing: $ROOT/config.json"
[[ -f "$CPU_TEMPLATE" && -f "$GPU_TEMPLATE" ]] || fail "systemd templates are missing"

render_unit "$CPU_TEMPLATE" "g1-speech.service" ".venv" "config.json"

if [[ -x "$ROOT/.venv-gpu/bin/g1-speech" && -f "$ROOT/config.gpu.json" ]]; then
    render_unit "$GPU_TEMPLATE" "g1-speech-gpu.service" ".venv-gpu" "config.gpu.json"
    cpu_audio_dropin="/etc/systemd/system/g1-speech.service.d/10-hollyland-mic.conf"
    if [[ -f "$cpu_audio_dropin" ]]; then
        sudo install -d -m 0755 /etc/systemd/system/g1-speech-gpu.service.d
        sudo install -m 0644 "$cpu_audio_dropin" \
            /etc/systemd/system/g1-speech-gpu.service.d/10-hollyland-mic.conf
    fi
    echo "[PASS] Installed GPU service: g1-speech-gpu.service"
else
    echo "[INFO] GPU environment is not installed; skipped GPU service."
fi

sudo install -m 0755 "$ROOT/scripts/g1-speech-service" /usr/local/bin/g1-speech-service
sudo systemctl daemon-reload
echo "[PASS] Installed CPU service: g1-speech.service"
echo "[PASS] Installed selector: /usr/local/bin/g1-speech-service"
