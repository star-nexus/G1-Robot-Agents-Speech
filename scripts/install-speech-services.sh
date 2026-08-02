#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SYSTEMD_USER="${SYSTEMD_USER:-$(stat -c '%U' "$ROOT")}"
SYSTEMD_AUDIO_DROPIN="${SYSTEMD_AUDIO_DROPIN:-}"
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

install_audio_dropin() {
    local service="$1" destination source uid runtime_dir generated=""
    destination="/etc/systemd/system/$service.d"
    source="$SYSTEMD_AUDIO_DROPIN"
    if [[ -z "$source" ]]; then
        uid="$(id -u "$SYSTEMD_USER")"
        runtime_dir="/run/user/$uid"
        if [[ ! -S "$runtime_dir/pulse/native" ]]; then
            sudo rm -f "$destination/10-audio.conf"
            return 0
        fi
        generated="$(mktemp)"
        printf '%s\n' \
            '[Service]' \
            "Environment=XDG_RUNTIME_DIR=$runtime_dir" \
            "Environment=PULSE_SERVER=unix:$runtime_dir/pulse/native" \
            > "$generated"
        source="$generated"
    fi
    [[ -f "$source" ]] || fail "Audio systemd drop-in not found: $source"
    sudo install -d -m 0755 "$destination"
    sudo install -m 0644 "$source" "$destination/10-audio.conf"
    [[ -z "$generated" ]] || rm -f "$generated"
}

install_audio_dropin "g1-speech.service"

if [[ -x "$ROOT/.venv-gpu/bin/g1-speech" && -f "$ROOT/config.gpu.json" ]]; then
    render_unit "$GPU_TEMPLATE" "g1-speech-gpu.service" ".venv-gpu" "config.gpu.json"
    install_audio_dropin "g1-speech-gpu.service"
    echo "[PASS] Installed GPU service: g1-speech-gpu.service"
else
    echo "[INFO] GPU environment is not installed; skipped GPU service."
fi

sudo install -m 0755 "$ROOT/scripts/g1-speech-service" /usr/local/bin/g1-speech-service
sudo systemctl daemon-reload
echo "[PASS] Installed CPU service: g1-speech.service"
echo "[PASS] Installed selector: /usr/local/bin/g1-speech-service"
