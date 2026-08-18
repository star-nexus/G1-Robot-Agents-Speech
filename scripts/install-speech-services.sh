#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ -f "$ROOT/deploy.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT/deploy.env"
    set +a
fi

SYSTEMD_USER="${SYSTEMD_USER:-$(stat -c '%U' "$ROOT")}"
SYSTEMD_AUDIO_DROPIN="${SYSTEMD_AUDIO_DROPIN:-}"
CPU_TEMPLATE="$ROOT/deploy/systemd/g1-speech.service"
GPU_TEMPLATE="$ROOT/deploy/systemd/g1-speech-gpu.service"
CPU_ROS2_TEMPLATE="$ROOT/deploy/systemd/g1-speech-ros2.service"
GPU_ROS2_TEMPLATE="$ROOT/deploy/systemd/g1-speech-gpu-ros2.service"
ROS2_OVERLAY="${ROS2_WORKSPACE_SETUP:-$ROOT/ros2_ws/install/setup.bash}"
INSTALLED_UNITS=()

fail() { echo "[FAIL] $*" >&2; exit 1; }
render_unit() {
    local source="$1" target="$2" tmp
    tmp="$(mktemp)"
    sed \
        -e "s|@SERVICE_USER@|$SYSTEMD_USER|g" \
        -e "s|@PROJECT_ROOT@|$ROOT|g" \
        "$source" > "$tmp"
    sudo install -m 0644 "$tmp" "/etc/systemd/system/$target"
    rm -f "$tmp"
    INSTALLED_UNITS+=("$target")
}

[[ -x "$ROOT/.venv/bin/g1-speech" ]] || fail "CPU environment is missing: $ROOT/.venv"
[[ -f "$ROOT/config.json" ]] || fail "CPU config is missing: $ROOT/config.json"
[[ -f "$CPU_TEMPLATE" && -f "$GPU_TEMPLATE" ]] || fail "DDS systemd templates are missing"

render_unit "$CPU_TEMPLATE" "g1-speech.service"

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

install_ros2_dropin() {
    local service="$1" destination generated domain_id localhost_only
    destination="/etc/systemd/system/$service.d"
    generated="$(mktemp)"
    domain_id="${ROS2_DOMAIN_ID:-${ROS_DOMAIN_ID:-0}}"
    localhost_only="${ROS_LOCALHOST_ONLY:-0}"
    [[ "$domain_id" =~ ^[0-9]+$ ]] || fail "ROS domain ID must be an integer: $domain_id"
    [[ "$localhost_only" == "0" || "$localhost_only" == "1" ]] || \
        fail "ROS_LOCALHOST_ONLY must be 0 or 1: $localhost_only"
    printf '%s\n' \
        '[Service]' \
        "Environment=ROS_DOMAIN_ID=$domain_id" \
        "Environment=ROS_LOCALHOST_ONLY=$localhost_only" \
        > "$generated"
    if [[ -n "${RMW_IMPLEMENTATION:-}" ]]; then
        printf 'Environment=RMW_IMPLEMENTATION=%s\n' "$RMW_IMPLEMENTATION" >> "$generated"
    fi
    sudo install -d -m 0755 "$destination"
    sudo install -m 0644 "$generated" "$destination/20-ros2.conf"
    rm -f "$generated"
    echo "[PASS] $service uses ROS_DOMAIN_ID=$domain_id"
}

if [[ -x "$ROOT/.venv-gpu/bin/g1-speech" && -f "$ROOT/config.gpu.json" ]]; then
    render_unit "$GPU_TEMPLATE" "g1-speech-gpu.service"
    echo "[PASS] Installed GPU service: g1-speech-gpu.service"
else
    echo "[INFO] GPU environment is not installed; skipped GPU service."
fi

if [[ -f "$ROS2_OVERLAY" ]]; then
    [[ -f "$CPU_ROS2_TEMPLATE" && -f "$GPU_ROS2_TEMPLATE" ]] || fail "ROS 2 systemd templates are missing"
    render_unit "$CPU_ROS2_TEMPLATE" "g1-speech-ros2.service"
    install_ros2_dropin "g1-speech-ros2.service"
    echo "[PASS] Installed ROS 2 CPU service: g1-speech-ros2.service"
    if [[ -x "$ROOT/.venv-gpu/bin/g1-speech" && -f "$ROOT/config.gpu.json" ]]; then
        render_unit "$GPU_ROS2_TEMPLATE" "g1-speech-gpu-ros2.service"
        install_ros2_dropin "g1-speech-gpu-ros2.service"
        echo "[PASS] Installed ROS 2 GPU service: g1-speech-gpu-ros2.service"
    fi
else
    echo "[INFO] ROS 2 workspace is not built; run scripts/setup-ros2.sh to add ROS 2 services."
fi

for service in "${INSTALLED_UNITS[@]}"; do
    install_audio_dropin "$service"
done

sudo install -m 0755 "$ROOT/scripts/g1-speech-service" /usr/local/bin/g1-speech-service
sudo systemctl daemon-reload
echo "[PASS] Installed CPU service: g1-speech.service"
echo "[PASS] Installed selector: /usr/local/bin/g1-speech-service"
