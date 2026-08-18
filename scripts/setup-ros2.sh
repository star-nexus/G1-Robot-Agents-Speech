#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE="$ROOT/ros2_ws"

if [[ -f "$ROOT/deploy.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT/deploy.env"
    set +a
fi

# shellcheck disable=SC1091
source "$ROOT/scripts/ros2-environment.sh"
source_ros2_base

link_package() {
    local source="$1" target="$2"
    if [[ -L "$target" ]]; then
        [[ "$(readlink -f "$target")" == "$(readlink -f "$source")" ]] || {
            echo "[FAIL] Existing symlink points elsewhere: $target" >&2
            exit 1
        }
    elif [[ -e "$target" ]]; then
        echo "[FAIL] Workspace path already exists and is not a symlink: $target" >&2
        exit 1
    else
        ln -s "$source" "$target"
    fi
}

mkdir -p "$WORKSPACE/src"
link_package "$ROOT/integrations/ros2/g1_speech_msgs" "$WORKSPACE/src/g1_speech_msgs"
link_package "$ROOT/integrations/ros2/g1_speech_ros2" "$WORKSPACE/src/g1_speech_ros2"

[[ -x "$ROOT/.venv/bin/python" ]] || {
    echo "[FAIL] CPU environment is missing. Run the CPU setup first." >&2
    exit 1
}

(cd "$WORKSPACE" && "$ROOT/.venv/bin/python" -m colcon build --symlink-install)
source_ros2_overlay "$ROOT"
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$ROOT/.venv/bin/python" -c 'import rclpy; from g1_speech_msgs.msg import SpeechEvent'

bash "$ROOT/scripts/install-speech-services.sh"
echo "[PASS] ROS 2 workspace and speech services are ready."
echo "       Start with: sudo g1-speech-service cpu ros2"
if [[ -x "$ROOT/.venv-gpu/bin/g1-speech" ]]; then
    echo "                   sudo g1-speech-service gpu ros2"
fi
