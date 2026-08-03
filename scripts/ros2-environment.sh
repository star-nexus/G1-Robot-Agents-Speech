#!/usr/bin/env bash

resolve_ros2_setup() {
    local candidate
    local -a candidates=()

    if [[ -n "${ROS2_SETUP:-}" ]]; then
        [[ -f "$ROS2_SETUP" ]] || {
            echo "[FAIL] ROS2_SETUP does not exist: $ROS2_SETUP" >&2
            return 1
        }
        printf '%s\n' "$ROS2_SETUP"
        return 0
    fi

    if [[ -n "${ROS_DISTRO:-}" ]]; then
        candidate="/opt/ros/$ROS_DISTRO/setup.bash"
        if [[ -f "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    fi

    shopt -s nullglob
    candidates=(/opt/ros/*/setup.bash)
    shopt -u nullglob
    case "${#candidates[@]}" in
        0)
            echo "[FAIL] ROS 2 was not found under /opt/ros. Install ROS 2 or set ROS2_SETUP." >&2
            return 1
            ;;
        1)
            printf '%s\n' "${candidates[0]}"
            ;;
        *)
            echo "[FAIL] Multiple ROS 2 distributions found. Set ROS2_SETUP in deploy.env." >&2
            return 1
            ;;
    esac
}

source_ros2_base() {
    local setup
    setup="$(resolve_ros2_setup)" || return 1
    set +u
    # shellcheck disable=SC1090
    source "$setup"
    set -u
}

source_ros2_overlay() {
    local project_root="$1"
    local overlay="${ROS2_WORKSPACE_SETUP:-$project_root/ros2_ws/install/setup.bash}"
    [[ -f "$overlay" ]] || {
        echo "[FAIL] ROS 2 workspace is not built: $overlay" >&2
        echo "       Run: bash $project_root/scripts/setup-ros2.sh" >&2
        return 1
    }
    set +u
    # shellcheck disable=SC1090
    source "$overlay"
    set -u
    export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
}
