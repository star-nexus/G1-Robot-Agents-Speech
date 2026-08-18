import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SYSTEMD = ROOT / "deploy" / "systemd"


MODES = {
    "g1-speech.service": ("cpu", "dds"),
    "g1-speech-gpu.service": ("gpu", "dds"),
    "g1-speech-ros2.service": ("cpu", "ros2"),
    "g1-speech-gpu-ros2.service": ("gpu", "ros2"),
}


def test_all_service_modes_are_mutually_exclusive_and_use_runner():
    names = set(MODES)
    for name, (backend, transport) in MODES.items():
        unit = (SYSTEMD / name).read_text()
        assert f"run-speech-service {backend} {transport}" in unit
        conflicts = next(
            line.removeprefix("Conflicts=").split()
            for line in unit.splitlines()
            if line.startswith("Conflicts=")
        )
        assert set(conflicts) == names - {name}
        assert "StartLimitIntervalSec=30" in unit
        assert "StartLimitBurst=3" in unit


def test_selector_exposes_transport_without_breaking_backend_defaults():
    selector = (ROOT / "scripts" / "g1-speech-service").read_text()
    assert 'select_mode "$command" "${2:-dds}"' in selector
    assert "cpu:ros2" in selector
    assert "gpu:ros2" in selector
    assert "logs [cpu|gpu] [dds|ros2]" in selector


def test_status_and_logs_report_the_effective_microphone():
    selector = (ROOT / "scripts" / "g1-speech-service").read_text()
    assert 'show_microphone "$unit"' in selector
    assert "pactl get-default-source" in selector
    assert "AUDIO_INPUT_BACKEND" in selector
    assert "ALSA_INPUT_CARD" in selector
    assert "PULSE_SOURCE" in selector
    assert "Actual backend/device" in selector
    assert "(pinned)" in selector


def test_logs_follow_selected_service_while_it_is_auto_restarting(tmp_path):
    systemctl = tmp_path / "systemctl"
    systemctl.write_text(
        """#!/usr/bin/env bash
case "$1:$2" in
  is-active:g1-speech-gpu.service) echo activating; exit 3 ;;
  is-active:*) echo inactive; exit 3 ;;
  is-enabled:g1-speech-gpu.service) exit 0 ;;
  is-enabled:*) exit 1 ;;
  show:*) exit 0 ;;
esac
exit 1
"""
    )
    systemctl.chmod(0o755)
    journalctl = tmp_path / "journalctl"
    journalctl.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*"\n')
    journalctl.chmod(0o755)

    environment = os.environ.copy()
    environment["PATH"] = f"{tmp_path}:{environment['PATH']}"
    result = subprocess.run(
        [ROOT / "scripts" / "g1-speech-service", "logs"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0
    assert "-u g1-speech-gpu.service -f" in result.stdout


def test_ros2_runner_discovers_environment_instead_of_hardcoding_distribution():
    runner = (ROOT / "scripts" / "run-speech-service").read_text()
    environment = (ROOT / "scripts" / "ros2-environment.sh").read_text()
    assert "source_ros2_base" in runner
    assert "source_ros2_overlay" in runner
    assert "/opt/ros/humble" not in runner + environment
    assert "ROS2_SETUP" in environment
    assert "ROS2_WORKSPACE_SETUP" in environment


def test_runner_can_switch_asr_configuration_without_changing_systemd_units():
    runner = (ROOT / "scripts" / "run-speech-service").read_text()
    assert "SPEECH_CONFIG_CPU" in runner
    assert "SPEECH_CONFIG_GPU" in runner
    assert "SPEECH_PYTHON_GPU" in runner
    assert 'export PYTHONPATH="$ROOT/src' in runner
    assert 'export CPATH="$CUDA_INCLUDE' in runner
    assert "SPEECH_GPU_LIBRARY_PATH" in runner
    assert "QWEN3_CUDSS_DIR" in runner  # compatibility with existing hosts
    assert 'CYCLONEDDS_HOME/lib/libddsc.so' in runner
    assert "scripts/install-cyclonedds.sh" in runner


def test_ros2_systemd_installation_preserves_the_selected_domain():
    installer = (ROOT / "scripts" / "install-speech-services.sh").read_text()
    runner = (ROOT / "scripts" / "run-speech-service").read_text()
    assert "Environment=ROS_DOMAIN_ID=$domain_id" in installer
    assert 'export ROS_DOMAIN_ID="$ROS2_DOMAIN_ID"' in runner


def test_orin_tts_launcher_uses_graph_safe_attention_and_forwards_overrides():
    launcher = (ROOT / "scripts" / "run-qwen3-tts-vllm-omni.sh").read_text()

    assert 'QWEN3_TTS_ATTENTION_BACKEND:-TRITON_ATTN' in launcher
    assert '--attention-backend "$ATTENTION_BACKEND"' in launcher
    assert '"$@"' in launcher
