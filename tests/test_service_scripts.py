from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SYSTEMD = ROOT / "systemd"


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
    assert "MICROPHONE_DEVICE in deploy.env" in selector
    assert "(pinned)" in selector


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


def test_ros2_systemd_installation_preserves_the_selected_domain():
    installer = (ROOT / "scripts" / "install-speech-services.sh").read_text()
    runner = (ROOT / "scripts" / "run-speech-service").read_text()
    assert "Environment=ROS_DOMAIN_ID=$domain_id" in installer
    assert 'export ROS_DOMAIN_ID="$ROS2_DOMAIN_ID"' in runner


def test_voice_chat_runner_inherits_active_ros2_service_domain():
    runner = (ROOT / "scripts" / "run-qwen-voice-chat").read_text()
    assert "active_ros2_unit" in runner
    assert '--property=Environment --value' in runner
    assert 'ROS_DOMAIN_ID="${assignment#ROS_DOMAIN_ID=}"' in runner


def test_qwen_vl_server_runner_has_bounded_jetson_memory_defaults():
    runner = (ROOT / "scripts" / "run-qwen-vl-server").read_text()
    assert 'CONTEXT="${QWEN_VL_CONTEXT:-1536}"' in runner
    assert 'BATCH="${QWEN_VL_BATCH:-512}"' in runner
    assert 'UBATCH="${QWEN_VL_UBATCH:-512}"' in runner
    assert 'KV_TYPE="${QWEN_VL_KV_TYPE:-q8_0}"' in runner
    assert "--cache-ram 0" in runner
    assert "--no-cache-prompt" in runner
    assert "--no-cache-idle-slots" in runner
    assert "--ctx-checkpoints 0" in runner
    assert "-np 1" in runner
    assert "/home/nvidia" not in runner
