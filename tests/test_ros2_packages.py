from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_ros2_package_manifests_are_valid_xml_and_declare_runtime_dependencies():
    messages = ET.parse(
        ROOT / "integrations/ros2/g1_speech_msgs/package.xml"
    ).getroot()
    node = ET.parse(
        ROOT / "integrations/ros2/g1_speech_ros2/package.xml"
    ).getroot()

    assert messages.findtext("name") == "g1_speech_msgs"
    dependencies = {item.text for item in node.findall("exec_depend")}
    assert {"rclpy", "rcl_interfaces", "lifecycle_msgs", "g1_speech_msgs"} <= dependencies


def test_ros_speech_event_contract_keeps_all_core_fields():
    fields = {
        line.split()[1]
        for line in (
            ROOT / "integrations/ros2/g1_speech_msgs/msg/SpeechEvent.msg"
        )
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    }

    assert fields == {
        "event_id",
        "session_id",
        "sequence",
        "created_unix_ns",
        "source",
        "text",
        "language",
        "audio_duration_ms",
        "inference_ms",
        "engine",
        "is_final",
    }


def test_lifecycle_launch_autostart_is_transition_driven():
    launch = (
        ROOT
        / "integrations/ros2/g1_speech_ros2/launch/speech_lifecycle.launch.py"
    ).read_text(encoding="utf-8")

    assert 'DeclareLaunchArgument(\n                "autostart"' in launch
    assert "OnProcessStart" in launch
    assert "OnStateTransition" in launch
    assert "TRANSITION_CONFIGURE" in launch
    assert "TRANSITION_ACTIVATE" in launch
    assert "TimerAction" not in launch


def test_lifecycle_config_path_is_protected_after_configure():
    node = (
        ROOT
        / "integrations/ros2/g1_speech_ros2/g1_speech_ros2/lifecycle_node.py"
    ).read_text(encoding="utf-8")

    assert "add_on_set_parameters_callback" in node
    assert "self._controller.configured" in node
    assert "config_file can only be changed" in node
    assert "is Unconfigured; run cleanup first" in node


def test_real_lifecycle_acceptance_covers_reactivation_and_parameter_policy():
    acceptance = (ROOT / "tests/acceptance/ros2_lifecycle.py").read_text(
        encoding="utf-8"
    )

    assert acceptance.count("Transition.TRANSITION_ACTIVATE") == 2
    assert acceptance.count("Transition.TRANSITION_DEACTIVATE") == 2
    assert "Transition.TRANSITION_CLEANUP" in acceptance
    assert 'output.count("Loading SenseVoice:") != 1' in acceptance
    assert 'output.count("Microphone started:") != 2' in acceptance
