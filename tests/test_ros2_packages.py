from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_ros2_package_manifests_are_valid_xml_and_declare_runtime_dependencies():
    messages = ET.parse(ROOT / "ros2/g1_speech_msgs/package.xml").getroot()
    node = ET.parse(ROOT / "ros2/g1_speech_ros2/package.xml").getroot()

    assert messages.findtext("name") == "g1_speech_msgs"
    dependencies = {item.text for item in node.findall("exec_depend")}
    assert {"rclpy", "lifecycle_msgs", "g1_speech_msgs"} <= dependencies


def test_ros_speech_event_contract_keeps_all_core_fields():
    fields = {
        line.split()[1]
        for line in (ROOT / "ros2/g1_speech_msgs/msg/SpeechEvent.msg")
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
