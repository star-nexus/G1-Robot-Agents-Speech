#!/usr/bin/env python3
"""Real ROS 2 graph acceptance test for both speech transport directions."""

from __future__ import annotations

import argparse
import time

import rclpy
from g1_speech_msgs.msg import PlaybackState, SpeechEvent as RosSpeechEvent
from rclpy.executors import SingleThreadedExecutor

from g1_speech.config import Ros2Config
from g1_speech.contracts import SpeechEvent
from g1_speech.gate import PlaybackGate
from g1_speech.ros2 import Ros2Transport


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()
    if args.timeout <= 0:
        raise ValueError("timeout must be greater than zero")

    rclpy.init()
    config = Ros2Config(node_name="g1_speech_acceptance")
    gate = PlaybackGate(resume_delay_ms=0)
    service_node = rclpy.create_node(config.node_name)
    agent_node = rclpy.create_node("g1_speech_acceptance_agent")
    transport = Ros2Transport(config, gate, node=service_node)
    received: list[RosSpeechEvent] = []
    speech_subscriber = agent_node.create_subscription(
        RosSpeechEvent,
        config.speech_topic,
        received.append,
        config.qos_depth,
    )
    playback_publisher = agent_node.create_publisher(
        PlaybackState,
        config.playback_topic,
        config.qos_depth,
    )
    executor = SingleThreadedExecutor()
    executor.add_node(service_node)
    executor.add_node(agent_node)

    try:
        transport.sink.start()
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            if (
                service_node.count_subscribers(config.speech_topic) > 0
                and agent_node.count_subscribers(config.playback_topic) > 0
            ):
                break
            executor.spin_once(timeout_sec=0.05)
        else:
            raise TimeoutError("ROS 2 publishers and subscribers did not match")

        event = SpeechEvent(
            event_id="ros2-acceptance-event",
            session_id="ros2-acceptance-session",
            sequence=7,
            created_unix_ns=123456789,
            source="acceptance",
            text="ROS 2 round trip",
            language="en",
            audio_duration_ms=640,
            inference_ms=12.5,
        )
        if not transport.sink.publish(event):
            raise RuntimeError("ROS 2 event sink rejected the acceptance event")

        playback = PlaybackState()
        playback.request_id = "ros2-acceptance-playback"
        playback.active = True
        playback.created_unix_ns = 987654321
        playback.source = "acceptance"
        playback_publisher.publish(playback)

        while time.monotonic() < deadline and (not received or not gate.active):
            executor.spin_once(timeout_sec=0.05)
        if not received:
            raise TimeoutError("SpeechEvent was not received")
        if not gate.active:
            raise TimeoutError("PlaybackState did not reach the playback gate")

        message = received[0]
        expected = {
            "event_id": event.event_id,
            "session_id": event.session_id,
            "sequence": event.sequence,
            "created_unix_ns": event.created_unix_ns,
            "source": event.source,
            "text": event.text,
            "language": event.language,
            "audio_duration_ms": event.audio_duration_ms,
            "inference_ms": event.inference_ms,
            "engine": event.engine,
            "is_final": event.is_final,
        }
        actual = {name: getattr(message, name) for name in expected}
        if actual != expected:
            raise AssertionError(f"SpeechEvent mismatch: {actual!r} != {expected!r}")
        print("[PASS] ROS 2 SpeechEvent and PlaybackState round trip")
        return 0
    finally:
        transport.close()
        agent_node.destroy_subscription(speech_subscriber)
        agent_node.destroy_publisher(playback_publisher)
        executor.shutdown()
        service_node.destroy_node()
        agent_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
