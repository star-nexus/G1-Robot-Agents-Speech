from __future__ import annotations

import sys
import types

from g1_speech.config import Ros2Config
from g1_speech.contracts import SpeechEvent
from g1_speech.gate import PlaybackGate
from g1_speech.ros2 import (
    Ros2EventSink,
    Ros2SpeechSubscriber,
    Ros2Transport,
    event_to_ros_message,
    ros_message_to_event,
)


class Message:
    pass


class Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class Node:
    def __init__(self):
        self.publisher = Publisher()
        self.subscription_callback = None
        self.destroyed = []

    def create_publisher(self, message_type, topic, qos_depth):
        assert message_type is Message
        assert topic == "/speech"
        assert qos_depth == 7
        return self.publisher

    def create_subscription(self, message_type, topic, callback, qos_depth):
        assert message_type is Message
        assert topic == "/playback"
        assert qos_depth == 7
        self.subscription_callback = callback
        return "subscription"

    def destroy_subscription(self, subscription):
        self.destroyed.append(subscription)

    def destroy_publisher(self, publisher):
        self.destroyed.append(publisher)


def event():
    return SpeechEvent(
        event_id="event-1",
        session_id="session-1",
        sequence=3,
        created_unix_ns=123,
        source="mic",
        text="hello",
        language="en",
        audio_duration_ms=420,
        inference_ms=18.5,
    )


def install_fake_ros_modules(monkeypatch):
    rclpy = types.ModuleType("rclpy")
    rclpy.ok = lambda: True
    messages = types.ModuleType("g1_speech_msgs.msg")
    messages.SpeechEvent = Message
    messages.PlaybackState = Message
    package = types.ModuleType("g1_speech_msgs")
    package.msg = messages
    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    monkeypatch.setitem(sys.modules, "g1_speech_msgs", package)
    monkeypatch.setitem(sys.modules, "g1_speech_msgs.msg", messages)


def test_event_to_ros_message_preserves_complete_contract():
    message = event_to_ros_message(event(), Message)

    assert vars(message) == {
        "event_id": "event-1",
        "session_id": "session-1",
        "sequence": 3,
        "created_unix_ns": 123,
        "source": "mic",
        "text": "hello",
        "language": "en",
        "audio_duration_ms": 420,
        "inference_ms": 18.5,
        "engine": "sensevoice",
        "is_final": True,
    }


def test_ros_message_to_event_preserves_complete_contract():
    message = event_to_ros_message(event(), Message)
    assert ros_message_to_event(message) == event()


def test_agent_side_ros2_subscriber_delivers_transport_neutral_event(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    node = Node()
    received = []
    subscriber = Ros2SpeechSubscriber(
        received.append,
        Ros2Config(speech_topic="/playback", qos_depth=7),
        node=node,
    )

    node.subscription_callback(event_to_ros_message(event(), Message))

    assert received == [event()]
    subscriber.close()
    assert "subscription" in node.destroyed


def test_ros2_event_sink_reports_publish_success_and_errors():
    publisher = Publisher()
    sink = Ros2EventSink(publisher, Message)
    assert not sink.publish(event())
    sink.start()
    assert sink.publish(event())
    assert publisher.messages[0].text == "hello"
    assert sink.published == 1
    assert sink.publish_errors == 0


def test_ros2_transport_publishes_and_applies_playback_gate(monkeypatch):
    install_fake_ros_modules(monkeypatch)
    node = Node()
    gate = PlaybackGate(resume_delay_ms=0)
    transport = Ros2Transport(
        Ros2Config(
            speech_topic="/speech",
            playback_topic="/playback",
            qos_depth=7,
        ),
        gate,
        node=node,
    )
    transport.start()
    transport.sink.start()

    assert transport.sink.publish(event())
    playback = Message()
    playback.active = True
    playback.request_id = "tts-1"
    node.subscription_callback(playback)

    assert gate.active
    assert transport.metrics() == {
        "transport_backend": "ros2",
        "ros2_published": 1,
        "ros2_publish_errors": 0,
    }
    transport.close()
    assert "subscription" in node.destroyed
    assert node.publisher in node.destroyed
