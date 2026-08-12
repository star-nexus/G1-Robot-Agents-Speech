"""Optional ROS 2 transport.

This module intentionally has no top-level ROS imports. Importing ``g1_speech``
therefore remains safe on systems without ROS 2; dependencies are resolved only
when the ROS 2 transport is selected.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from .config import Ros2Config
from .contracts import EventSink, SpeechEvent
from .gate import PlaybackGate

logger = logging.getLogger(__name__)


def event_to_ros_message(event: SpeechEvent, message_type: type) -> Any:
    message = message_type()
    message.event_id = event.event_id
    message.session_id = event.session_id
    message.sequence = event.sequence
    message.created_unix_ns = event.created_unix_ns
    message.source = event.source
    message.text = event.text
    message.language = event.language
    message.audio_duration_ms = event.audio_duration_ms
    message.inference_ms = event.inference_ms
    message.engine = event.engine
    message.is_final = event.is_final
    return message


class Ros2EventSink(EventSink):
    def __init__(self, publisher: Any, message_type: type) -> None:
        self._publisher = publisher
        self._message_type = message_type
        self._started = False
        self.published = 0
        self.publish_errors = 0

    def start(self) -> None:
        self._started = True

    def publish(self, event: SpeechEvent) -> bool:
        if not self._started:
            return False
        try:
            self._publisher.publish(event_to_ros_message(event, self._message_type))
        except Exception:  # noqa: BLE001
            self.publish_errors += 1
            logger.exception("ROS 2 SpeechEvent publish failed: event_id=%s", event.event_id)
            return False
        self.published += 1
        return True

    def close(self) -> None:
        self._started = False


class Ros2Transport:
    """ROS 2 publisher/subscriber pair for standalone or lifecycle operation."""

    def __init__(
        self,
        config: Ros2Config,
        gate: PlaybackGate,
        *,
        node: Any | None = None,
        lifecycle: bool = False,
    ) -> None:
        try:
            import rclpy
            from g1_speech_msgs.msg import PlaybackState, SpeechEvent as RosSpeechEvent
        except ImportError as exc:
            raise RuntimeError(
                "ROS 2 transport requires a sourced ROS 2 environment and the "
                "g1_speech_msgs package; build the packages under ros2/ with colcon"
            ) from exc

        if lifecycle and node is None:
            raise ValueError("lifecycle ROS 2 transport requires an existing LifecycleNode")

        self._rclpy = rclpy
        self._config = config
        self._gate = gate
        self._lifecycle = lifecycle
        self._owns_node = node is None
        self._owns_context = False
        self._executor = None
        self._executor_thread: threading.Thread | None = None
        self._tts_subscription = None
        self._playback_publisher = None

        if node is None:
            if not rclpy.ok():
                rclpy.init(args=None)
                self._owns_context = True
            node = rclpy.create_node(config.node_name)
        self._node = node

        if lifecycle:
            publisher = node.create_lifecycle_publisher(
                RosSpeechEvent,
                config.speech_topic,
                config.qos_depth,
            )
        else:
            publisher = node.create_publisher(
                RosSpeechEvent,
                config.speech_topic,
                config.qos_depth,
            )
        self._publisher = publisher
        self._subscription = node.create_subscription(
            PlaybackState,
            config.playback_topic,
            self._on_playback,
            config.qos_depth,
        )
        self.sink = Ros2EventSink(publisher, RosSpeechEvent)

    def start(self) -> None:
        if not self._owns_node or self._executor_thread is not None:
            return
        from rclpy.executors import SingleThreadedExecutor

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._executor_thread = threading.Thread(
            target=self._executor.spin,
            name="speech-ros2-executor",
            daemon=True,
        )
        self._executor_thread.start()
        logger.info("ROS 2 transport node started: %s", self._config.node_name)

    def stop(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
        if self._executor_thread is not None:
            self._executor_thread.join(timeout=2.0)
        self._executor = None
        self._executor_thread = None

    def close(self) -> None:
        self.stop()
        if self._subscription is not None:
            self._node.destroy_subscription(self._subscription)
            self._subscription = None
        if self._tts_subscription is not None:
            self._node.destroy_subscription(self._tts_subscription)
            self._tts_subscription = None
        if self._playback_publisher is not None:
            self._node.destroy_publisher(self._playback_publisher)
            self._playback_publisher = None
        if self._publisher is not None:
            if self._lifecycle and hasattr(self._node, "destroy_lifecycle_publisher"):
                self._node.destroy_lifecycle_publisher(self._publisher)
            else:
                self._node.destroy_publisher(self._publisher)
            self._publisher = None
        if self._owns_node:
            self._node.destroy_node()
        if self._owns_context and self._rclpy.ok():
            self._rclpy.shutdown()

    def metrics(self) -> dict[str, Any]:
        return {
            "transport_backend": "ros2",
            "ros2_published": self.sink.published,
            "ros2_publish_errors": self.sink.publish_errors,
        }

    def _on_playback(self, message: Any) -> None:
        self._gate.set_active(bool(message.active))
        logger.info(
            "ROS 2 playback gate active=%s request_id=%s",
            message.active,
            message.request_id,
        )

    def register_tts_handler(self, callback: Any) -> None:
        if self._tts_subscription is not None:
            raise RuntimeError("TTS handler is already registered")
        try:
            from g1_speech_msgs.msg import PlaybackState, TtsTextChunk as RosTtsTextChunk
        except ImportError as exc:
            raise RuntimeError(
                "TTS over ROS 2 requires rebuilding g1_speech_msgs with TtsTextChunk.msg"
            ) from exc
        from .tts import TtsTextChunk

        def on_tts(message: Any) -> None:
            callback(
                TtsTextChunk(
                    request_id=message.request_id,
                    sequence=int(message.sequence),
                    text=message.text,
                    is_final=bool(message.is_final),
                    interrupt=bool(message.interrupt),
                    language=message.language,
                    voice=message.voice,
                    instructions=message.instructions,
                    created_unix_ns=int(message.created_unix_ns),
                    source=message.source,
                )
            )

        self._tts_subscription = self._node.create_subscription(
            RosTtsTextChunk,
            self._config.tts_topic,
            on_tts,
            self._config.qos_depth,
        )
        self._playback_message_type = PlaybackState
        self._playback_publisher = self._node.create_publisher(
            PlaybackState,
            self._config.playback_topic,
            self._config.qos_depth,
        )

    def publish_playback_state(self, active: bool, request_id: str) -> None:
        if self._playback_publisher is None:
            return
        message = self._playback_message_type()
        message.request_id = request_id
        message.active = active
        message.created_unix_ns = time.time_ns()
        message.source = "g1_speech_tts"
        self._playback_publisher.publish(message)
