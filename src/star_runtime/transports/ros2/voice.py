"""Agent-side ROS 2 implementation of the generic voice input/output ports."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from ...core.events import SpeechEvent
from ..config import Ros2Config
from ..contracts import PublishResult, SpeechEventLike


def message_to_speech_event(message: Any) -> SpeechEvent:
    return SpeechEvent(
        event_id=message.event_id,
        session_id=message.session_id,
        sequence=int(message.sequence),
        created_unix_ns=int(message.created_unix_ns),
        source=message.source,
        text=message.text,
        language=message.language,
        audio_duration_ms=int(message.audio_duration_ms),
        inference_ms=float(message.inference_ms),
        engine=message.engine,
        is_final=bool(message.is_final),
    )


class Ros2VoicePort:
    """One shared ROS 2 node serving both Agent ears and mouth."""

    def __init__(
        self,
        config: Ros2Config,
        *,
        node: Any | None = None,
        node_name: str = "star_agent_voice",
        source: str = "star-agent-runtime",
    ) -> None:
        try:
            import rclpy
            from g1_speech_msgs.msg import SpeechEvent as RosSpeechEvent
            from g1_speech_msgs.msg import TtsTextChunk
        except ImportError as exc:
            raise RuntimeError(
                "ROS 2 voice transport requires a sourced ROS environment and "
                "the g1_speech_msgs package"
            ) from exc

        self._rclpy = rclpy
        self._config = config
        self._source = source
        self._handler: Callable[[SpeechEventLike], None] = lambda _event: None
        self._owns_node = node is None
        self._owns_context = False
        self._executor: Any | None = None
        self._thread: threading.Thread | None = None
        self._closed = False

        if node is None:
            if not rclpy.ok():
                rclpy.init(args=None)
                self._owns_context = True
            node = rclpy.create_node(node_name)
        self._node = node
        self._message_type = TtsTextChunk
        self._publisher = node.create_publisher(
            TtsTextChunk,
            config.tts_topic,
            config.qos_depth,
        )
        self._subscription = node.create_subscription(
            RosSpeechEvent,
            config.speech_topic,
            self._on_speech,
            config.qos_depth,
        )

    @property
    def endpoint(self) -> str:
        return f"ros2:{self._config.speech_topic}->{self._config.tts_topic}"

    def set_handler(self, handler: Callable[[SpeechEventLike], None]) -> None:
        if self._thread is not None:
            raise RuntimeError("cannot replace ROS 2 speech handler after start")
        self._handler = handler

    def start(self) -> None:
        if not self._owns_node or self._thread is not None:
            return
        from rclpy.executors import SingleThreadedExecutor

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(
            target=self._executor.spin,
            name="star-agent-ros2-voice",
            daemon=True,
        )
        self._thread.start()

    def publish(
        self,
        *,
        request_id: str,
        sequence: int,
        text: str,
        is_final: bool = False,
        interrupt: bool = False,
        language: str = "",
        voice: str = "",
        instructions: str = "",
        timeout: float = 0.25,
    ) -> PublishResult:
        del timeout  # ROS 2 publish is asynchronous.
        if self._closed:
            return PublishResult(False, detail="ROS 2 voice port is closed")
        message = self._message_type()
        message.request_id = request_id
        message.sequence = sequence
        message.text = text
        message.is_final = is_final
        message.interrupt = interrupt
        message.language = language
        message.voice = voice
        message.instructions = instructions
        message.created_unix_ns = time.time_ns()
        message.source = self._source
        self._publisher.publish(message)
        return PublishResult(True, delivered=None, detail="queued by ROS 2 publisher")

    def close(self) -> None:
        if self._closed:
            return
        if self._executor is not None:
            self._executor.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._executor = None
        self._thread = None
        self._node.destroy_subscription(self._subscription)
        self._node.destroy_publisher(self._publisher)
        if self._owns_node:
            self._node.destroy_node()
        if self._owns_context and self._rclpy.ok():
            self._rclpy.shutdown()
        self._closed = True

    def _on_speech(self, message: Any) -> None:
        self._handler(message_to_speech_event(message))
