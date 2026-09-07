"""Speech-runtime side Cyclone DDS transport and playback adapters."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from ...core.control import EpochInvalidated
from ...core.events import PlaybackState, TtsTextChunk
from ..config import DdsConfig
from .reliability import DdsEventWriter, RetryingEventSink
from .runtime import _DdsReader, _DdsWriter, initialize_dds
from .types import EpochInvalidatedMessage, PlaybackStateMessage, TtsTextChunkMessage

logger = logging.getLogger(__name__)

class DdsPlaybackSubscriber:
    def __init__(
        self,
        playback_handler: Callable[[bool], None],
        *,
        topic: str = "rt/g1/hri/playback/state",
    ) -> None:
        self._playback_handler = playback_handler
        self._topic = topic
        self._subscriber: _DdsReader | None = None

    def start(self) -> None:
        if self._subscriber is not None:
            return
        self._subscriber = _DdsReader(
            self._topic,
            PlaybackStateMessage,
            self._on_message,
            8,
        )
        self._subscriber.start()
        logger.info("DDS playback gate subscriber: %s", self._topic)

    def close(self) -> None:
        if self._subscriber is not None:
            self._subscriber.close()
            self._subscriber = None

    def _on_message(self, message: PlaybackStateMessage) -> None:
        state = PlaybackState(
            request_id=message.request_id,
            active=bool(message.active),
            created_unix_ns=int(message.created_unix_ns),
            source=message.source,
            session_id=getattr(message, "session_id", ""),
            turn_id=getattr(message, "turn_id", ""),
            epoch=int(getattr(message, "epoch", 0)),
        )
        try:
            self._playback_handler(state)
        except TypeError:
            self._playback_handler(state.active)
        logger.info("Playback gate active=%s request_id=%s", message.active, message.request_id)


class DdsTtsSubscriber:
    def __init__(self, callback: Callable[[Any], None], *, topic: str) -> None:
        self._callback = callback
        self._topic = topic
        self._subscriber: _DdsReader | None = None

    def start(self) -> None:
        if self._subscriber is not None:
            return
        self._subscriber = _DdsReader(
            self._topic,
            TtsTextChunkMessage,
            self._on_message,
            32,
        )
        self._subscriber.start()
        logger.info("DDS TtsTextChunk subscriber: %s", self._topic)

    def close(self) -> None:
        if self._subscriber is not None:
            self._subscriber.close()
            self._subscriber = None

    def _on_message(self, message: TtsTextChunkMessage) -> None:
        self._callback(
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
                session_id=getattr(message, "session_id", ""),
                turn_id=getattr(message, "turn_id", ""),
                epoch=int(getattr(message, "epoch", 0)),
            )
        )

class DdsPlaybackPublisher:
    """Agent-side helper for wrapping TTS/playback with set_active(True/False)."""

    def __init__(self, *, topic: str = "rt/g1/hri/playback/state", source: str = "agent") -> None:
        self._topic = topic
        self._source = source
        self._publisher = _DdsWriter(topic, PlaybackStateMessage)

    def start(self) -> None:
        self._publisher.start()

    def set_active(
        self,
        active: bool,
        *,
        request_id: str,
        timeout: float = 0.5,
        session_id: str = "",
        turn_id: str = "",
        epoch: int = 0,
    ) -> bool:
        message = PlaybackStateMessage(
            request_id=request_id,
            active=active,
            created_unix_ns=time.time_ns(),
            source=self._source,
            session_id=session_id,
            turn_id=turn_id,
            epoch=epoch,
        )
        return self._publisher.write(message, timeout)

    def set_active_reliably(
        self,
        active: bool,
        *,
        request_id: str,
        delivery_timeout: float = 2.0,
        write_timeout: float = 0.25,
        retry_interval: float = 0.1,
    ) -> None:
        """Wait for the playback gate subscriber before reporting success."""
        deadline = time.monotonic() + delivery_timeout
        while time.monotonic() < deadline:
            if self.set_active(active, request_id=request_id, timeout=write_timeout):
                return
            time.sleep(retry_interval)
        raise TimeoutError(
            f"Playback gate was not reached within {delivery_timeout:.1f}s: "
            f"active={active} request_id={request_id}"
        )

    def close(self) -> None:
        self._publisher.close()


class DdsTransport:
    """Complete DDS transport: speech output plus playback-gate input."""

    def __init__(
        self,
        config: DdsConfig,
        playback_handler: Callable[[bool], None],
    ) -> None:
        initialize_dds(config.domain_id, config.network_interface)
        self._playback = DdsPlaybackSubscriber(
            playback_handler,
            topic=config.playback_topic,
        )
        self._config = config
        self._control_writer = _DdsWriter(
            config.control_topic,
            EpochInvalidatedMessage,
        )
        self._tts: DdsTtsSubscriber | None = None
        self._tts_playback: DdsPlaybackPublisher | None = None
        writer = DdsEventWriter(config.speech_topic)
        self.sink = RetryingEventSink(
            writer,
            capacity=config.outbox_capacity,
            write_timeout_seconds=config.write_timeout_seconds,
            retry_interval_seconds=config.retry_interval_seconds,
            delivery_ttl_seconds=config.delivery_ttl_seconds,
        )

    def start(self) -> None:
        self._control_writer.start()
        self._playback.start()
        if self._tts is not None:
            self._tts.start()
        if self._tts_playback is not None:
            self._tts_playback.start()

    def stop(self) -> None:
        self._control_writer.close()
        if self._tts is not None:
            self._tts.close()
        if self._tts_playback is not None:
            self._tts_playback.close()
        self._playback.close()

    def register_tts_handler(self, callback: Callable[[Any], None]) -> None:
        if self._tts is not None:
            raise RuntimeError("TTS handler is already registered")
        self._tts = DdsTtsSubscriber(callback, topic=self._config.tts_topic)
        self._tts_playback = DdsPlaybackPublisher(
            topic=self._config.playback_topic,
            source="g1_speech_tts",
        )

    def publish_playback_state(self, state: PlaybackState) -> None:
        if self._tts_playback is None:
            return
        if not self._tts_playback.set_active(
            state.active,
            request_id=state.request_id,
            session_id=state.session_id,
            turn_id=state.turn_id,
            epoch=state.epoch,
            timeout=0.25,
        ):
            logger.warning(
                "DDS playback state was not delivered: active=%s request_id=%s",
                state.active,
                state.request_id,
            )

    def publish_control(self, event: EpochInvalidated) -> None:
        message = EpochInvalidatedMessage(
            event_id=event.event_id,
            session_id=event.session_id,
            turn_id=event.turn_id,
            epoch=event.epoch,
            next_epoch=event.next_epoch,
            reason=event.reason,
            created_unix_ns=event.created_unix_ns,
            source=event.source,
        )
        if not self._control_writer.write(message, 0.25):
            logger.warning(
                "DDS epoch invalidation was not delivered: turn_id=%s epoch=%d",
                event.turn_id,
                event.epoch,
            )

    def close(self) -> None:
        self.stop()

    def metrics(self) -> dict[str, Any]:
        return {
            "transport_backend": "dds",
            "dds_outbox_size": self.sink.queue_size,
            "dds_delivered": self.sink.delivered,
            "dds_retries": self.sink.retries,
            "dds_expired": self.sink.expired,
            "dds_dropped": self.sink.dropped,
        }


__all__ = [
    "DdsEventWriter", "DdsPlaybackPublisher", "DdsTransport", "DdsTtsSubscriber", "RetryingEventSink",
]
