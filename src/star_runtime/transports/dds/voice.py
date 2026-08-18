"""Agent-side Cyclone DDS ears and mouth adapters."""

from __future__ import annotations

import logging
import time
from typing import Callable

from ...core.events import SpeechEvent
from ..contracts import PublishResult
from .codec import message_to_event
from .reliability import EventDeduplicator
from .runtime import _DdsReader, _DdsWriter, initialize_dds
from .types import SpeechEventMessage, TtsTextChunkMessage

logger = logging.getLogger(__name__)

class DdsSpeechSubscriber:
    """Agent-side subscriber. Duplicate retries never reach the Agent callback."""

    def __init__(
        self,
        callback: Callable[[SpeechEvent], None],
        *,
        topic: str = "rt/g1/hri/speech/final",
        deduplicator: EventDeduplicator | None = None,
        queue_len: int = 32,
    ) -> None:
        self._callback = callback
        self._topic = topic
        self._dedupe = deduplicator or EventDeduplicator()
        self._queue_len = queue_len
        self._subscriber: _DdsReader | None = None
        self.duplicates = 0

    @property
    def endpoint(self) -> str:
        return self._topic

    def set_handler(self, callback: Callable[[SpeechEvent], None]) -> None:
        if self._subscriber is not None:
            raise RuntimeError("cannot replace DDS speech handler after start")
        self._callback = callback

    def start(self) -> None:
        if self._subscriber is not None:
            return
        self._subscriber = _DdsReader(
            self._topic,
            SpeechEventMessage,
            self._on_message,
            self._queue_len,
        )
        self._subscriber.start()
        logger.info("DDS SpeechEvent subscriber: %s", self._topic)

    def close(self) -> None:
        if self._subscriber is not None:
            self._subscriber.close()
            self._subscriber = None

    def _on_message(self, message: SpeechEventMessage) -> None:
        event = message_to_event(message)
        if not self._dedupe.accept(event.event_id):
            self.duplicates += 1
            logger.info("Ignoring duplicate SpeechEvent: %s", event.event_id)
            return
        logger.info(
            "DDS SpeechEvent received: event_id=%s created_to_received=%.1fms",
            event.event_id,
            max(0, time.time_ns() - event.created_unix_ns) / 1_000_000,
        )
        self._callback(event)

class DdsTtsPublisher:
    """Agent-side incremental text publisher for the robot mouth."""

    def __init__(self, *, topic: str = "rt/g1/hri/tts/request", source: str = "agent") -> None:
        self._source = source
        self._topic = topic
        self._publisher = _DdsWriter(topic, TtsTextChunkMessage)

    @property
    def endpoint(self) -> str:
        return self._topic

    def start(self) -> None:
        self._publisher.start()

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
        accepted = self._publisher.write(
            TtsTextChunkMessage(
                request_id=request_id,
                sequence=sequence,
                text=text,
                is_final=is_final,
                interrupt=interrupt,
                language=language,
                voice=voice,
                instructions=instructions,
                created_unix_ns=time.time_ns(),
                source=self._source,
            ),
            timeout,
        )
        return PublishResult(
            accepted,
            matched=accepted,
            delivered=accepted if accepted else None,
            detail="written by Cyclone DDS" if accepted else "no matched DDS reader",
        )

    def close(self) -> None:
        self._publisher.close()


__all__ = [
    "DdsSpeechSubscriber", "DdsTtsPublisher", "EventDeduplicator", "initialize_dds",
]

