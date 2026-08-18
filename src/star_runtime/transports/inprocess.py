"""Zero-serialization voice transport for one integrated Runtime process."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from ..core.events import SpeechEvent, TtsTextChunk
from .contracts import PublishResult, SpeechEventLike


class _InProcessEventSink:
    def __init__(self, transport: InProcessTransport) -> None:
        self._transport = transport
        self._started = False

    def start(self) -> None:
        self._started = True

    def publish(self, event: SpeechEvent) -> PublishResult:
        if not self._started:
            return PublishResult(False, detail="in-process speech sink is not started")
        return self._transport._deliver_speech(event)

    def close(self) -> None:
        self._started = False


class InProcessVoicePort:
    """Agent-side ears and mouth connected directly to the speech runtime."""

    endpoint = "inprocess://voice"

    def __init__(self, transport: InProcessTransport) -> None:
        self._transport = transport
        self._started = False

    def set_handler(self, handler: Callable[[SpeechEventLike], None]) -> None:
        self._transport._set_speech_handler(handler)

    def start(self) -> None:
        self._started = True

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
        del timeout
        if not self._started:
            return PublishResult(False, detail="in-process voice port is not started")
        return self._transport._deliver_tts(
            TtsTextChunk(
                request_id=request_id,
                sequence=sequence,
                text=text,
                is_final=is_final,
                interrupt=interrupt,
                language=language,
                voice=voice,
                instructions=instructions,
                created_unix_ns=time.time_ns(),
                source="star-agent-runtime",
            )
        )

    def close(self) -> None:
        self._started = False


class InProcessTransport:
    """Direct callbacks for the memory-constrained, single-process deployment.

    The callbacks only enqueue work in their consumers, so ASR, Agent and TTS
    retain their existing worker boundaries without serialization, discovery,
    middleware threads, or a second copy of an event payload.
    """

    def __init__(self, playback_handler: Callable[[bool], None]) -> None:
        self._playback_handler = playback_handler
        self._speech_handler: Callable[[SpeechEventLike], None] | None = None
        self._tts_handler: Callable[[TtsTextChunk], None] | None = None
        self._lock = threading.Lock()
        self._started = False
        self._speech_delivered = 0
        self._tts_delivered = 0
        self._delivery_errors = 0
        self.sink = _InProcessEventSink(self)
        self.voice = InProcessVoicePort(self)

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def close(self) -> None:
        self.stop()

    def register_tts_handler(self, callback: Callable[[TtsTextChunk], None]) -> None:
        with self._lock:
            if self._tts_handler is not None:
                raise RuntimeError("TTS handler is already registered")
            self._tts_handler = callback

    def publish_playback_state(self, active: bool, request_id: str) -> None:
        del request_id
        self._playback_handler(active)

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "transport_backend": "inprocess",
                "inprocess_speech_delivered": self._speech_delivered,
                "inprocess_tts_delivered": self._tts_delivered,
                "inprocess_delivery_errors": self._delivery_errors,
            }

    def _set_speech_handler(
        self, handler: Callable[[SpeechEventLike], None]
    ) -> None:
        with self._lock:
            self._speech_handler = handler

    def _deliver_speech(self, event: SpeechEvent) -> PublishResult:
        with self._lock:
            handler = self._speech_handler
        if not self._started:
            return PublishResult(False, detail="in-process transport is not started")
        if handler is None:
            return PublishResult(False, detail="Agent speech handler is not registered")
        try:
            handler(event)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._delivery_errors += 1
            return PublishResult(False, matched=True, delivered=False, detail=str(exc))
        with self._lock:
            self._speech_delivered += 1
        return PublishResult(True, matched=True, delivered=True)

    def _deliver_tts(self, chunk: TtsTextChunk) -> PublishResult:
        with self._lock:
            handler = self._tts_handler
        if not self._started:
            return PublishResult(False, detail="in-process transport is not started")
        if handler is None:
            return PublishResult(False, detail="speech TTS handler is not registered")
        try:
            handler(chunk)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._delivery_errors += 1
            return PublishResult(False, matched=True, delivered=False, detail=str(exc))
        with self._lock:
            self._tts_delivered += 1
        return PublishResult(True, matched=True, delivered=True)


__all__ = ["InProcessTransport", "InProcessVoicePort"]
