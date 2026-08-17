"""DDS voice adapter for an otherwise transport-agnostic Agent Runtime."""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass

from ..config import ServiceConfig
from ..contracts import SpeechEvent
from ..dds import DdsSpeechSubscriber, DdsTtsPublisher
from .runtime import AgentRuntime

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VoiceBridgeSettings:
    max_speech_age_seconds: float = 5.0
    tts_voice: str = ""
    tts_language: str = ""
    tts_instructions: str = ""

    def validate(self) -> None:
        if self.max_speech_age_seconds <= 0:
            raise ValueError("max_speech_age_seconds must be greater than zero")


class VoiceBridgeAdapter:
    """Serializes ASR turns and streams runtime output to the DDS TTS mouth."""

    def __init__(
        self,
        config: ServiceConfig,
        runtime: AgentRuntime,
        settings: VoiceBridgeSettings,
        *,
        publisher: DdsTtsPublisher | None = None,
        subscriber: DdsSpeechSubscriber | None = None,
    ) -> None:
        settings.validate()
        self.runtime = runtime
        self._settings = settings
        self._publisher = publisher or DdsTtsPublisher(
            topic=config.dds.tts_topic,
            source="local-agent-runtime",
        )
        self._subscriber = subscriber or DdsSpeechSubscriber(
            self._on_speech,
            topic=config.dds.speech_topic,
        )
        self._queue: queue.Queue[SpeechEvent] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._publisher.start()
        self._subscriber.start()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="voice-bridge-adapter",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._subscriber.close()
        self._publisher.close()

    def _on_speech(self, event: SpeechEvent) -> None:
        if not event.is_final or not event.text.strip():
            return
        age_seconds = max(0, time.time_ns() - event.created_unix_ns) / 1_000_000_000
        if age_seconds > self._settings.max_speech_age_seconds:
            logger.warning(
                "Ignoring stale speech turn: event_id=%s age=%.1fs text=%r",
                event.event_id,
                age_seconds,
                event.text,
            )
            return
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            try:
                stale = self._queue.get_nowait()
                logger.warning(
                    "Replacing queued speech turn: stale=%s newest=%s",
                    stale.event_id,
                    event.event_id,
                )
            except queue.Empty:
                pass
            self._queue.put_nowait(event)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                event = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._answer(event)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Agent Runtime turn failed: event_id=%s text=%r",
                    event.event_id,
                    event.text,
                )

    def _answer(self, event: SpeechEvent) -> None:
        request_id = f"agent-{uuid.uuid4().hex}"
        started_ns = time.monotonic_ns()
        first_text_ns: int | None = None
        sequence = 0
        fragments: list[str] = []
        pending_fragment: str | None = None
        tts_started = False

        logger.info("Agent user: %s", event.text)
        try:
            for fragment in self.runtime.stream_response(event.text):
                if self._stop.is_set():
                    if tts_started:
                        self._abort_tts(request_id, sequence)
                    return
                if first_text_ns is None:
                    first_text_ns = time.monotonic_ns()
                fragments.append(fragment)
                # Preserve one-delta look-ahead: there is no new buffering or
                # model call compared with the original bridge hot path.
                if pending_fragment is not None:
                    if not self._publish(
                        request_id,
                        sequence,
                        pending_fragment,
                        timeout=1.0 if sequence == 0 else 0.25,
                    ):
                        raise RuntimeError(
                            "no DDS TTS subscriber matched; is the speech service running?"
                        )
                    tts_started = True
                    sequence += 1
                pending_fragment = fragment

            answer = "".join(fragments).strip()
            if not answer or pending_fragment is None:
                raise RuntimeError("Agent model returned an empty spoken answer")
            if not self._publish(
                request_id,
                sequence,
                pending_fragment,
                is_final=True,
                timeout=1.0,
            ):
                raise RuntimeError("failed to deliver the final TTS chunk over DDS")
            tts_started = True
            sequence += 1
        except Exception:
            if tts_started:
                self._abort_tts(request_id, sequence)
            raise

        # Commit only after the complete answer reached TTS. Failed partial
        # turns never contaminate the memory provider.
        self.runtime.commit_turn(event.text, answer)
        finished_ns = time.monotonic_ns()
        first_ms = (
            0.0
            if first_text_ns is None
            else (first_text_ns - started_ns) / 1_000_000
        )
        logger.info(
            "Agent assistant: %s (first_text=%.1fms total=%.1fms chunks=%d)",
            answer,
            first_ms,
            (finished_ns - started_ns) / 1_000_000,
            sequence,
        )

    def _publish(
        self,
        request_id: str,
        sequence: int,
        text: str,
        *,
        is_final: bool = False,
        timeout: float,
    ) -> bool:
        return self._publisher.publish(
            request_id=request_id,
            sequence=sequence,
            text=text,
            is_final=is_final,
            language=self._settings.tts_language,
            voice=self._settings.tts_voice,
            instructions=self._settings.tts_instructions,
            timeout=timeout,
        )

    def _abort_tts(self, request_id: str, sequence: int) -> None:
        if not self._publisher.publish(
            request_id=request_id,
            sequence=sequence,
            text="",
            is_final=True,
            interrupt=True,
            language=self._settings.tts_language,
            voice=self._settings.tts_voice,
            instructions=self._settings.tts_instructions,
            timeout=0.25,
        ):
            logger.warning("Failed to deliver TTS abort: request_id=%s", request_id)

    def subscriber_topic(self) -> str:
        return str(getattr(self._subscriber, "_topic", "configured"))

    def publisher_topic(self) -> str:
        writer = getattr(self._publisher, "_publisher", None)
        return str(getattr(writer, "_topic_name", "configured"))
