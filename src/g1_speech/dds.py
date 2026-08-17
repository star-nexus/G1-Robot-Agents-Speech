"""Cyclone DDS transport, reliable delivery, and Agent-side deduplication."""

from __future__ import annotations

import html
import logging
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .contracts import EventSink, SpeechEvent
from .config import DdsConfig
from .dds_types import (
    DDS_IDL_AVAILABLE,
    PlaybackStateMessage,
    SpeechEventMessage,
    TtsTextChunkMessage,
)
from .gate import PlaybackGate

logger = logging.getLogger(__name__)

_runtime_lock = threading.Lock()
_dds_domain: Any = None
_dds_participant: Any = None
_dds_settings: tuple[int, str | None] | None = None


def _domain_config(network_interface: str | None) -> str:
    if network_interface:
        interface = html.escape(network_interface, quote=True)
        selection = (
            f'<NetworkInterface name="{interface}" priority="default" multicast="default"/>'
        )
    else:
        selection = (
            '<NetworkInterface autodetermine="true" priority="default" multicast="default"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<CycloneDDS><Domain Id="any"><General><Interfaces>'
        f"{selection}"
        '</Interfaces></General></Domain></CycloneDDS>'
    )


def initialize_dds(domain_id: int = 0, network_interface: str | None = None) -> Any:
    """Initialize the process-wide Cyclone DDS participant once."""

    global _dds_domain, _dds_participant, _dds_settings
    if not DDS_IDL_AVAILABLE:
        raise RuntimeError("cyclonedds is not installed; install the DDS runtime first")

    settings = (domain_id, network_interface)
    with _runtime_lock:
        if _dds_participant is not None:
            if settings != _dds_settings:
                raise RuntimeError(
                    "DDS is already initialized with different settings: "
                    f"current={_dds_settings!r}, requested={settings!r}"
                )
            return _dds_participant

        from cyclonedds.domain import Domain, DomainParticipant

        _dds_domain = Domain(domain_id, _domain_config(network_interface))
        _dds_participant = DomainParticipant(domain_id)
        _dds_settings = settings
        logger.info(
            "Cyclone DDS initialized: domain_id=%d network_interface=%r",
            domain_id,
            network_interface,
        )
        return _dds_participant


def _participant() -> Any:
    if _dds_participant is None:
        raise RuntimeError("DDS is not initialized; call initialize_dds() first")
    return _dds_participant


def event_to_message(event: SpeechEvent) -> SpeechEventMessage:
    return SpeechEventMessage(
        event_id=event.event_id,
        session_id=event.session_id,
        sequence=event.sequence,
        created_unix_ns=event.created_unix_ns,
        source=event.source,
        text=event.text,
        language=event.language,
        audio_duration_ms=event.audio_duration_ms,
        inference_ms=event.inference_ms,
        engine=event.engine,
        is_final=event.is_final,
    )


def message_to_event(message: SpeechEventMessage) -> SpeechEvent:
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


class _DdsWriter:
    def __init__(self, topic_name: str, data_type: Any) -> None:
        self._topic_name = topic_name
        self._data_type = data_type
        self._topic = None
        self._writer = None
        self._listener = None
        self._matched_readers = 0
        self._condition = threading.Condition()

    def start(self) -> None:
        if self._writer is not None:
            return
        from cyclonedds.core import Listener
        from cyclonedds.pub import DataWriter
        from cyclonedds.topic import Topic

        participant = _participant()
        self._topic = Topic(participant, self._topic_name, self._data_type)
        self._listener = Listener(on_publication_matched=self._on_publication_matched)
        self._writer = DataWriter(participant, self._topic, listener=self._listener)

    def write(self, sample: Any, timeout: float | None = None) -> bool:
        if self._writer is None:
            raise RuntimeError("DDS writer is not started")

        if timeout is not None:
            deadline = time.monotonic() + max(timeout, 0.0)
            with self._condition:
                while self._matched_readers == 0:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._condition.wait(timeout=remaining)
        try:
            self._writer.write(sample)
        except Exception:  # noqa: BLE001
            logger.exception("DDS write failed on topic %s", self._topic_name)
            return False
        return True

    def close(self) -> None:
        with self._condition:
            self._writer = None
            self._topic = None
            self._listener = None
            self._matched_readers = 0
            self._condition.notify_all()

    def _on_publication_matched(self, _writer: Any, status: Any) -> None:
        with self._condition:
            self._matched_readers = int(status.current_count)
            self._condition.notify_all()


class _DdsReader:
    def __init__(
        self,
        topic_name: str,
        data_type: Any,
        callback: Callable[[Any], None],
        queue_len: int,
    ) -> None:
        if queue_len <= 0:
            raise ValueError("DDS reader queue_len must be greater than zero")
        self._topic_name = topic_name
        self._data_type = data_type
        self._callback = callback
        self._capacity = queue_len
        self._topic = None
        self._reader = None
        self._listener = None
        self._queue: deque[Any] = deque()
        self._condition = threading.Condition()
        self._stop = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._reader is not None:
            return
        from cyclonedds.core import Listener
        from cyclonedds.sub import DataReader
        from cyclonedds.topic import Topic

        participant = _participant()
        self._stop = False
        self._thread = threading.Thread(
            target=self._dispatch,
            name="speech-dds-reader",
            daemon=True,
        )
        self._thread.start()
        self._topic = Topic(participant, self._topic_name, self._data_type)
        self._listener = Listener(on_data_available=self._on_data_available)
        self._reader = DataReader(participant, self._topic, listener=self._listener)

    def close(self) -> None:
        self._reader = None
        self._topic = None
        self._listener = None
        with self._condition:
            self._stop = True
            self._queue.clear()
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _on_data_available(self, reader: Any) -> None:
        from cyclonedds.internal import InvalidSample

        try:
            samples: list[Any] = []
            while True:
                batch = reader.take(self._capacity)
                if not batch:
                    break
                samples.extend(batch)
                if len(batch) < self._capacity:
                    break
        except Exception:  # noqa: BLE001
            logger.exception("DDS read failed on topic %s", self._topic_name)
            return
        if not samples:
            return
        dropped = 0
        with self._condition:
            if self._stop:
                return
            for sample in samples:
                if isinstance(sample, InvalidSample):
                    continue
                if len(self._queue) >= self._capacity:
                    dropped += 1
                    continue
                self._queue.append(sample)
            self._condition.notify_all()
        if dropped:
            logger.warning(
                "DDS reader queue is full; dropped %d sample(s) from %s",
                dropped,
                self._topic_name,
            )

    def _dispatch(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._stop:
                    self._condition.wait()
                if self._stop:
                    return
                sample = self._queue.popleft()
            try:
                self._callback(sample)
            except Exception:  # noqa: BLE001
                logger.exception("DDS subscriber callback failed on topic %s", self._topic_name)


class EventWriter(Protocol):
    def start(self) -> None: ...

    def write(self, event: SpeechEvent, timeout: float) -> bool: ...

    def close(self) -> None: ...


class DdsEventWriter:
    def __init__(self, topic: str = "rt/g1/hri/speech/final") -> None:
        self._topic = topic
        self._writer = _DdsWriter(topic, SpeechEventMessage)

    def start(self) -> None:
        self._writer.start()
        logger.info("DDS SpeechEvent publisher: %s", self._topic)

    def write(self, event: SpeechEvent, timeout: float) -> bool:
        return self._writer.write(event_to_message(event), timeout)

    def close(self) -> None:
        self._writer.close()


@dataclass
class _Pending:
    event: SpeechEvent
    expires_monotonic: float


class RetryingEventSink(EventSink):
    """Bounded outbox that retries the same event_id across short DDS outages."""

    def __init__(
        self,
        writer: EventWriter,
        *,
        capacity: int = 128,
        write_timeout_seconds: float = 0.5,
        retry_interval_seconds: float = 0.2,
        delivery_ttl_seconds: float = 30.0,
    ) -> None:
        self._writer = writer
        self._capacity = capacity
        self._write_timeout = write_timeout_seconds
        self._retry_interval = retry_interval_seconds
        self._delivery_ttl = delivery_ttl_seconds
        self._pending: deque[_Pending] = deque()
        self._condition = threading.Condition()
        self._stop = False
        self._thread: threading.Thread | None = None
        self.delivered = 0
        self.expired = 0
        self.dropped = 0
        self.retries = 0

    @property
    def queue_size(self) -> int:
        with self._condition:
            return len(self._pending)

    @property
    def capacity(self) -> int:
        return self._capacity

    def start(self) -> None:
        if self._thread is not None:
            return
        self._writer.start()
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="speech-dds-outbox", daemon=True)
        self._thread.start()

    def publish(self, event: SpeechEvent) -> bool:
        with self._condition:
            if len(self._pending) >= self._capacity:
                self.dropped += 1
                return False
            self._pending.append(
                _Pending(event=event, expires_monotonic=time.monotonic() + self._delivery_ttl)
            )
            self._condition.notify()
            return True

    def close(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=self._write_timeout + 2.0)
            self._thread = None
        self._writer.close()

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._stop:
                    self._condition.wait(timeout=0.5)
                if self._stop:
                    return
                pending = self._pending[0]
            if time.monotonic() >= pending.expires_monotonic:
                with self._condition:
                    if self._pending and self._pending[0] is pending:
                        self._pending.popleft()
                        self.expired += 1
                logger.error("SpeechEvent delivery expired: %s", pending.event.event_id)
                continue
            try:
                delivered = self._writer.write(pending.event, self._write_timeout)
            except Exception:  # noqa: BLE001
                logger.exception("DDS write failed; retrying event_id=%s", pending.event.event_id)
                delivered = False
            if delivered:
                with self._condition:
                    if self._pending and self._pending[0] is pending:
                        self._pending.popleft()
                        self.delivered += 1
                continue
            self.retries += 1
            with self._condition:
                if self._stop:
                    return
                self._condition.wait(timeout=self._retry_interval)


class EventDeduplicator:
    def __init__(self, *, capacity: int = 4096, ttl_seconds: float = 3600.0) -> None:
        self._capacity = capacity
        self._ttl = ttl_seconds
        self._seen: OrderedDict[str, float] = OrderedDict()
        self._lock = threading.Lock()

    def accept(self, event_id: str, *, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        with self._lock:
            while self._seen:
                _, timestamp = next(iter(self._seen.items()))
                if current - timestamp <= self._ttl:
                    break
                self._seen.popitem(last=False)
            if event_id in self._seen:
                self._seen.move_to_end(event_id)
                return False
            self._seen[event_id] = current
            while len(self._seen) > self._capacity:
                self._seen.popitem(last=False)
            return True


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


class DdsPlaybackSubscriber:
    def __init__(
        self,
        gate: PlaybackGate,
        *,
        topic: str = "rt/g1/hri/playback/state",
    ) -> None:
        self._gate = gate
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
        self._gate.set_active(bool(message.active))
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
        from .tts import TtsTextChunk

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
            )
        )


class DdsTtsPublisher:
    """Agent-side incremental text publisher for the robot mouth."""

    def __init__(self, *, topic: str = "rt/g1/hri/tts/request", source: str = "agent") -> None:
        self._source = source
        self._publisher = _DdsWriter(topic, TtsTextChunkMessage)

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
    ) -> bool:
        return self._publisher.write(
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

    def close(self) -> None:
        self._publisher.close()


class DdsPlaybackPublisher:
    """Agent-side helper for wrapping TTS/playback with set_active(True/False)."""

    def __init__(self, *, topic: str = "rt/g1/hri/playback/state", source: str = "agent") -> None:
        self._topic = topic
        self._source = source
        self._publisher = _DdsWriter(topic, PlaybackStateMessage)

    def start(self) -> None:
        self._publisher.start()

    def set_active(self, active: bool, *, request_id: str, timeout: float = 0.5) -> bool:
        message = PlaybackStateMessage(
            request_id=request_id,
            active=active,
            created_unix_ns=time.time_ns(),
            source=self._source,
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

    def __init__(self, config: DdsConfig, gate: PlaybackGate) -> None:
        initialize_dds(config.domain_id, config.network_interface)
        self._playback = DdsPlaybackSubscriber(gate, topic=config.playback_topic)
        self._config = config
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
        self._playback.start()
        if self._tts is not None:
            self._tts.start()
        if self._tts_playback is not None:
            self._tts_playback.start()

    def stop(self) -> None:
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

    def publish_playback_state(self, active: bool, request_id: str) -> None:
        if self._tts_playback is None:
            return
        if not self._tts_playback.set_active(
            active,
            request_id=request_id,
            timeout=0.25,
        ):
            logger.warning(
                "DDS playback state was not delivered: active=%s request_id=%s",
                active,
                request_id,
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
