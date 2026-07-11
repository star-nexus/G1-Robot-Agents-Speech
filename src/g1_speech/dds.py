"""Unitree SDK2 DDS adapters, retry outbox, and Agent-side deduplication."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Callable, Protocol

from .contracts import EventSink, SpeechEvent
from .dds_types import DDS_IDL_AVAILABLE, PlaybackStateMessage, SpeechEventMessage
from .gate import PlaybackGate

logger = logging.getLogger(__name__)


def initialize_unitree_dds(domain_id: int = 0, network_interface: str | None = None) -> None:
    if not DDS_IDL_AVAILABLE:
        raise RuntimeError("缺少 cyclonedds；请先安装 unitree_sdk2_python 及其 CycloneDDS 依赖")
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    ChannelFactoryInitialize(domain_id, network_interface)


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


class EventWriter(Protocol):
    def start(self) -> None: ...

    def write(self, event: SpeechEvent, timeout: float) -> bool: ...

    def close(self) -> None: ...


class UnitreeDdsEventWriter:
    def __init__(self, topic: str = "rt/g1/hri/speech/final") -> None:
        self._topic = topic
        self._publisher = None

    def start(self) -> None:
        if self._publisher is not None:
            return
        from unitree_sdk2py.core.channel import ChannelPublisher

        self._publisher = ChannelPublisher(self._topic, SpeechEventMessage)
        self._publisher.Init()
        logger.info("DDS SpeechEvent publisher: %s", self._topic)

    def write(self, event: SpeechEvent, timeout: float) -> bool:
        if self._publisher is None:
            raise RuntimeError("DDS publisher 尚未启动")
        return bool(self._publisher.Write(event_to_message(event), timeout))

    def close(self) -> None:
        if self._publisher is not None:
            self._publisher.Close()
            self._publisher = None


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
                logger.error("SpeechEvent 过期未送达: %s", pending.event.event_id)
                continue
            try:
                delivered = self._writer.write(pending.event, self._write_timeout)
            except Exception:  # noqa: BLE001
                logger.exception("DDS 写入异常，将重试 event_id=%s", pending.event.event_id)
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
        self._subscriber = None
        self.duplicates = 0

    def start(self) -> None:
        from unitree_sdk2py.core.channel import ChannelSubscriber

        self._subscriber = ChannelSubscriber(self._topic, SpeechEventMessage)
        self._subscriber.Init(self._on_message, self._queue_len)
        logger.info("DDS SpeechEvent subscriber: %s", self._topic)

    def close(self) -> None:
        if self._subscriber is not None:
            self._subscriber.Close()
            self._subscriber = None

    def _on_message(self, message: SpeechEventMessage) -> None:
        event = message_to_event(message)
        if not self._dedupe.accept(event.event_id):
            self.duplicates += 1
            logger.info("忽略重复 SpeechEvent: %s", event.event_id)
            return
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
        self._subscriber = None

    def start(self) -> None:
        from unitree_sdk2py.core.channel import ChannelSubscriber

        self._subscriber = ChannelSubscriber(self._topic, PlaybackStateMessage)
        self._subscriber.Init(self._on_message, 8)
        logger.info("DDS playback gate subscriber: %s", self._topic)

    def close(self) -> None:
        if self._subscriber is not None:
            self._subscriber.Close()
            self._subscriber = None

    def _on_message(self, message: PlaybackStateMessage) -> None:
        self._gate.set_active(bool(message.active))
        logger.info("播放门控 active=%s request_id=%s", message.active, message.request_id)


class DdsPlaybackPublisher:
    """Agent-side helper: wrap G1 TTS/playback with set_active(True/False)."""

    def __init__(self, *, topic: str = "rt/g1/hri/playback/state", source: str = "agent") -> None:
        self._topic = topic
        self._source = source
        self._publisher = None

    def start(self) -> None:
        from unitree_sdk2py.core.channel import ChannelPublisher

        self._publisher = ChannelPublisher(self._topic, PlaybackStateMessage)
        self._publisher.Init()

    def set_active(self, active: bool, *, request_id: str, timeout: float = 0.5) -> bool:
        if self._publisher is None:
            raise RuntimeError("Playback publisher 尚未启动")
        message = PlaybackStateMessage(
            request_id=request_id,
            active=active,
            created_unix_ns=time.time_ns(),
            source=self._source,
        )
        return bool(self._publisher.Write(message, timeout))

    def set_active_reliably(
        self,
        active: bool,
        *,
        request_id: str,
        delivery_timeout: float = 2.0,
        write_timeout: float = 0.25,
        retry_interval: float = 0.1,
    ) -> None:
        """Do not begin G1 playback until the Orin gate subscriber is matched."""
        deadline = time.monotonic() + delivery_timeout
        while time.monotonic() < deadline:
            if self.set_active(active, request_id=request_id, timeout=write_timeout):
                return
            time.sleep(retry_interval)
        raise TimeoutError(
            f"播放门控未在 {delivery_timeout:.1f}s 内送达: active={active} request_id={request_id}"
        )

    def close(self) -> None:
        if self._publisher is not None:
            self._publisher.Close()
            self._publisher = None
