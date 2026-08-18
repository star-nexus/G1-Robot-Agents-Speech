"""Bounded retry and deduplication policies independent of Agent composition."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Protocol

from ...core.events import SpeechEvent
from ..contracts import EventSink
from .codec import event_to_message
from .runtime import _DdsWriter
from .types import SpeechEventMessage

logger = logging.getLogger(__name__)

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

