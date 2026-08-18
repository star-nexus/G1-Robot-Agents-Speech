"""Process-wide Cyclone DDS lifecycle and low-level I/O primitives."""

from __future__ import annotations

import html
import logging
import threading
import time
from collections import deque
from typing import Any, Callable

from .types import DDS_IDL_AVAILABLE

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

