from __future__ import annotations

import time

from g1_speech.contracts import SpeechEvent
from g1_speech.dds import (
    DdsPlaybackPublisher,
    DdsSpeechSubscriber,
    DdsTtsPublisher,
    DdsTtsSubscriber,
    EventDeduplicator,
    RetryingEventSink,
    _DdsReader,
    event_to_message,
    message_to_event,
)
from g1_speech.dds_types import (
    DDS_IDL_AVAILABLE,
    SpeechEventMessage,
    TtsTextChunkMessage,
)


def make_event(event_id: str = "event-1") -> SpeechEvent:
    return SpeechEvent(
        event_id=event_id,
        session_id="session-1",
        sequence=1,
        created_unix_ns=123,
        source="mic",
        text="你好",
        language="zh",
        audio_duration_ms=500,
        inference_ms=12.5,
    )


class FakeWriter:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.event_ids = []
        self.started = False

    def start(self):
        self.started = True

    def write(self, event, timeout):
        self.event_ids.append(event.event_id)
        return next(self.outcomes, True)

    def close(self):
        self.started = False


def test_message_round_trip():
    event = make_event()
    assert message_to_event(event_to_message(event)) == event


def test_cyclonedds_idl_type_can_be_populated():
    if DDS_IDL_AVAILABLE:
        SpeechEventMessage.__idl__.populate()


def test_retry_reuses_same_event_id_until_reconnected():
    writer = FakeWriter([False, False, True])
    sink = RetryingEventSink(
        writer,
        capacity=4,
        write_timeout_seconds=0.01,
        retry_interval_seconds=0.01,
        delivery_ttl_seconds=2,
    )
    sink.start()
    assert sink.publish(make_event())
    deadline = time.monotonic() + 1
    while sink.delivered == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    sink.close()

    assert sink.delivered == 1
    assert sink.retries == 2
    assert writer.event_ids == ["event-1", "event-1", "event-1"]


def test_duplicate_event_is_accepted_only_once():
    dedupe = EventDeduplicator(capacity=10, ttl_seconds=60)
    assert dedupe.accept("same", now=1.0)
    assert not dedupe.accept("same", now=2.0)
    assert dedupe.accept("different", now=3.0)


def test_agent_subscriber_never_forwards_duplicate_to_callback(caplog):
    caplog.set_level("INFO", logger="star_runtime.transports.dds.voice")
    received = []
    subscriber = DdsSpeechSubscriber(received.append)
    message = event_to_message(make_event())
    subscriber._on_message(message)
    subscriber._on_message(message)

    assert [event.event_id for event in received] == ["event-1"]
    assert subscriber.duplicates == 1
    assert "DDS SpeechEvent received: event_id=event-1" in caplog.text
    assert "created_to_received=" in caplog.text


def test_outbox_is_bounded():
    writer = FakeWriter([False] * 100)
    sink = RetryingEventSink(writer, capacity=1, write_timeout_seconds=0.01)
    assert sink.publish(make_event("first"))
    assert not sink.publish(make_event("second"))
    assert sink.queue_size == 1
    assert sink.dropped == 1


def test_playback_does_not_start_until_gate_is_deliverable():
    class FakePublisher:
        def __init__(self):
            self.outcomes = iter([False, False, True])
            self.calls = 0

        def write(self, message, timeout):
            self.calls += 1
            return next(self.outcomes)

    gate = DdsPlaybackPublisher()
    gate._publisher = FakePublisher()
    gate.set_active_reliably(
        True,
        request_id="tts-1",
        delivery_timeout=1,
        write_timeout=0.01,
        retry_interval=0.01,
    )
    assert gate._publisher.calls == 3


def test_tts_dds_contract_preserves_incremental_request_fields():
    received = []
    subscriber = DdsTtsSubscriber(received.append, topic="tts")
    subscriber._on_message(
        TtsTextChunkMessage(
            request_id="answer-1",
            sequence=7,
            text="Hello,",
            is_final=False,
            interrupt=False,
            language="English",
            voice="Ryan",
            instructions="Calm",
            created_unix_ns=123,
            source="agent",
        )
    )

    chunk = received[0]
    assert (chunk.request_id, chunk.sequence, chunk.text) == ("answer-1", 7, "Hello,")
    assert (chunk.language, chunk.voice, chunk.instructions) == (
        "English",
        "Ryan",
        "Calm",
    )


def test_tts_dds_publisher_sets_idempotency_and_interrupt_fields():
    class CaptureWriter:
        def __init__(self):
            self.message = None

        def write(self, message, timeout):
            self.message = message
            return timeout == 0.1

    publisher = DdsTtsPublisher(topic="tts", source="brain")
    writer = CaptureWriter()
    publisher._publisher = writer

    assert publisher.publish(
        request_id="answer-2",
        sequence=9,
        text="",
        is_final=True,
        interrupt=True,
        timeout=0.1,
    )
    assert writer.message.request_id == "answer-2"
    assert writer.message.sequence == 9
    assert writer.message.is_final is True
    assert writer.message.interrupt is True
    assert writer.message.source == "brain"


def test_dds_listener_drains_coalesced_streaming_samples_in_one_notification():
    class BatchReader:
        def __init__(self):
            self.calls = 0

        def take(self, capacity):
            assert capacity == 4
            self.calls += 1
            return ["delta", "final"] if self.calls == 1 else []

    reader = _DdsReader("tts", str, lambda _sample: None, queue_len=4)
    source = BatchReader()

    reader._on_data_available(source)

    assert list(reader._queue) == ["delta", "final"]
    assert source.calls == 1
