from __future__ import annotations

import time

from g1_speech.contracts import SpeechEvent
from g1_speech.dds import (
    DdsPlaybackPublisher,
    DdsSpeechSubscriber,
    EventDeduplicator,
    RetryingEventSink,
    event_to_message,
    message_to_event,
)
from g1_speech.dds_types import DDS_IDL_AVAILABLE, SpeechEventMessage


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


def test_agent_subscriber_never_forwards_duplicate_to_callback():
    received = []
    subscriber = DdsSpeechSubscriber(received.append)
    message = event_to_message(make_event())
    subscriber._on_message(message)
    subscriber._on_message(message)

    assert [event.event_id for event in received] == ["event-1"]
    assert subscriber.duplicates == 1


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
