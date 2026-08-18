from __future__ import annotations

import time

from star_runtime.core.events import SpeechEvent
from star_runtime.transports.inprocess import InProcessTransport


def _event() -> SpeechEvent:
    return SpeechEvent(
        event_id="event-1",
        session_id="session-1",
        sequence=1,
        created_unix_ns=time.time_ns(),
        source="test",
        text="你好",
        language="zh",
        audio_duration_ms=100,
        inference_ms=5.0,
    )


def test_inprocess_transport_delivers_same_speech_object_without_serialization():
    playback = []
    speech = []
    transport = InProcessTransport(playback.append)
    transport.voice.set_handler(speech.append)
    transport.start()
    transport.sink.start()
    transport.voice.start()
    event = _event()

    result = transport.sink.publish(event)

    assert result.accepted and result.delivered
    assert speech == [event]
    assert speech[0] is event
    assert transport.metrics()["inprocess_speech_delivered"] == 1


def test_inprocess_transport_connects_agent_text_to_tts_and_playback_gate():
    playback = []
    chunks = []
    transport = InProcessTransport(playback.append)
    transport.register_tts_handler(chunks.append)
    transport.start()
    transport.voice.start()

    result = transport.voice.publish(
        request_id="request-1",
        sequence=0,
        text="你好",
        is_final=True,
    )
    transport.publish_playback_state(True, "request-1")

    assert result.accepted and result.delivered
    assert chunks[0].text == "你好"
    assert chunks[0].is_final is True
    assert playback == [True]
    assert transport.metrics()["inprocess_tts_delivered"] == 1


def test_inprocess_transport_rejects_unbound_or_stopped_routes():
    transport = InProcessTransport(lambda _active: None)
    transport.start()
    transport.sink.start()
    assert not transport.sink.publish(_event())

    transport.stop()
    transport.voice.start()
    result = transport.voice.publish(
        request_id="request-1",
        sequence=0,
        text="你好",
    )
    assert not result
