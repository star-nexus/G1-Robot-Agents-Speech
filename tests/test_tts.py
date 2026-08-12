from __future__ import annotations

import json
import threading

import pytest

from g1_speech.config import ServiceConfig, TtsConfig
from g1_speech.tts import (
    SentenceAssembler,
    TtsAudioChunk,
    TtsController,
    TtsSynthesisRequest,
    TtsTextChunk,
    VllmOmniStreamingTts,
)


def test_sentence_assembler_handles_incremental_text_style_and_final_boundary():
    assembler = SentenceAssembler(TtsConfig())

    first = assembler.feed(
        TtsTextChunk("answer-1", 0, "[sad]It's in the top drawer,")
    )
    second = assembler.feed(
        TtsTextChunk("answer-1", 1, " but it is empty!", is_final=True)
    )

    assert [request.text for request in first + second] == [
        "It's in the top drawer,",
        "but it is empty!",
    ]
    assert not first[0].final_sentence
    assert second[0].final_sentence
    assert "sad" in first[0].instructions.lower()


def test_sentence_assembler_deduplicates_retried_dds_sequence():
    assembler = SentenceAssembler(TtsConfig())
    chunk = TtsTextChunk("answer-2", 4, "Hello")

    assert assembler.feed(chunk) == []
    assert assembler.feed(chunk) == []
    requests = assembler.feed(TtsTextChunk("answer-2", 5, "!", is_final=True))

    assert [request.text for request in requests] == ["Hello!"]


class FakeSocket:
    def __init__(self):
        self.sent = []
        self.messages = [
            json.dumps(
                {
                    "type": "audio.start",
                    "sample_rate": 24000,
                    "sentence_text": "Hello",
                }
            ),
            b"\x00\x00" * 240,
            json.dumps({"type": "audio.done", "error": False}),
            json.dumps({"type": "session.done"}),
        ]

    def settimeout(self, _timeout):
        pass

    def send(self, message):
        self.sent.append(json.loads(message))

    def recv(self):
        return self.messages.pop(0)

    def close(self):
        pass


class FakeWebsocket:
    def __init__(self):
        self.socket = FakeSocket()
        self.connections = 0

    def create_connection(self, *_args, **_kwargs):
        self.connections += 1
        return self.socket


def test_vllm_omni_client_uses_incremental_pcm_websocket_protocol():
    websocket = FakeWebsocket()
    engine = VllmOmniStreamingTts(TtsConfig(), websocket_module=websocket)
    request = TtsSynthesisRequest(
        request_id="answer-3",
        text="Hello",
        language="English",
        voice="Ryan",
        instructions="Calm",
        final_sentence=True,
    )

    audio = list(engine.stream(request, threading.Event()))

    assert len(audio) == 1
    assert audio[0].sample_rate == 24000
    assert websocket.socket.sent[0]["type"] == "session.config"
    assert websocket.socket.sent[0]["stream_audio"] is True
    assert websocket.socket.sent[1] == {"type": "input.text", "text": "Hello"}
    assert websocket.socket.sent[2] == {"type": "input.done"}

    websocket.socket.messages.extend(
        [
            json.dumps({"type": "audio.start", "sample_rate": 24000}),
            b"\x00\x00" * 120,
            json.dumps({"type": "audio.done", "error": False}),
            json.dumps({"type": "session.done"}),
        ]
    )
    assert len(list(engine.stream(request, threading.Event()))) == 1
    assert websocket.connections == 1


def test_custom_voice_rejects_reference_audio_to_prevent_false_cloning():
    settings = TtsConfig(
        task_type="CustomVoice",
        reference_audio="audio.wav",
        reference_text="reference",
    )
    with pytest.raises(ValueError, match="only valid with a Qwen3-TTS Base"):
        ServiceConfig(tts=settings).validate()


class BlockingEngine:
    def __init__(self):
        self.cancelled = threading.Event()
        self.closed = False

    def stream(self, _request, cancel):
        yield TtsAudioChunk(b"\0\0" * 240, 24000)
        cancel.wait(2.0)

    def cancel(self):
        self.cancelled.set()

    def close(self):
        self.closed = True


class InterruptiblePlayer:
    def __init__(self):
        self.written = threading.Event()
        self.aborts = 0

    def start(self):
        pass

    def begin(self, sample_rate):
        assert sample_rate == 24000

    def write(self, pcm):
        assert pcm
        self.written.set()
        return True

    def finish(self):
        pass

    def abort(self):
        self.aborts += 1

    def close(self):
        pass

    def metrics(self):
        return {}


def test_vad_barge_in_cancels_generation_aborts_playback_and_publishes_one_edge():
    engine = BlockingEngine()
    player = InterruptiblePlayer()
    states = []
    controller = TtsController(
        TtsConfig(),
        engine=engine,
        player=player,
        playback_state=lambda active, request_id: states.append((active, request_id)),
    )
    controller.start()
    controller.accept(TtsTextChunk("answer-4", 0, "Hello!", is_final=True))
    assert player.written.wait(1.0)

    controller.interrupt(reason="vad")
    controller.close()

    assert engine.cancelled.is_set()
    assert player.aborts >= 1
    assert states == [(True, "answer-4"), (False, "answer-4")]
