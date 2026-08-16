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

    assert first == []
    assert [request.text for request in second] == [
        "It's in the top drawer, but it is empty!",
    ]
    assert second[0].final_sentence
    assert "sad" in second[0].instructions.lower()


def test_final_robot_utterance_keeps_commas_in_one_prosodic_request():
    assembler = SentenceAssembler(TtsConfig())

    requests = assembler.feed(
        TtsTextChunk(
            "answer-robot",
            0,
            "我是Olaf,冰雪奇缘里的雪宝机器人,欢迎来到迪士尼公园.",
            is_final=True,
        )
    )

    assert [request.text for request in requests] == [
        "我是Olaf,冰雪奇缘里的雪宝机器人,欢迎来到迪士尼公园."
    ]
    assert requests[0].final_sentence


def test_streaming_robot_utterance_waits_for_a_complete_soft_clause():
    assembler = SentenceAssembler(TtsConfig())

    assert assembler.feed(TtsTextChunk("answer-stream", 0, "我是Olaf,")) == []
    middle = assembler.feed(
        TtsTextChunk("answer-stream", 1, "冰雪奇缘里的雪宝机器人,")
    )
    final = assembler.feed(
        TtsTextChunk("answer-stream", 2, "欢迎来到迪士尼公园", is_final=True)
    )

    assert [request.text for request in middle] == [
        "我是Olaf,冰雪奇缘里的雪宝机器人,"
    ]
    assert [request.text for request in final] == ["欢迎来到迪士尼公园"]
    assert not middle[0].final_sentence
    assert final[0].final_sentence


def test_chunker_does_not_treat_decimal_point_as_sentence_end():
    assembler = SentenceAssembler(TtsConfig())

    requests = assembler.feed(
        TtsTextChunk("answer-decimal", 0, "电压是3.14伏，请继续。", is_final=True)
    )

    assert [request.text for request in requests] == ["电压是3.14伏，请继续。"]


def test_unpunctuated_stream_is_bounded_by_estimated_spoken_length():
    assembler = SentenceAssembler(TtsConfig())

    requests = assembler.feed(
        TtsTextChunk("answer-long", 0, "这是没有任何标点符号的连续中文文本需要及时开始合成避免等待太久")
    )

    assert len(requests) == 1
    assert 8 <= SentenceAssembler._speech_units(requests[0].text) <= 24


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


def test_robot_chunk_speech_unit_thresholds_must_be_ordered():
    settings = TtsConfig(
        min_chunk_speech_units=12,
        preferred_chunk_speech_units=8,
        max_chunk_speech_units=24,
    )
    with pytest.raises(ValueError, match="min <= preferred <= max"):
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

    def enqueue(self, pcm, sample_rate, *, cancel=None):
        assert sample_rate == 24000
        assert pcm
        assert cancel is not None
        self.written.set()
        return True

    def wait_until_idle(self, *, cancel=None, timeout=None):
        del cancel, timeout
        return True

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


class RecordingEngine:
    def __init__(self):
        self.requests = []

    def stream(self, request, _cancel):
        self.requests.append(request.text)
        yield TtsAudioChunk(b"\0\0" * 240, 24000)

    def cancel(self):
        pass

    def close(self):
        pass


class BufferedTimelinePlayer:
    def __init__(self):
        self.enqueued = []
        self.waiting_for_drain = threading.Event()
        self.release_drain = threading.Event()

    def start(self):
        pass

    def enqueue(self, pcm, sample_rate, *, cancel=None):
        assert cancel is not None
        self.enqueued.append((pcm, sample_rate))
        return True

    def wait_until_idle(self, *, cancel=None, timeout=None):
        del timeout
        self.waiting_for_drain.set()
        while not self.release_drain.wait(0.01):
            if cancel is not None and cancel.is_set():
                return False
        return True

    def abort(self):
        self.release_drain.set()

    def close(self):
        self.release_drain.set()

    def metrics(self):
        return {}


def test_controller_generates_all_sentences_before_waiting_for_playback_drain():
    engine = RecordingEngine()
    player = BufferedTimelinePlayer()
    controller = TtsController(
        TtsConfig(),
        engine=engine,
        player=player,
        playback_state=lambda _active, _request_id: None,
    )
    controller.start()
    controller.accept(TtsTextChunk("answer-5", 0, "One! Two! Three!", is_final=True))

    assert player.waiting_for_drain.wait(1.0)
    assert engine.requests == ["One!", "Two!", "Three!"]
    assert len(player.enqueued) == 3

    player.release_drain.set()
    controller.close()
