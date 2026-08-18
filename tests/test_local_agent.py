from __future__ import annotations

import json
import time

import pytest

from g1_speech.config import ServiceConfig
from g1_speech.contracts import SpeechEvent
from g1_speech.local_agent import (
    LlamaChatClient,
    LocalAgentSettings,
    LocalVoiceAgent,
    load_role_prompt,
)


class FakeResponse:
    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def __iter__(self):
        return iter(self._lines)


def test_llama_client_reads_only_spoken_sse_content():
    payloads = [
        {"choices": [{"delta": {"reasoning_content": "先思考"}}]},
        {"choices": [{"delta": {"content": "你好"}}]},
        {"choices": [{"delta": {"content": "！"}}]},
    ]
    lines = [f"data: {json.dumps(item)}\n".encode() for item in payloads]
    lines.append(b"data: [DONE]\n")
    captured = {}

    def opener(http_request, timeout):
        captured["body"] = json.loads(http_request.data)
        captured["timeout"] = timeout
        return FakeResponse(lines)

    settings = LocalAgentSettings(request_timeout_seconds=7)
    client = LlamaChatClient(settings, opener=opener)

    assert list(client.stream([{"role": "user", "content": "你好"}])) == ["你好", "！"]
    assert captured["body"]["stream"] is True
    assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["timeout"] == 7


def test_llama_client_can_enable_budgeted_thinking_without_speaking_it():
    payloads = [
        {"choices": [{"delta": {"reasoning_content": "短く考える"}}]},
        {"choices": [{"delta": {"content": "答えです。"}}]},
    ]
    lines = [f"data: {json.dumps(item)}\n".encode() for item in payloads]
    lines.append(b"data: [DONE]\n")
    captured = {}

    def opener(http_request, timeout):
        captured["body"] = json.loads(http_request.data)
        return FakeResponse(lines)

    client = LlamaChatClient(
        LocalAgentSettings(
            enable_thinking=True,
            reasoning_budget_tokens=12,
        ),
        opener=opener,
    )

    assert list(client.stream([{"role": "user", "content": "質問"}])) == [
        "答えです。"
    ]
    assert captured["body"]["chat_template_kwargs"] == {
        "enable_thinking": True
    }
    assert captured["body"]["reasoning_budget_tokens"] == 12
    assert captured["body"]["reasoning_format"] == "deepseek"
    assert "只输出最终回答" in captured["body"]["reasoning_budget_message"]


def test_budgeted_thinking_buffers_and_strips_reasoning_markup():
    payloads = [
        {"choices": [{"delta": {"reasoning_content": "hidden"}}]},
        {"choices": [{"delta": {"content": "</think>\n"}}]},
        {"choices": [{"delta": {"content": "私はティファ。"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    lines = [f"data: {json.dumps(item)}\n".encode() for item in payloads]
    lines.append(b"data: [DONE]\n")

    client = LlamaChatClient(
        LocalAgentSettings(enable_thinking=True, reasoning_budget_tokens=12),
        opener=lambda *_args, **_kwargs: FakeResponse(lines),
    )

    assert list(client.stream([{"role": "user", "content": "質問"}])) == [
        "私はティファ。"
    ]


def test_budgeted_thinking_rejects_length_truncated_content_before_tts():
    payloads = [
        {"choices": [{"delta": {"reasoning_content": "hidden"}}]},
        {"choices": [{"delta": {"content": "我还在分析"}}]},
        {"choices": [{"delta": {}, "finish_reason": "length"}]},
    ]
    lines = [f"data: {json.dumps(item)}\n".encode() for item in payloads]
    lines.append(b"data: [DONE]\n")
    client = LlamaChatClient(
        LocalAgentSettings(enable_thinking=True, reasoning_budget_tokens=12),
        opener=lambda *_args, **_kwargs: FakeResponse(lines),
    )

    with pytest.raises(RuntimeError, match="exhausted max_tokens"):
        list(client.stream([{"role": "user", "content": "質問"}]))


def test_role_prompt_is_loaded_from_a_reusable_utf8_file(tmp_path):
    role = tmp_path / "角色.md"
    role.write_text("\n你是雪宝。\n", encoding="utf-8")

    assert load_role_prompt(role) == "你是雪宝。"


def test_empty_role_prompt_is_rejected(tmp_path):
    role = tmp_path / "empty.md"
    role.write_text(" \n", encoding="utf-8")

    with pytest.raises(ValueError, match="role file is empty"):
        load_role_prompt(role)


class FakeClient:
    def stream(self, messages):
        assert messages[-1] == {"role": "user", "content": "你是谁？"}
        yield "我是"
        yield "本地机器人。"


class FakePublisher:
    def __init__(self):
        self.calls = []

    def start(self):
        pass

    def publish(self, **kwargs):
        self.calls.append(kwargs)
        return True

    def close(self):
        pass


class FakeSubscriber:
    def __init__(self):
        self.closed = False

    def start(self):
        pass

    def close(self):
        self.closed = True


def test_voice_agent_streams_with_the_last_text_fragment_marked_final():
    publisher = FakePublisher()
    agent = LocalVoiceAgent(
        ServiceConfig(),
        LocalAgentSettings(
            history_turns=1,
            tts_voice="Ono_Anna",
            tts_language="Japanese",
        ),
        client=FakeClient(),
        publisher=publisher,
        subscriber=FakeSubscriber(),
    )
    event = SpeechEvent(
        event_id="speech-1",
        session_id="session-1",
        sequence=1,
        created_unix_ns=1,
        source="mic",
        text="你是谁？",
        language="zh",
        audio_duration_ms=500,
        inference_ms=20,
    )

    agent._answer(event)

    assert [call["text"] for call in publisher.calls] == ["我是", "本地机器人。"]
    assert [call["sequence"] for call in publisher.calls] == [0, 1]
    assert publisher.calls[-1]["is_final"] is True
    assert all(call["voice"] == "Ono_Anna" for call in publisher.calls)
    assert all(call["language"] == "Japanese" for call in publisher.calls)
    assert agent._turns == [
        {"role": "user", "content": "你是谁？"},
        {"role": "assistant", "content": "我是本地机器人。"},
    ]


def test_nonfinal_and_empty_speech_events_are_ignored():
    agent = LocalVoiceAgent(
        ServiceConfig(),
        LocalAgentSettings(),
        client=FakeClient(),
        publisher=FakePublisher(),
        subscriber=FakeSubscriber(),
    )
    base = dict(
        event_id="speech-1",
        session_id="session-1",
        sequence=1,
        created_unix_ns=1,
        source="mic",
        language="zh",
        audio_duration_ms=500,
        inference_ms=20,
    )

    agent._on_speech(SpeechEvent(text="你好", is_final=False, **base))
    agent._on_speech(SpeechEvent(text="  ", is_final=True, **base))

    assert agent._queue.empty()


def test_stale_speech_is_not_replayed_when_agent_joins_dds_late():
    agent = LocalVoiceAgent(
        ServiceConfig(),
        LocalAgentSettings(max_speech_age_seconds=5),
        client=FakeClient(),
        publisher=FakePublisher(),
        subscriber=FakeSubscriber(),
    )
    event = SpeechEvent(
        event_id="old-speech",
        session_id="session-1",
        sequence=1,
        created_unix_ns=time.time_ns() - 10_000_000_000,
        source="mic",
        text="十秒前的问题",
        language="zh",
        audio_duration_ms=500,
        inference_ms=20,
    )

    agent._on_speech(event)

    assert agent._queue.empty()


def test_partial_agent_failure_interrupts_tts_instead_of_leaving_gate_active():
    class FailingClient:
        def stream(self, _messages):
            yield "已经"
            yield "开始"
            raise RuntimeError("connection lost")

    publisher = FakePublisher()
    agent = LocalVoiceAgent(
        ServiceConfig(),
        LocalAgentSettings(),
        client=FailingClient(),
        publisher=publisher,
        subscriber=FakeSubscriber(),
    )
    event = SpeechEvent(
        event_id="speech-fail",
        session_id="session-1",
        sequence=1,
        created_unix_ns=time.time_ns(),
        source="mic",
        text="测试异常",
        language="zh",
        audio_duration_ms=500,
        inference_ms=20,
    )

    with pytest.raises(RuntimeError, match="connection lost"):
        agent._answer(event)

    assert publisher.calls[0]["text"] == "已经"
    assert publisher.calls[-1]["interrupt"] is True
    assert publisher.calls[-1]["is_final"] is True
