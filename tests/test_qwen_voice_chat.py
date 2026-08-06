from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "qwen_voice_chat", ROOT / "examples/qwen_voice_chat.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeResponse:
    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        return iter(self._lines)


def test_conversation_keeps_only_configured_number_of_turns():
    conversation = MODULE.Conversation("system", history_turns=2)
    for index in range(3):
        conversation.commit(f"user-{index}", f"assistant-{index}")

    assert conversation.messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user-1"},
        {"role": "assistant", "content": "assistant-1"},
        {"role": "user", "content": "user-2"},
        {"role": "assistant", "content": "assistant-2"},
    ]


def test_normalize_phrase_removes_asr_punctuation():
    assert MODULE.normalize_phrase("  清空对话。 ") == "清空对话"


def test_transport_can_be_selected_explicitly():
    args = MODULE.build_parser().parse_args(["--transport", "ros2"])
    assert args.transport == "ros2"


def test_ollama_stream_accumulates_content_and_metrics(monkeypatch):
    chunks = [
        {"message": {"content": "你"}, "done": False},
        {
            "message": {"content": "好"},
            "done": True,
            "prompt_eval_count": 12,
            "eval_count": 2,
            "eval_duration": 500_000_000,
        },
    ]
    monkeypatch.setattr(
        MODULE,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(
            [(json.dumps(chunk) + "\n").encode() for chunk in chunks]
        ),
    )
    client = MODULE.OllamaChatClient(
        base_url="http://localhost:11434",
        model="qwen3",
        timeout=1.0,
        keep_alive="5m",
        num_ctx=4096,
        num_predict=32,
        temperature=0.6,
        think=False,
    )
    emitted = []
    result = client.chat([{"role": "user", "content": "你好"}], emitted.append)

    assert result.content == "你好"
    assert emitted == ["你", "好"]
    assert result.prompt_tokens == 12
    assert result.generated_tokens == 2
    assert result.tokens_per_second == 4.0


def test_llamacpp_sse_stream_accumulates_content_and_metrics(monkeypatch):
    chunks = [
        {"choices": [{"delta": {"content": "你"}}]},
        {"choices": [{"delta": {"content": "好"}}]},
        {
            "choices": [],
            "usage": {"prompt_tokens": 12, "completion_tokens": 2},
            "timings": {"predicted_per_second": 8.5},
        },
    ]
    lines = [f"data: {json.dumps(chunk)}\n".encode() for chunk in chunks]
    lines.append(b"data: [DONE]\n")
    monkeypatch.setattr(
        MODULE,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(lines),
    )
    client = MODULE.LlamaCppChatClient(
        base_url="http://localhost:8080",
        model="qwen3",
        timeout=1.0,
        num_predict=32,
        temperature=0.6,
    )
    emitted = []
    result = client.chat([{"role": "user", "content": "你好"}], emitted.append)

    assert result.content == "你好"
    assert emitted == ["你", "好"]
    assert result.prompt_tokens == 12
    assert result.generated_tokens == 2
    assert result.tokens_per_second == 8.5


def test_llamacpp_is_the_default_provider():
    assert MODULE.build_parser().parse_args([]).provider == "llama"


def test_mjpeg_parser_handles_split_and_consecutive_frames():
    parser = MODULE.MjpegStreamParser()
    first = b"\xff\xd8first\xff\xd9"
    second = b"\xff\xd8second\xff\xd9"

    assert parser.feed(b"noise" + first[:5]) == []
    assert parser.feed(first[5:] + second) == [first, second]


def test_conversation_attaches_image_only_to_current_request():
    conversation = MODULE.Conversation("system", history_turns=2)
    request = conversation.request("你看到了什么？", "data:image/jpeg;base64,abc")

    assert request[-1] == {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,abc"},
            },
            {
                "type": "text",
                "text": (
                    "以上是机器人摄像头刚刚捕获的当前画面。"
                    "请结合画面回答用户的语音：你看到了什么？"
                ),
            },
        ],
    }

    conversation.commit("你看到了什么？", "一张测试图。")
    assert conversation.messages[-2] == {
        "role": "user",
        "content": "你看到了什么？",
    }


def test_camera_command_copies_mjpeg_without_reencoding():
    camera = MODULE.LatestFrameCamera(
        device="/dev/video0",
        width=1280,
        height=720,
        fps=5,
        ffmpeg="ffmpeg",
    )

    command = camera._command()
    assert command[command.index("-input_format") + 1] == "mjpeg"
    assert command[command.index("-video_size") + 1] == "1280x720"
    assert command[command.index("-c:v") + 1] == "copy"


def test_camera_rejects_ollama_provider():
    args = MODULE.build_parser().parse_args(
        ["--provider", "ollama", "--camera", "/dev/video0"]
    )

    with pytest.raises(ValueError, match="requires --provider llama"):
        MODULE.validate_args(args)
