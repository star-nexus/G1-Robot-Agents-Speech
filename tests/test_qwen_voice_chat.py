from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


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
