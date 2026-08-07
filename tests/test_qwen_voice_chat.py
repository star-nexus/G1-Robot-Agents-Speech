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


class FakeJsonResponse(FakeResponse):
    def __init__(self, payload):
        super().__init__([])
        self._payload = json.dumps(payload).encode()

    def read(self):
        return self._payload


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


def test_conversation_rejects_internal_see_marker():
    conversation = MODULE.Conversation("system", history_turns=2)

    with pytest.raises(ValueError, match="must never enter conversation history"):
        conversation.commit("门关了吗？", "普通回答。[SEE]")

    assert conversation.messages == [{"role": "system", "content": "system"}]


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
        temperature=None,
    )
    emitted = []
    result = client.chat([{"role": "user", "content": "你好"}], emitted.append)

    assert result.content == "你好"
    assert emitted == ["你", "好"]
    assert result.prompt_tokens == 12
    assert result.generated_tokens == 2
    assert result.tokens_per_second == 8.5


def test_llamacpp_see_classifier_uses_dedicated_prompt_and_grammar(monkeypatch):
    captured = {}

    def fake_urlopen(request, **_kwargs):
        captured["body"] = json.loads(request.data)
        return FakeJsonResponse(
            {"choices": [{"message": {"content": "[SEE]"}}]}
        )

    monkeypatch.setattr(MODULE, "urlopen", fake_urlopen)
    client = MODULE.LlamaCppChatClient(
        base_url="http://localhost:8080",
        model="qwen3-vl",
        timeout=1.0,
        num_predict=32,
        temperature=None,
    )
    messages = [
        {"role": "system", "content": MODULE.SEE_CLASSIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": '{"CURRENT": "Is the door closed?"}'},
    ]

    route, reason = client.classify_see_marker(messages)

    assert route == "VISION"
    assert "[SEE]" in reason
    assert captured["body"]["messages"] == messages
    assert captured["body"]["stream"] is False
    assert captured["body"]["max_tokens"] == 8
    assert captured["body"]["temperature"] == 0
    assert captured["body"]["grammar"] == 'root ::= "[SEE]" | "[TEXT]"'


def test_see_classifier_session_is_independent_and_keeps_four_turns():
    requests = []

    class FakeClient:
        def classify_see_marker(self, messages):
            requests.append(messages)
            return ("VISION", "[SEE]") if len(requests) % 2 else ("TEXT", "[TEXT]")

    session = MODULE.SeeClassifierSession(FakeClient(), history_turns=4)
    main_history = [
        {"role": "assistant", "content": "main persona answer must stay isolated"}
    ]
    for index in range(6):
        session.classify(f"prompt-{index}", main_history, index % 2 == 1)

    assert len(session.messages) == 1 + 2 * 4
    assert len(requests[-1]) == 1 + 2 * 4 + 1
    flattened = json.dumps(requests[-1], ensure_ascii=False)
    assert "main persona answer" not in flattened
    assert "prompt-0" not in flattened
    assert "prompt-1" in flattened
    assert requests[-1][0] == {
        "role": "system",
        "content": MODULE.SEE_CLASSIFIER_SYSTEM_PROMPT,
    }
    assert all(
        message["content"] in {"[SEE]", "[TEXT]"}
        for message in session.messages
        if message["role"] == "assistant"
    )

    session.reset()
    assert session.messages == [
        {"role": "system", "content": MODULE.SEE_CLASSIFIER_SYSTEM_PROMPT}
    ]


def test_llamacpp_final_answer_discards_split_see_marker(monkeypatch):
    captured = {}
    chunks = [
        {"choices": [{"delta": {"content": "画面里有一个雪人"}}]},
        {"choices": [{"delta": {"content": "[SE"}}]},
        {"choices": [{"delta": {"content": "E]"}}]},
        {
            "choices": [],
            "usage": {"prompt_tokens": 600, "completion_tokens": 12},
            "timings": {"predicted_per_second": 12.0},
        },
    ]
    lines = [f"data: {json.dumps(chunk)}\n".encode() for chunk in chunks]
    lines.append(b"data: [DONE]\n")
    def fake_urlopen(request, **_kwargs):
        captured["body"] = json.loads(request.data)
        return FakeResponse(lines)

    monkeypatch.setattr(MODULE, "urlopen", fake_urlopen)
    client = MODULE.LlamaCppChatClient(
        base_url="http://localhost:8080",
        model="qwen3-vl",
        timeout=1.0,
        num_predict=32,
        temperature=None,
    )
    emitted = []

    result = client.chat(
        [{"role": "user", "content": "画面里有什么？"}],
        emitted.append,
        discard_see_markers=True,
        use_vision_sampling=True,
    )

    assert result.content == "画面里有一个雪人"
    assert emitted == ["画面里有一个雪人"]
    assert captured["body"]["temperature"] == 0.7
    assert captured["body"]["top_p"] == 0.8
    assert captured["body"]["top_k"] == 20
    assert captured["body"]["repeat_penalty"] == 1.0
    assert captured["body"]["presence_penalty"] == 1.5
    assert captured["body"]["frequency_penalty"] == 0.0
    assert captured["body"]["repeat_last_n"] == 64


def test_llamacpp_is_the_default_provider():
    assert MODULE.build_parser().parse_args([]).provider == "llama"


def test_snowball_role_sets_prompt_conversation_and_sampling_defaults():
    args = MODULE.build_parser().parse_args(["--role", "snowball"])
    role = MODULE.load_role_profile(args.role)
    MODULE.apply_role_defaults(args, role)
    text, vision = MODULE.resolve_sampling_configs(args, role)

    assert role.path == ROOT / "roles" / "snowball.role"
    assert args.history_turns == 4
    assert args.queue_size == 4
    assert args.num_predict == 48
    assert "《冰雪奇缘》中的雪宝" in args.system
    assert text.repeat_penalty == 1.15
    assert text.presence_penalty == 1.1
    assert text.frequency_penalty == 1.2
    assert text.repeat_last_n == 256
    assert vision.temperature == 0.7
    assert vision.repeat_penalty == 1.05


def test_cli_sampling_and_conversation_options_override_role():
    args = MODULE.build_parser().parse_args(
        [
            "--role",
            "snowball",
            "--history-turns",
            "2",
            "--num-predict",
            "32",
            "--repeat-penalty",
            "1.1",
            "--frequency-penalty",
            "0.4",
        ]
    )
    role = MODULE.load_role_profile(args.role)
    MODULE.apply_role_defaults(args, role)
    text, vision = MODULE.resolve_sampling_configs(args, role)

    assert args.history_turns == 2
    assert args.num_predict == 32
    assert text.repeat_penalty == vision.repeat_penalty == 1.1
    assert text.frequency_penalty == vision.frequency_penalty == 0.4


def test_role_rejects_unknown_request_fields(tmp_path):
    role_path = tmp_path / "unsafe.role"
    role_path.write_text(
        json.dumps(
            {
                "version": 1,
                "name": "unsafe",
                "sampling": {"text": {"messages": []}},
            }
        )
    )

    with pytest.raises(ValueError, match="unknown role.sampling.text field"):
        MODULE.load_role_profile(str(role_path))


def test_list_roles_does_not_connect_to_model(capsys):
    assert MODULE.main(["--list-roles"]) == 0
    output = capsys.readouterr().out
    assert "snowball" in output
    assert "robot-assistant" in output


def test_llamacpp_vision_classifier_uses_single_grammar_label(monkeypatch):
    captured = {}

    def fake_urlopen(request, **_kwargs):
        captured["body"] = json.loads(request.data)
        return FakeJsonResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "T"
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(MODULE, "urlopen", fake_urlopen)
    client = MODULE.LlamaCppChatClient(
        base_url="http://localhost:8080",
        model="qwen3-vl",
        timeout=1.0,
        num_predict=32,
        temperature=0.6,
    )

    route, reason = client.classify_vision_need(
        "你叫什么名字？",
        [{"role": "assistant", "content": "画面里有一件衣服。"}],
        True,
    )

    assert (route, reason) == ("TEXT", "Qwen semantic route label T")
    assert captured["body"]["stream"] is False
    assert captured["body"]["temperature"] == 0
    assert captured["body"]["max_tokens"] == 2
    assert captured["body"]["grammar"] == 'root ::= "V" | "T" | "U"'
    assert "只做分类，不回答用户" in captured["body"]["messages"][0]["content"]


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


def test_vision_strategy_defaults_to_see_and_keeps_legacy_auto_alias():
    parser = MODULE.build_parser()
    default_args = parser.parse_args(["--camera", "/dev/video0"])
    assert MODULE.resolve_vision_strategy(default_args) == "see"
    assert default_args.vision_router_history_turns == 4
    assert MODULE.resolve_vision_strategy(
        parser.parse_args(["--camera", "/dev/video0", "--vision-mode", "auto"])
    ) == "qwen"
    args = parser.parse_args(["--vision-strategy", "always"])

    with pytest.raises(ValueError, match="always vision strategy requires --camera"):
        MODULE.validate_args(args)
