from __future__ import annotations

import json
import stat
import time

from star_runtime.agent import ConversationalLoop, WindowMemory
from star_runtime.perception import ImageFrame
from star_runtime.perception.vision import (
    TEXT,
    UNCERTAIN,
    VISION,
    LatestFrameCamera,
    MjpegStreamParser,
    RoutedVisionInput,
    VisionRouteRecorder,
    VisionRouter,
)


class StaticCamera:
    def __init__(self, data: bytes = b"jpeg") -> None:
        self.frame = ImageFrame(data, time.monotonic())
        self.max_ages: list[float] = []

    def snapshot(self, max_age_seconds: float) -> ImageFrame:
        self.max_ages.append(max_age_seconds)
        return self.frame


class RecordingModel:
    def __init__(self) -> None:
        self.messages = []

    def stream(self, messages):
        self.messages = messages
        return iter(("回答",))


def test_mjpeg_parser_extracts_split_and_multiple_frames():
    parser = MjpegStreamParser()

    assert parser.feed(b"noise\xff") == []
    assert parser.feed(b"\xd8one\xff\xd9\xff\xd8two") == [b"\xff\xd8one\xff\xd9"]
    assert parser.feed(b"\xff\xd9") == [b"\xff\xd8two\xff\xd9"]


def test_camera_command_copies_mjpeg_without_reencoding():
    command = LatestFrameCamera(
        device="/dev/video3", width=1280, height=720, fps=10
    ).command()

    assert command[command.index("-input_format") + 1] == "mjpeg"
    assert command[command.index("-video_size") + 1] == "1280x720"
    assert command[command.index("-c:v") + 1] == "copy"


def test_explicit_and_rescue_language_bypass_semantic_classifier():
    calls = []
    router = VisionRouter(
        classifier=lambda *_args: calls.append(True),  # type: ignore[arg-type,return-value]
    )

    explicit = router.decide("你在画面里看到了什么？", [])
    rescue = router.decide("不对，你再仔细看一下。", [])

    assert explicit.effective_route == VISION
    assert explicit.source == "explicit_trigger"
    assert rescue.source == "rescue_trigger"
    assert calls == []


def test_semantic_text_and_uncertain_routes_are_handled():
    text = VisionRouter(
        classifier=lambda *_args: (TEXT, "identity question")
    ).decide("你叫什么名字？", [])
    uncertain = VisionRouter(
        classifier=lambda *_args: (UNCERTAIN, "ambiguous reference")
    ).decide("那个呢？", [])

    assert not text.attach_image
    assert uncertain.requested_route == UNCERTAIN
    assert uncertain.attach_image


def test_classifier_failure_fails_safe_to_a_fresh_frame():
    def fail(*_args):
        raise RuntimeError("server unavailable")

    decision = VisionRouter(classifier=fail).decide("门关了吗？", [])

    assert decision.attach_image
    assert decision.source == "classifier_fallback"
    assert "server unavailable" in (decision.classifier_error or "")


def test_router_passes_recent_context_and_previous_visual_state():
    received = []

    def classify(text, messages, last_used_vision):
        received.append((text, messages, last_used_vision))
        return VISION, "visual follow-up"

    router = VisionRouter(classifier=classify)
    first = router.decide("看看画面", [])
    router.observe(first)
    history = [
        {"role": "user", "content": "画面里有什么？"},
        {"role": "assistant", "content": "有一扇门。"},
    ]

    assert router.decide("门呢？", history).attach_image
    assert received == [("门呢？", history, True)]


def test_routed_visual_turn_attaches_only_current_image_and_keeps_text_memory():
    camera = StaticCamera(b"frame")
    vision = RoutedVisionInput(VisionRouter(mode="always"), camera)
    model = RecordingModel()
    memory = WindowMemory(max_turns=1)
    memory.record_turn("上一问", "上一答")
    loop = ConversationalLoop(
        system_prompt="角色",
        model=model,
        memory=memory,
        vision=vision,
    )

    assert list(loop.stream_response("这里有什么？")) == ["回答"]

    assert model.messages[1:3] == [
        {"role": "user", "content": "上一问"},
        {"role": "assistant", "content": "上一答"},
    ]
    content = model.messages[-1]["content"]
    assert isinstance(content, list)
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[1]["text"].endswith("这里有什么？")
    assert camera.max_ages == [2.0]


def test_text_route_does_not_touch_camera():
    camera = StaticCamera()
    vision = RoutedVisionInput(VisionRouter(mode="off"), camera)

    assert vision.image_for("你好", []) is None
    assert camera.max_ages == []


def test_route_recorder_writes_private_jsonl(tmp_path):
    path = tmp_path / "state" / "router.jsonl"
    recorder = VisionRouteRecorder(path)
    decision = VisionRouter(mode="off").decide("你好", [])

    recorder.append(decision)

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["schema_version"] == 1
    assert record["effective_route"] == TEXT
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
