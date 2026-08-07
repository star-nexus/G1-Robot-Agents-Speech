from __future__ import annotations

import json
import stat

from g1_speech.vision_routing import (
    SeeVisionStrategy,
    TEXT,
    UNCERTAIN,
    VISION,
    VisionRouteRecorder,
    VisionRouter,
)


def test_explicit_visual_request_bypasses_classifier():
    calls = []
    router = VisionRouter(
        mode="auto",
        classifier=lambda *_args: calls.append(True),  # type: ignore[arg-type,return-value]
    )

    decision = router.decide("你在画面里看到了什么？", [])

    assert decision.effective_route == VISION
    assert decision.source == "explicit_trigger"
    assert calls == []


def test_visual_correction_is_recorded_as_rescue_trigger():
    router = VisionRouter(
        mode="auto",
        classifier=lambda *_args: (TEXT, "not used"),
    )

    decision = router.decide("不对，你再仔细看一下。", [])

    assert decision.attach_image
    assert decision.source == "rescue_trigger"
    assert decision.explicit_trigger == "仔细看"


def test_semantic_text_route_does_not_attach_image():
    router = VisionRouter(
        mode="auto",
        classifier=lambda *_args: (TEXT, "identity question"),
    )

    decision = router.decide("你叫什么名字？", [])

    assert not decision.attach_image
    assert decision.source == "semantic_classifier"


def test_uncertain_and_classifier_failure_fail_safe_to_vision():
    uncertain = VisionRouter(
        mode="auto",
        classifier=lambda *_args: (UNCERTAIN, "ambiguous reference"),
    ).decide("那个呢？", [])

    def fail(*_args):
        raise RuntimeError("server unavailable")

    failed = VisionRouter(mode="auto", classifier=fail).decide("门关了吗？", [])

    assert uncertain.requested_route == UNCERTAIN
    assert uncertain.attach_image
    assert failed.requested_route == UNCERTAIN
    assert failed.attach_image
    assert "server unavailable" in (failed.classifier_error or "")


def test_router_passes_recent_context_and_visual_state_to_classifier():
    received = []

    def classify(text, messages, last_used_vision):
        received.append((text, messages, last_used_vision))
        return VISION, "visual follow-up"

    router = VisionRouter(mode="auto", classifier=classify)
    router.observe(router.decide("看看画面", []))
    history = [
        {"role": "user", "content": "画面里有什么？"},
        {"role": "assistant", "content": "有衣服和挂衣架。"},
    ]

    decision = router.decide("挂衣架有吗？", history)

    assert decision.attach_image
    assert received == [("挂衣架有吗？", history, True)]


def test_see_strategy_uses_classifier_instead_of_main_generation():
    received = []

    def classify(text, messages, last_used_vision):
        received.append((text, messages, last_used_vision))
        return TEXT, "independent classifier emitted [TEXT]"

    strategy = SeeVisionStrategy(classify)
    emitted = []
    main_history = [{"role": "assistant", "content": "main answer"}]

    prepared = strategy.prepare(
        "Who are you?",
        main_history,
        [{"role": "user", "content": "Who are you?"}],
        emitted.append,
    )

    assert prepared.decision.effective_route == TEXT
    assert prepared.decision.source == "see_classifier"
    assert prepared.direct_response is None
    assert emitted == []
    assert received == [("Who are you?", main_history, False)]


def test_see_strategy_classifier_and_failure_route_to_vision():
    marker = SeeVisionStrategy(
        lambda *_args: (VISION, "independent classifier emitted [SEE]")
    ).prepare("门关了吗？", [], [], lambda _text: None)

    def fail(*_args):
        raise RuntimeError("server unavailable")

    fallback = SeeVisionStrategy(fail).prepare("门关了吗？", [], [], lambda _text: None)

    assert marker.decision.attach_image
    assert marker.decision.source == "see_classifier"
    assert fallback.decision.attach_image
    assert fallback.decision.source == "see_classifier_fallback"


def test_see_strategy_reset_clears_classifier_session():
    reset_calls = []
    strategy = SeeVisionStrategy(
        lambda *_args: (VISION, "[SEE]"),
        lambda: reset_calls.append(True),
    )
    strategy.observe(
        strategy.prepare("看画面", [], [], lambda _text: None).decision
    )

    strategy.reset()

    assert not strategy.last_used_vision
    assert reset_calls == [True]


def test_recorder_writes_private_jsonl(tmp_path):
    path = tmp_path / "state" / "router.jsonl"
    recorder = VisionRouteRecorder(path)

    recorder.append({"event_type": "interaction", "user_text": "你好"})

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["schema_version"] == 1
    assert record["user_text"] == "你好"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
