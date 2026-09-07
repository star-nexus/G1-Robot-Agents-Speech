from __future__ import annotations

from star_runtime.core.control import ControlStamp, RuntimeControlPlane
from star_runtime.core.timing import RuntimeTimingAudit
from star_runtime.speech.service import SpeechService


class _Tts:
    def __init__(self, interrupted: bool = True, calls=None) -> None:
        self.interrupted = interrupted
        self.reasons: list[str] = []
        self.provider_cancels = 0
        self.calls = calls

    def interrupt(self, *, reason: str = "vad") -> bool:
        interrupted = self.preempt_output(reason=reason)
        if interrupted:
            self.cancel_provider()
        return interrupted

    def preempt_output(self, *, reason: str = "vad") -> bool:
        self.reasons.append(reason)
        if self.calls is not None:
            self.calls.append("output-preempted")
        return self.interrupted

    def cancel_provider(self) -> None:
        self.provider_cancels += 1
        if self.calls is not None:
            self.calls.append("tts-provider-cancelled")


class _Transport:
    def __init__(self, calls=None) -> None:
        self.events = []
        self.calls = calls

    def publish_control(self, event) -> None:
        self.events.append(event)
        if self.calls is not None:
            self.calls.append("agent-cancel-published")


def _service(*, timing: RuntimeTimingAudit | None = None) -> SpeechService:
    service = object.__new__(SpeechService)
    service.control = RuntimeControlPlane("session")
    service.timing_audit = timing
    service.tts = _Tts()
    service.transport = _Transport()
    return service


def test_vad_edge_invalidates_then_preempts_active_response():
    timing = RuntimeTimingAudit()
    service = _service(timing=timing)
    old = service.control.begin_turn("old-turn")
    detected_ns = 12_345_678_900

    service._on_speech_start(detected_ns)  # noqa: SLF001

    assert service.control.current == ControlStamp(
        old.session_id,
        old.turn_id,
        old.epoch + 1,
    )
    assert service.tts.reasons == ["vad"]
    assert service.tts.provider_cancels == 1
    assert len(service.transport.events) == 1
    event = service.transport.events[0]
    assert event.stamp == old
    assert event.reason == "barge-in"
    events = timing.snapshot(old)
    assert events["vad_speech_edge"].monotonic_ns == detected_ns
    assert events["invalidated"].monotonic_ns >= detected_ns
    assert events["vad_output_preempted"].monotonic_ns >= events[
        "invalidated"
    ].monotonic_ns


def test_vad_preemption_cuts_output_before_provider_cancellation():
    calls = []
    service = _service()
    service.tts = _Tts(calls=calls)
    service.transport = _Transport(calls=calls)
    service.control.begin_turn("old-turn")

    service._on_speech_start(12_345_678_900)  # noqa: SLF001

    assert calls == [
        "output-preempted",
        "agent-cancel-published",
        "tts-provider-cancelled",
    ]


def test_asr_final_stale_output_cleanup_cannot_invalidate_new_turn():
    timing = RuntimeTimingAudit()
    service = _service(timing=timing)
    previous = service.control.begin_turn("old-turn")
    new = service.control.begin_turn("new-turn")

    service._flush_stale_output(previous, new)  # noqa: SLF001

    assert service.control.current == new
    assert service.control.invalidations == 0
    assert service.tts.reasons == ["asr-final-superseded"]
    assert service.tts.provider_cancels == 1
    assert service.transport.events == []
    cleanup = timing.snapshot(previous)["asr_final_fallback_cleanup"]
    assert cleanup.fields == {
        "by_turn_id": new.turn_id,
        "output_interrupted": True,
    }
