from __future__ import annotations

import json

from star_runtime.core.control import ControlStamp
from star_runtime.core.timing import RuntimeTimingAudit


def test_turn_timeline_uses_observed_monotonic_events_and_emits_once(caplog):
    caplog.set_level("INFO", logger="star_runtime.core.timing")
    audit = RuntimeTimingAudit()
    stamp = ControlStamp("session", "turn", 1)
    base = 1_000_000_000

    assert audit.mark(stamp, "speech_start", at_ns=base)
    assert audit.mark(stamp, "speech_end", at_ns=base + 400_000_000)
    assert audit.mark(stamp, "vad_ready", at_ns=base + 600_000_000)
    assert audit.mark(stamp, "asr_start", at_ns=base + 610_000_000)
    assert audit.mark(stamp, "asr_final", at_ns=base + 710_000_000)
    assert audit.mark(stamp, "agent_start", at_ns=base + 720_000_000)
    assert audit.mark(stamp, "agent_first_delta", at_ns=base + 920_000_000)
    assert not audit.mark(stamp, "agent_first_delta", at_ns=base + 999_000_000)
    assert audit.mark(stamp, "agent_complete", at_ns=base + 1_500_000_000)
    assert audit.finish_if_complete(stamp) is None
    assert audit.mark(stamp, "playback_complete", at_ns=base + 2_000_000_000)

    payload = audit.finish_if_complete(stamp)

    assert payload is not None
    assert payload["outcome"] == "completed"
    assert payload["origin_monotonic_ns"] == base
    assert payload["durations_ms"]["vad_tail_ms"] == 200.0
    assert payload["durations_ms"]["asr_queue_ms"] == 10.0
    assert payload["durations_ms"]["asr_inference_ms"] == 100.0
    assert payload["durations_ms"]["agent_first_delta_ms"] == 200.0
    assert payload["timeline"][0] == {
        "event": "speech_start",
        "at_ms": 0.0,
        "since_previous_ms": 0.0,
    }
    assert "tts_first_pcm" in payload["missing_events"]
    assert audit.finish_if_complete(stamp) is None
    assert not audit.mark(stamp, "tts_first_pcm", at_ns=base + 3_000_000_000)

    lines = [line for line in caplog.messages if line.startswith("RUNTIME_TURN_TIMELINE ")]
    assert len(lines) == 1
    logged = json.loads(lines[0].split(" ", 1)[1])
    assert logged["turn_id"] == "turn"


def test_invalidated_turn_reports_missing_stages_without_inventing_timestamps():
    audit = RuntimeTimingAudit()
    stamp = ControlStamp("session", "interrupted", 2)
    audit.mark(stamp, "agent_start", at_ns=10)
    audit.mark(stamp, "invalidated", at_ns=20, reason="barge-in")

    payload = audit.finish(stamp, "invalidated", reason="barge-in")

    assert payload is not None
    assert [event["event"] for event in payload["timeline"]] == [
        "agent_start",
        "invalidated",
    ]
    assert payload["durations_ms"] == {}
    assert payload["finish"] == {"reason": "barge-in"}
    assert "hardware_write_begin" in payload["missing_events"]
