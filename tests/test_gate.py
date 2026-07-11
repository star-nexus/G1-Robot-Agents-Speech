from g1_speech.gate import PlaybackGate


def test_playback_and_resume_delay_are_muted():
    gate = PlaybackGate(resume_delay_ms=250, max_active_seconds=30)
    gate.set_active(True, now_ns=1_000_000_000)
    assert gate.is_muted(now_ns=2_000_000_000)

    gate.set_active(False, now_ns=2_000_000_000)
    assert gate.is_muted(now_ns=2_200_000_000)
    assert not gate.is_muted(now_ns=2_251_000_000)


def test_stuck_playback_gate_fails_open_after_limit():
    gate = PlaybackGate(resume_delay_ms=100, max_active_seconds=2)
    gate.set_active(True, now_ns=1_000_000_000)
    assert gate.is_muted(now_ns=3_100_000_000)
    assert not gate.is_muted(now_ns=3_201_000_000)
