from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).parent / "acceptance" / "acoustic_path_diagnostic.py"
)
SPEC = importlib.util.spec_from_file_location("acoustic_path_diagnostic", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
diagnostic = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = diagnostic
SPEC.loader.exec_module(diagnostic)


def test_acoustic_trace_recovers_stable_delay_and_aec_attenuation():
    sample_rate = 16000
    origin_ns = 10_000_000_000
    duration_seconds = 4
    delay_samples = round(0.08 * sample_rate)
    rng = np.random.default_rng(7)
    render = rng.normal(0.0, 0.08, duration_seconds * sample_rate).astype(np.float32)
    raw = np.zeros(render.size + delay_samples + sample_rate, dtype=np.float32)
    raw[delay_samples : delay_samples + render.size] = render * 0.5
    post = raw * 0.1
    recorder = diagnostic.AcousticTraceRecorder()

    render_block = sample_rate // 100
    for start in range(0, render.size, render_block):
        values = render[start : start + render_block]
        recorder.record_render_reference(
            values,
            sample_rate,
            origin_ns + round(start * 1_000_000_000 / sample_rate),
        )
    capture_block = sample_rate // 50
    for start in range(0, raw.size, capture_block):
        raw_values = raw[start : start + capture_block]
        post_values = post[start : start + capture_block]
        endpoint_ns = origin_ns + round(
            (start + raw_values.size) * 1_000_000_000 / sample_rate
        )
        recorder.record_raw_capture(raw_values, sample_rate, endpoint_ns)
        recorder.record_post_aec_capture(post_values, sample_rate, endpoint_ns)
    recorder.record_vad_edge(origin_ns + 2_000_000_000)

    result, _arrays = diagnostic.analyze_trace(recorder)

    assert result["delay"]["median_ms"] == 80.0
    assert result["delay"]["span_ms"] == 0.0
    assert result["delay"]["drift_assessable"] is True
    assert result["delay"]["material_drift"] is False
    assert result["continuity"]["render_reference"][
        "gaps_over_1_5x_nominal"
    ] == 0
    assert result["continuity"]["raw_capture"][
        "gaps_over_1_5x_nominal"
    ] == 0
    assert 19.9 <= result["residual"]["median_aec_attenuation_db"] <= 20.1
    assert abs(result["vad_edges"][0]["since_render_start_ms"] - 2000.0) < 0.1


def test_acoustic_trace_freeze_keeps_analysis_and_saved_snapshot_identical():
    recorder = diagnostic.AcousticTraceRecorder()
    frame = np.ones(160, dtype=np.float32)
    recorder.record_render_reference(frame, 16000, 1_000_000_000)
    recorder.record_raw_capture(frame, 16000, 1_010_000_000)
    recorder.record_post_aec_capture(frame, 16000, 1_010_000_000)
    recorder.record_vad_edge(1_005_000_000)
    recorder.freeze()

    recorder.record_render_reference(frame, 16000, 2_000_000_000)
    recorder.record_raw_capture(frame, 16000, 2_010_000_000)
    recorder.record_post_aec_capture(frame, 16000, 2_010_000_000)
    recorder.record_vad_edge(2_005_000_000)

    render, raw, post, edges = recorder.snapshot()
    assert len(render) == len(raw) == len(post) == 1
    assert edges == [1_005_000_000]
