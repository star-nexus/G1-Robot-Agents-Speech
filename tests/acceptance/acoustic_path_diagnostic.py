#!/usr/bin/env python3
"""Capture and analyze synchronized robot-only render/raw/post-AEC audio.

This is an opt-in JP6.2 diagnostic. It does not change VAD thresholds or the
runtime cancellation policy. The three streams use one host monotonic clock:

* render timestamps mark the final pre-write AEC reference submission;
* raw/post-AEC timestamps are the same PortAudio capture callback endpoint;
* VAD timestamps are the exact early edge delivered to SpeechService.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import threading
import time
import wave
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import resample_poly

from star_runtime.apps.integrated_runtime import build_integrated_runtime
from star_runtime.apps.local_voice_agent import LocalAgentSettings
from star_runtime.core.events import TtsTextChunk
from star_runtime.speech.config import load_config


ROBOT_ONLY_TEXT = (
    "真正可靠的朋友会认真倾听你的想法会在你遇到困难时保持耐心会尊重你的选择"
    "会诚实表达自己的意见也会在漫长的相处中始终用行动兑现承诺。"
)


@dataclass(frozen=True)
class TraceBlock:
    at_ns: int
    sample_rate: int
    samples: np.ndarray


class AcousticTraceRecorder:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._accepting = True
        self._render: list[TraceBlock] = []
        self._raw: list[TraceBlock] = []
        self._post: list[TraceBlock] = []
        self._vad_edges_ns: list[int] = []

    def record_render_reference(
        self,
        samples: np.ndarray,
        sample_rate: int,
        submitted_ns: int,
    ) -> None:
        self._append(self._render, samples, sample_rate, submitted_ns)

    def record_raw_capture(
        self,
        samples: np.ndarray,
        sample_rate: int,
        captured_ns: int,
    ) -> None:
        self._append(self._raw, samples, sample_rate, captured_ns)

    def record_post_aec_capture(
        self,
        samples: np.ndarray,
        sample_rate: int,
        captured_ns: int,
    ) -> None:
        self._append(self._post, samples, sample_rate, captured_ns)

    def record_vad_edge(self, at_ns: int) -> None:
        with self._lock:
            if self._accepting:
                self._vad_edges_ns.append(int(at_ns))

    def freeze(self) -> None:
        """Atomically stop accepting blocks before analysis and persistence."""

        with self._lock:
            self._accepting = False

    def snapshot(
        self,
    ) -> tuple[list[TraceBlock], list[TraceBlock], list[TraceBlock], list[int]]:
        with self._lock:
            return (
                list(self._render),
                list(self._raw),
                list(self._post),
                list(self._vad_edges_ns),
            )

    def _append(
        self,
        target: list[TraceBlock],
        samples: np.ndarray,
        sample_rate: int,
        at_ns: int,
    ) -> None:
        block = TraceBlock(
            int(at_ns),
            int(sample_rate),
            np.asarray(samples, dtype=np.float32).reshape(-1).copy(),
        )
        with self._lock:
            if self._accepting:
                target.append(block)


def wait_for(predicate, description: str, timeout: float = 90.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise TimeoutError(f"timed out waiting for {description}")


def _resample(values: np.ndarray, input_rate: int, output_rate: int) -> np.ndarray:
    if input_rate == output_rate:
        return np.asarray(values, dtype=np.float32)
    divisor = math.gcd(input_rate, output_rate)
    return resample_poly(
        values,
        output_rate // divisor,
        input_rate // divisor,
    ).astype(np.float32)


def _block_bounds(block: TraceBlock, *, capture_endpoint: bool) -> tuple[int, int]:
    duration_ns = round(block.samples.size * 1_000_000_000 / block.sample_rate)
    if capture_endpoint:
        return block.at_ns - duration_ns, block.at_ns
    return block.at_ns, block.at_ns + duration_ns


def _render_to_grid(
    blocks: list[TraceBlock],
    *,
    origin_ns: int,
    sample_rate: int,
    size: int,
    capture_endpoint: bool,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros(size, dtype=np.float32)
    counts = np.zeros(size, dtype=np.uint16)
    for block in blocks:
        samples = _resample(block.samples, block.sample_rate, sample_rate)
        start_ns, _end_ns = _block_bounds(block, capture_endpoint=capture_endpoint)
        start = round((start_ns - origin_ns) * sample_rate / 1_000_000_000)
        source_start = max(0, -start)
        target_start = max(0, start)
        count = min(samples.size - source_start, size - target_start)
        if count <= 0:
            continue
        target = slice(target_start, target_start + count)
        values[target] += samples[source_start : source_start + count]
        counts[target] += 1
    valid = counts > 0
    values[valid] /= counts[valid]
    return values, valid


def _normalized_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 1600:
        return float("nan")
    x = x.astype(np.float64, copy=False) - float(np.mean(x))
    y = y.astype(np.float64, copy=False) - float(np.mean(y))
    denominator = math.sqrt(float(np.dot(x, x)) * float(np.dot(y, y)))
    if denominator <= 1e-12:
        return float("nan")
    return float(np.dot(x, y) / denominator)


def _rms_dbfs(values: np.ndarray) -> float | None:
    if not values.size:
        return None
    rms = math.sqrt(float(np.mean(np.square(values.astype(np.float64)))))
    return round(20 * math.log10(max(rms, 1e-9)), 3)


def analyze_trace(
    recorder: AcousticTraceRecorder,
    *,
    sample_rate: int = 16000,
    max_delay_ms: int = 300,
    window_ms: int = 700,
    hop_ms: int = 350,
    minimum_correlation: float = 0.5,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    render, raw, post, vad_edges_ns = recorder.snapshot()
    if not render or not raw or not post:
        raise RuntimeError(
            "trace requires non-empty render, raw capture, and post-AEC capture"
        )
    render_nominal_ms = float(
        np.median(
            [block.samples.size * 1000 / block.sample_rate for block in render]
        )
    )
    capture_nominal_ms = float(
        np.median([block.samples.size * 1000 / block.sample_rate for block in raw])
    )
    render_intervals_ms = np.diff([block.at_ns for block in render]) / 1_000_000
    capture_intervals_ms = np.diff([block.at_ns for block in raw]) / 1_000_000

    def continuity(
        intervals: np.ndarray,
        nominal_ms: float,
    ) -> dict[str, Any]:
        gaps = intervals > nominal_ms * 1.5
        return {
            "nominal_block_ms": round(nominal_ms, 3),
            "median_interval_ms": (
                None if not intervals.size else round(float(np.median(intervals)), 3)
            ),
            "max_interval_ms": (
                None if not intervals.size else round(float(np.max(intervals)), 3)
            ),
            "gaps_over_1_5x_nominal": int(np.count_nonzero(gaps)),
            "gap_excess_ms": round(
                float(np.sum(np.maximum(intervals - nominal_ms, 0))),
                3,
            ),
        }
    starts = [
        *(_block_bounds(block, capture_endpoint=False)[0] for block in render),
        *(_block_bounds(block, capture_endpoint=True)[0] for block in raw),
    ]
    ends = [
        *(_block_bounds(block, capture_endpoint=False)[1] for block in render),
        *(_block_bounds(block, capture_endpoint=True)[1] for block in raw),
    ]
    origin_ns = min(starts)
    end_ns = max(ends)
    size = math.ceil((end_ns - origin_ns) * sample_rate / 1_000_000_000) + 1
    render_grid, render_valid = _render_to_grid(
        render,
        origin_ns=origin_ns,
        sample_rate=sample_rate,
        size=size,
        capture_endpoint=False,
    )
    raw_grid, raw_valid = _render_to_grid(
        raw,
        origin_ns=origin_ns,
        sample_rate=sample_rate,
        size=size,
        capture_endpoint=True,
    )
    post_grid, post_valid = _render_to_grid(
        post,
        origin_ns=origin_ns,
        sample_rate=sample_rate,
        size=size,
        capture_endpoint=True,
    )

    active = np.flatnonzero(render_valid & (np.abs(render_grid) > 1e-4))
    if not active.size:
        raise RuntimeError("render trace contains no audible PCM")
    render_start = int(active[0])
    render_end = int(active[-1]) + 1
    window = round(window_ms * sample_rate / 1000)
    hop = round(hop_ms * sample_rate / 1000)
    max_lag = round(max_delay_ms * sample_rate / 1000)
    delay_windows: list[dict[str, Any]] = []
    starts_samples = list(range(render_start, max(render_start + 1, render_end - window + 1), hop))
    if not starts_samples:
        starts_samples = [render_start]

    for start in starts_samples:
        stop = min(render_end, start + window)
        if stop - start < sample_rate // 3:
            continue
        best: tuple[float, int, float] | None = None
        for lag in range(0, max_lag + 1):
            shifted_start = start + lag
            shifted_stop = stop + lag
            if shifted_stop > size:
                break
            mask = (
                render_valid[start:stop]
                & raw_valid[shifted_start:shifted_stop]
            )
            if np.count_nonzero(mask) < sample_rate // 3:
                continue
            correlation = _normalized_correlation(
                render_grid[start:stop][mask],
                raw_grid[shifted_start:shifted_stop][mask],
            )
            if not math.isfinite(correlation):
                continue
            score = abs(correlation)
            if best is None or score > best[0]:
                best = (score, lag, correlation)
        if best is None:
            continue
        _score, lag, correlation = best
        shifted_start = start + lag
        shifted_stop = stop + lag
        residual_mask = (
            render_valid[start:stop]
            & raw_valid[shifted_start:shifted_stop]
            & post_valid[shifted_start:shifted_stop]
        )
        raw_values = raw_grid[shifted_start:shifted_stop][residual_mask]
        post_values = post_grid[shifted_start:shifted_stop][residual_mask]
        raw_dbfs = _rms_dbfs(raw_values)
        post_dbfs = _rms_dbfs(post_values)
        delay_windows.append(
            {
                "center_ms": round(
                    ((start + stop) / 2 - render_start) * 1000 / sample_rate,
                    3,
                ),
                "delay_ms": round(lag * 1000 / sample_rate, 3),
                "correlation": round(correlation, 6),
                "accepted_for_delay": abs(correlation) >= minimum_correlation,
                "raw_echo_dbfs": raw_dbfs,
                "post_aec_dbfs": post_dbfs,
                "aec_attenuation_db": (
                    None
                    if raw_dbfs is None or post_dbfs is None
                    else round(raw_dbfs - post_dbfs, 3)
                ),
            }
        )

    if not delay_windows:
        raise RuntimeError("no correlation windows had sufficient synchronized data")
    accepted_windows = [
        item for item in delay_windows if item["accepted_for_delay"]
    ]
    if not accepted_windows:
        raise RuntimeError(
            "no correlation window met the minimum delay confidence "
            f"of {minimum_correlation}"
        )
    delays = np.asarray(
        [item["delay_ms"] for item in accepted_windows],
        dtype=float,
    )
    centers = np.asarray(
        [item["center_ms"] for item in accepted_windows],
        dtype=float,
    )
    median_delay = float(np.median(delays))
    delay_span = float(np.max(delays) - np.min(delays))
    slope_ms_per_s = 0.0
    if delays.size >= 2 and float(np.ptp(centers)) > 0:
        slope_ms_per_s = float(np.polyfit(centers / 1000.0, delays, 1)[0])
    drift_assessable = delays.size >= 3
    material_drift = delay_span > 20.0 if drift_assessable else None

    vad_positions = []
    for edge_ns in vad_edges_ns:
        edge_index = round((edge_ns - origin_ns) * sample_rate / 1_000_000_000)
        radius = round(0.1 * sample_rate)
        left = max(0, edge_index - radius)
        right = min(size, edge_index + radius)
        local = post_grid[left:right][post_valid[left:right]]
        vad_positions.append(
            {
                "edge_monotonic_ns": edge_ns,
                "since_render_start_ms": round(
                    (edge_index - render_start) * 1000 / sample_rate,
                    3,
                ),
                "post_aec_local_dbfs": _rms_dbfs(local),
                "post_aec_local_peak": (
                    None if not local.size else round(float(np.max(np.abs(local))), 6)
                ),
            }
        )

    raw_levels = [item["raw_echo_dbfs"] for item in accepted_windows]
    post_levels = [item["post_aec_dbfs"] for item in accepted_windows]
    attenuations = [item["aec_attenuation_db"] for item in accepted_windows]
    result = {
        "schema": "star.runtime.acoustic_trace.v1",
        "sample_rate": sample_rate,
        "render_blocks": len(render),
        "raw_capture_blocks": len(raw),
        "post_aec_blocks": len(post),
        "render_duration_ms": round((render_end - render_start) * 1000 / sample_rate, 3),
        "continuity": {
            "render_reference": continuity(
                render_intervals_ms,
                render_nominal_ms,
            ),
            "raw_capture": continuity(
                capture_intervals_ms,
                capture_nominal_ms,
            ),
        },
        "delay": {
            "median_ms": round(median_delay, 3),
            "min_ms": round(float(np.min(delays)), 3),
            "max_ms": round(float(np.max(delays)), 3),
            "span_ms": round(delay_span, 3),
            "slope_ms_per_s": round(slope_ms_per_s, 3),
            "material_drift": material_drift,
            "drift_assessable": drift_assessable,
            "material_drift_rule": "window delay span > 20 ms",
            "minimum_absolute_correlation": minimum_correlation,
            "accepted_windows": len(accepted_windows),
            "total_windows": len(delay_windows),
        },
        "residual": {
            "median_raw_echo_dbfs": round(float(np.median(raw_levels)), 3),
            "median_post_aec_dbfs": round(float(np.median(post_levels)), 3),
            "median_aec_attenuation_db": round(float(np.median(attenuations)), 3),
        },
        "windows": delay_windows,
        "vad_edges": vad_positions,
    }
    arrays = {
        "render": render_grid,
        "render_valid": render_valid,
        "raw_capture": raw_grid,
        "raw_valid": raw_valid,
        "post_aec": post_grid,
        "post_valid": post_valid,
    }
    return result, arrays


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    pcm = np.rint(np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


def save_trace(
    artifact_dir: Path,
    result: dict[str, Any],
    arrays: dict[str, np.ndarray],
    recorder: AcousticTraceRecorder,
    runtime_metrics: dict[str, Any],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    rate = int(result["sample_rate"])
    _write_wav(artifact_dir / "render_reference.wav", arrays["render"], rate)
    _write_wav(artifact_dir / "raw_microphone.wav", arrays["raw_capture"], rate)
    _write_wav(artifact_dir / "post_aec.wav", arrays["post_aec"], rate)
    render, raw, post, vad_edges = recorder.snapshot()
    np.savez_compressed(
        artifact_dir / "synchronized_streams.npz",
        **arrays,
        render_block_ns=np.asarray([block.at_ns for block in render], dtype=np.int64),
        raw_block_ns=np.asarray([block.at_ns for block in raw], dtype=np.int64),
        post_block_ns=np.asarray([block.at_ns for block in post], dtype=np.int64),
        vad_edges_ns=np.asarray(vad_edges, dtype=np.int64),
    )
    payload = dict(result)
    payload["runtime_metrics"] = runtime_metrics
    (artifact_dir / "analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--block-ms", type=int, default=20)
    parser.add_argument("--stream-delay-ms", type=int, default=80)
    parser.add_argument("--voice", default="Ono_Anna")
    parser.add_argument("--text", default=ROBOT_ONLY_TEXT)
    parser.add_argument("--force-cpu-asr", action="store_true")
    parser.add_argument("--settle-seconds", type=float, default=0.2)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(args.config, runtime_environment=os.environ)
    config = replace(
        config,
        audio=replace(config.audio, block_ms=args.block_ms),
        audio_processing=replace(
            config.audio_processing,
            mode="webrtc",
            stream_delay_ms=args.stream_delay_ms,
        ),
        transport=replace(config.transport, backend="inprocess"),
        tts=replace(config.tts, enabled=True),
    )
    if args.force_cpu_asr:
        config = replace(
            config,
            sensevoice=replace(config.sensevoice, device="cpu"),
        )
    config.validate()
    runtime = build_integrated_runtime(
        config,
        LocalAgentSettings(
            system_prompt="回答适合直接朗读。",
            max_tokens=128,
            history_turns=1,
            tts_voice=args.voice,
            tts_language="Chinese",
        ),
    )
    recorder = AcousticTraceRecorder()
    runtime.speech.audio_source.set_acoustic_trace_sink(recorder)
    set_processor_trace = getattr(
        runtime.speech.audio_processor,
        "set_acoustic_trace_sink",
        None,
    )
    if set_processor_trace is None:
        raise RuntimeError("selected audio processor does not support acoustic trace")
    set_processor_trace(recorder)
    # Acoustic echo may become a finalized ASR event after a false VAD edge.
    # Keep that event out of the Agent so a second robot response cannot pollute
    # the fixed robot-only stimulus. Control/ASR/VAD behavior remains observable.
    set_speech_handler = getattr(
        runtime.speech.transport,
        "_set_speech_handler",
        None,
    )
    if set_speech_handler is not None:
        set_speech_handler(lambda _event: None)
    request_id = "acoustic-robot-only"
    try:
        runtime.start()
        stamp = runtime.speech.control.begin_turn("acoustic-robot-only")
        runtime.speech.tts.accept(
            TtsTextChunk(
                request_id=request_id,
                sequence=0,
                text=args.text,
                is_final=True,
                language="Chinese",
                voice=args.voice,
                created_unix_ns=time.time_ns(),
                source="acoustic-path-diagnostic",
                session_id=stamp.session_id,
                turn_id=stamp.turn_id,
                epoch=stamp.epoch,
            )
        )
        wait_for(
            lambda: bool(runtime.metrics()["tts_playback_active"]),
            "robot-only playback start",
        )
        wait_for(
            lambda: not bool(runtime.metrics()["tts_playback_active"]),
            "robot-only playback completion or VAD interruption",
        )
        time.sleep(args.settle_seconds)
        recorder.freeze()
        metrics = runtime.metrics()
        result, arrays = analyze_trace(recorder)
        result["run"] = {
            "audio_block_ms": args.block_ms,
            "actual_input_latency_ms": metrics.get("audio_input_latency_ms"),
            "stream_delay_ms": args.stream_delay_ms,
            "vad_edge_count": len(result["vad_edges"]),
            "control_invalidations": metrics.get("control_invalidations"),
            "playback_interruptions": metrics.get("audio_output_interruptions"),
        }
        save_trace(
            Path(args.artifact_dir),
            result,
            arrays,
            recorder,
            metrics,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
