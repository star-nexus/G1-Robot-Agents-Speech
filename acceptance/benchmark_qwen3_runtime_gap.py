"""Strict same-runtime attribution benchmark for direct and VAD-fed Qwen3-ASR.

This harness intentionally does not optimize the model.  It keeps one model object
on one Python/CUDA thread and separates input preparation, synchronized ASR stage
profiling, profiling-overhead A/B, token-length scaling, and shape tracing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import sys
import threading
import time
import traceback
import wave
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from g1_speech.asr import create_asr_engine
from g1_speech.config import load_config
from g1_speech.contracts import AudioChunk, Utterance
from g1_speech.qwen3_asr import Qwen3AsrEngine
from g1_speech.vad import SileroVadSegmenter

from qwen3_benchmarking import (
    PROFILE_METRICS,
    load_audio,
    percentile,
    summarize_profiles,
)


ENVIRONMENT_KEYS = (
    "PATH",
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
    "CUDA_HOME",
    "CUDA_VISIBLE_DEVICES",
    "TRITON_PTXAS_PATH",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCH_LOGS",
    "TORCH_COMPILE_DEBUG",
    "TORCHDYNAMO_VERBOSE",
    "TORCH_CUDA_ARCH_LIST",
    "PYTORCH_CUDA_ALLOC_CONF",
    "CUDACXX",
    "CPATH",
    "CPLUS_INCLUDE_PATH",
)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _shape(value: Any) -> dict[str, Any]:
    return {
        "shape": list(getattr(value, "shape", ())),
        "dtype": str(getattr(value, "dtype", "unknown")),
        "device": str(getattr(value, "device", "cpu")),
    }


def _text_fingerprint(text: str) -> dict[str, Any]:
    """Preserve determinism evidence without publishing host-local transcripts."""
    return {
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text_characters": len(text),
    }


def _tensor_shapes(values: Any) -> dict[str, dict[str, Any]]:
    if hasattr(values, "items"):
        return {
            str(name): _shape(value)
            for name, value in values.items()
            if hasattr(value, "shape")
        }
    return {}


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = np.round(clipped * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _extract_first_vad_utterance(
    samples: np.ndarray,
    sample_rate: int,
    config,
) -> tuple[Utterance, dict[str, Any]]:
    vad = SileroVadSegmenter(settings=config.vad, sample_rate=sample_rate)
    block = round(sample_rate * config.audio.block_ms / 1000)
    synthetic_ns = 10_000_000_000
    fed_samples = 0
    started = time.perf_counter()
    utterance = None
    for offset in range(0, samples.size, block):
        frame = samples[offset : offset + block]
        fed_samples += frame.size
        synthetic_ns += round(frame.size * 1_000_000_000 / sample_rate)
        segments = vad.accept(AudioChunk(frame, sample_rate, synthetic_ns))
        if segments:
            utterance = segments[0]
            break
    silence_blocks = 0
    silence = np.zeros(block, dtype=np.float32)
    while utterance is None and silence_blocks < 40:
        silence_blocks += 1
        synthetic_ns += round(block * 1_000_000_000 / sample_rate)
        segments = vad.accept(AudioChunk(silence, sample_rate, synthetic_ns))
        if segments:
            utterance = segments[0]
    elapsed_ms = (time.perf_counter() - started) * 1000
    if utterance is None:
        raise RuntimeError("VAD did not emit an utterance within four seconds of silence")
    return utterance, {
        "source_sample_count": int(samples.size),
        "fed_source_samples": fed_samples,
        "utterance_sample_count": int(utterance.samples.size),
        "utterance_audio_seconds": utterance.samples.size / sample_rate,
        "silence_blocks_after_source": silence_blocks,
        "vad_preparation_ms": elapsed_ms,
    }


def _processor_shapes(engine: Qwen3AsrEngine, samples: np.ndarray) -> dict[str, Any]:
    request: dict[str, Any] = {"audio": samples}
    if engine._language:  # noqa: SLF001 - benchmark fingerprint
        request["language"] = engine._language  # noqa: SLF001
    if engine._prompt:  # noqa: SLF001
        request["prompt"] = engine._prompt  # noqa: SLF001
    processed = engine._processor.apply_transcription_request(**request)  # noqa: SLF001
    shapes = _tensor_shapes(processed)
    return {
        "audio_sample_count": int(samples.size),
        "audio_seconds": samples.size / engine._sample_rate,  # noqa: SLF001
        "processor_output_shapes": shapes,
        "input_ids_shape": shapes.get("input_ids", {}).get("shape"),
        "prefill_sequence_length": (
            shapes.get("input_ids", {}).get("shape", [None, None])[-1]
            if shapes.get("input_ids")
            else None
        ),
        "audio_feature_shape": next(
            (
                details["shape"]
                for name, details in shapes.items()
                if "feature" in name or "audio" in name
            ),
            None,
        ),
    }


def _runtime_fingerprint(engine: Qwen3AsrEngine, config, config_path: Path) -> dict[str, Any]:
    import torch
    import transformers

    try:
        import triton

        triton_info = {
            "available": True,
            "version": getattr(triton, "__version__", "unknown"),
            "file": getattr(triton, "__file__", None),
        }
    except Exception as error:  # noqa: BLE001
        triton_info = {"available": False, "error": repr(error)}

    model = engine._model  # noqa: SLF001
    language_model = getattr(getattr(model, "model", None), "language_model", None)
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None and hasattr(generation_config, "to_dict"):
        generation_config = generation_config.to_dict()
    compile_config = engine._generation_compile_config  # noqa: SLF001
    return {
        "recorded_at": datetime.now().astimezone().isoformat(),
        "argv": sys.argv,
        "argv0": sys.argv[0],
        "sys_executable": sys.executable,
        "python_version": platform.python_version(),
        "python_build": platform.python_build(),
        "platform": platform.platform(),
        "thread": {"name": threading.current_thread().name, "ident": threading.get_ident()},
        "torch": {
            "version": torch.__version__,
            "file": torch.__file__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        },
        "transformers": {
            "version": transformers.__version__,
            "file": transformers.__file__,
        },
        "triton": triton_info,
        "engine": {
            "class": f"{type(engine).__module__}.{type(engine).__qualname__}",
            "object_id": id(engine),
            "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
            "model_object_id": id(model),
            "language_model_class": (
                f"{type(language_model).__module__}.{type(language_model).__qualname__}"
                if language_model is not None
                else None
            ),
            "language_model_object_id": id(language_model) if language_model else None,
            "actual_device": str(engine._device),  # noqa: SLF001
            "actual_dtype": str(engine._dtype),  # noqa: SLF001
            "actual_attention_implementation": engine._effective_attention_implementation,  # noqa: SLF001
            "cache_implementation": engine._cache_implementation,  # noqa: SLF001
            "compile_model": engine._compile_model,  # noqa: SLF001
            "compile_mode": engine._compile_mode,  # noqa: SLF001
            "compile_dynamic": engine._compile_dynamic,  # noqa: SLF001
            "generation_compile_config": _json_safe(compile_config),
            "max_new_tokens": engine._max_new_tokens,  # noqa: SLF001
            "generation_config": _json_safe(generation_config),
        },
        "service": {
            "config_path": str(config_path),
            "configured_python": os.environ.get("SPEECH_PYTHON_GPU"),
            "configured_gpu_config": os.environ.get("SPEECH_CONFIG_GPU"),
            "launch_description": "direct benchmark process; production uses scripts/run-speech-service gpu dds",
        },
        "config": _json_safe(config),
        "environment": {name: os.environ.get(name) for name in ENVIRONMENT_KEYS},
    }


def _profile_run(
    engine: Qwen3AsrEngine,
    utterance: Utterance,
    *,
    path: str,
    run: int,
) -> dict[str, Any]:
    result, profile = engine.transcribe_profiled(utterance)
    return {
        "path": path,
        "run": run,
        "engine_object_id": id(engine),
        "model_object_id": id(engine._model),  # noqa: SLF001
        "sample_object_id": id(utterance.samples),
        "audio_sample_count": int(utterance.samples.size),
        **_text_fingerprint(result.text),
        **profile.as_dict(),
    }


def _same_input_ab(
    engine: Qwen3AsrEngine,
    controlled: np.ndarray,
    controlled_vad: Utterance,
    real_vad: Utterance,
    sample_rate: int,
    warmup: int,
    runs: int,
) -> dict[str, Any]:
    cases = [
        ("controlled_direct_full", Utterance(controlled, sample_rate, 0, 0)),
        ("controlled_vad_extracted_e2e", controlled_vad),
        (
            "controlled_vad_extracted_direct",
            Utterance(controlled_vad.samples, sample_rate, 0, 0),
        ),
        ("real_vad_extracted_e2e", real_vad),
        ("real_vad_extracted_direct", Utterance(real_vad.samples, sample_rate, 0, 0)),
    ]
    warmup_rows = []
    for case_name, utterance in cases:
        for warmup_index in range(warmup):
            warmup_rows.append(
                _profile_run(
                    engine,
                    utterance,
                    path=case_name,
                    run=-(warmup - warmup_index),
                )
            )
    rows = []
    # Interleave paths so temperature and clock drift cannot systematically favor
    # one path. Reverse every other pass to balance first/last ordering.
    for run in range(1, runs + 1):
        ordered = cases if run % 2 else list(reversed(cases))
        for case_name, utterance in ordered:
            rows.append(_profile_run(engine, utterance, path=case_name, run=run))
    output_cases = []
    for case_name, utterance in cases:
        selected = [row for row in rows if row["path"] == case_name]
        text_hashes = {row["text_sha256"] for row in selected}
        if len(text_hashes) != 1:
            raise RuntimeError(f"non-deterministic transcript for {case_name}")
        profiles = [
            {key: row[key] for key in PROFILE_METRICS}
            for row in selected
        ]
        output_cases.append(
            {
                "name": case_name,
                "audio_sample_count": int(utterance.samples.size),
                "audio_seconds": utterance.samples.size / sample_rate,
                "sample_object_id": id(utterance.samples),
                "text_sha256": selected[0]["text_sha256"],
                "text_characters": selected[0]["text_characters"],
                "summary": summarize_profiles(profiles),
            }
        )
    return {"warmup_rows": warmup_rows, "runs": rows, "cases": output_cases}


def _profiling_overhead_ab(
    engine: Qwen3AsrEngine,
    utterance: Utterance,
    warmup: int,
    runs: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        engine._transcribe(utterance, collect_profile=False)  # noqa: SLF001
        engine.transcribe_profiled(utterance)
    rows = []
    for run in range(1, runs + 1):
        order = ("unprofiled", "synchronized_profile")
        if run % 2 == 0:
            order = tuple(reversed(order))
        for mode in order:
            if mode == "unprofiled":
                result, _ = engine._transcribe(  # noqa: SLF001
                    utterance, collect_profile=False
                )
                rows.append(
                    {
                        "mode": mode,
                        "run": run,
                        **_text_fingerprint(result.text),
                        "total_ms": result.inference_ms,
                        "generate_ms": None,
                    }
                )
            else:
                result, profile = engine.transcribe_profiled(utterance)
                rows.append(
                    {
                        "mode": mode,
                        "run": run,
                        **_text_fingerprint(result.text),
                        "total_ms": profile.total_ms,
                        "generate_ms": profile.generate_ms,
                    }
                )
    summaries = {}
    for mode in ("unprofiled", "synchronized_profile"):
        selected = [row for row in rows if row["mode"] == mode]
        text_hashes = {row["text_sha256"] for row in selected}
        if len(text_hashes) != 1:
            raise RuntimeError(f"non-deterministic transcript for {mode}")
        summaries[mode] = {
            "total_ms": _summary([row["total_ms"] for row in selected]),
            "generate_ms": (
                _summary([row["generate_ms"] for row in selected])
                if mode == "synchronized_profile"
                else None
            ),
            "text_sha256s": sorted(text_hashes),
        }
    baseline = summaries["unprofiled"]["total_ms"]["mean"]
    profiled = summaries["synchronized_profile"]["total_ms"]["mean"]
    return {
        "metric_note": (
            "generate_ms is intentionally unavailable when profiling is disabled; "
            "inventing a stage boundary would invalidate the no-instrumentation control"
        ),
        "rows": rows,
        "summary": summaries,
        "profiled_minus_unprofiled_ms": profiled - baseline,
        "profiled_overhead_percent": (profiled / baseline - 1) * 100,
    }


def _linear_regression(points: list[tuple[float, float]]) -> dict[str, float]:
    x = np.asarray([point[0] for point in points], dtype=np.float64)
    y = np.asarray([point[1] for point in points], dtype=np.float64)
    matrix = np.column_stack((np.ones_like(x), x))
    intercept, slope = np.linalg.lstsq(matrix, y, rcond=None)[0]
    predicted = intercept + slope * x
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1 - residual / total if total else 1.0
    return {
        "estimated_fixed_generation_ms": float(intercept),
        "estimated_ms_per_generated_token": float(slope),
        "estimated_decode_tokens_per_second": float(1000 / slope) if slope > 0 else float("inf"),
        "r_squared": r_squared,
    }


def _token_scaling(
    engine: Qwen3AsrEngine,
    samples: np.ndarray,
    sample_rate: int,
    targets: list[int],
    warmup: int,
    runs: int,
) -> dict[str, Any]:
    utterance = Utterance(samples, sample_rate, 0, 0)
    original_max = engine._max_new_tokens  # noqa: SLF001
    rows = []
    cases = []
    try:
        for target in targets:
            engine._max_new_tokens = target  # noqa: SLF001 - controlled variable
            warmup_measurements = []
            for index in range(warmup):
                warmup_measurements.append(
                    _profile_run(
                        engine,
                        utterance,
                        path=f"max_new_tokens_{target}",
                        run=-(warmup - index),
                    )
                )
            case_rows = []
            for run in range(1, runs + 1):
                row = _profile_run(
                    engine,
                    utterance,
                    path=f"max_new_tokens_{target}",
                    run=run,
                )
                row["max_new_tokens"] = target
                rows.append(row)
                case_rows.append(row)
            text_hashes = {row["text_sha256"] for row in case_rows}
            if len(text_hashes) != 1:
                raise RuntimeError(
                    f"non-deterministic transcript for max_new_tokens={target}"
                )
            profiles = [
                {key: row[key] for key in PROFILE_METRICS}
                for row in case_rows
            ]
            cases.append(
                {
                    "max_new_tokens": target,
                    "generated_tokens": statistics.fmean(
                        row["generated_tokens"] for row in case_rows
                    ),
                    "text_sha256": case_rows[0]["text_sha256"],
                    "text_characters": case_rows[0]["text_characters"],
                    "first_request_ms": warmup_measurements[0]["generate_ms"],
                    "warmup_generate_ms": [
                        item["generate_ms"] for item in warmup_measurements
                    ],
                    "summary": summarize_profiles(profiles),
                }
            )
    finally:
        engine._max_new_tokens = original_max  # noqa: SLF001
    points = [
        (
            case["generated_tokens"],
            case["summary"]["generate_ms"]["mean"],
        )
        for case in cases
    ]
    regression = _linear_regression(points)
    regression["model"] = "generate_ms = fixed_ms + generated_tokens * ms_per_token"
    regression["first_token_latency"] = None
    regression["first_token_latency_omission"] = (
        "Not measured: token stream callbacks force device-to-host synchronization "
        "and would perturb the compiled decode path. The intercept is an estimated "
        "fixed generation cost, not a direct first-token measurement."
    )
    return {
        "audio_sample_count": int(samples.size),
        "audio_seconds": samples.size / sample_rate,
        "targets": targets,
        "runs": rows,
        "cases": cases,
        "regression": regression,
    }


def _shape_trace(engine: Qwen3AsrEngine, utterance: Utterance) -> dict[str, Any]:
    model = engine._model  # noqa: SLF001
    original = model.prepare_inputs_for_generation
    calls = []

    def traced(*args, **kwargs):
        prepared = original(*args, **kwargs)
        calls.append(
            {
                "call": len(calls) + 1,
                "input_shapes": _tensor_shapes(prepared),
                "cache_position_shape": _shape(prepared["cache_position"])
                if prepared.get("cache_position") is not None
                else None,
            }
        )
        return prepared

    model.prepare_inputs_for_generation = traced
    try:
        result, profile = engine.transcribe_profiled(utterance)
    finally:
        model.prepare_inputs_for_generation = original
    return {
        "instrumented": True,
        "performance_comparable": False,
        "reason": "Python shape hook is diagnostic-only and excluded from benchmark summaries",
        **_text_fingerprint(result.text),
        "profile": profile.as_dict(),
        "prepare_inputs_call_count": len(calls),
        "calls": calls,
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--controlled-audio", required=True)
    parser.add_argument("--real-audio", required=True)
    parser.add_argument("--scaling-audio", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--scaling-warmup", type=int, default=2)
    parser.add_argument("--scaling-runs", type=int, default=10)
    parser.add_argument("--token-targets", default="5,10,15,20,30,50")
    args = parser.parse_args()
    if min(args.warmup, args.runs, args.scaling_warmup, args.scaling_runs) < 1:
        raise ValueError("all warmup/run counts must be positive")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    failure_path = output_dir / "benchmark.failure.txt"
    failure_path.unlink(missing_ok=True)

    def save_failure(exception_type, exception, exception_traceback):
        failure_path.write_text(
            "".join(traceback.format_exception(exception_type, exception, exception_traceback)),
            encoding="utf-8",
        )
        sys.__excepthook__(exception_type, exception, exception_traceback)

    sys.excepthook = save_failure
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    controlled, controlled_rate = load_audio(
        args.controlled_audio, config.audio.sample_rate
    )
    real, real_rate = load_audio(args.real_audio, config.audio.sample_rate)
    scaling, scaling_rate = load_audio(args.scaling_audio, config.audio.sample_rate)
    if len({controlled_rate, real_rate, scaling_rate, config.audio.sample_rate}) != 1:
        raise ValueError("all audio inputs must use the configured sample rate")
    controlled_vad, controlled_vad_info = _extract_first_vad_utterance(
        controlled, controlled_rate, config
    )
    real_vad, real_vad_info = _extract_first_vad_utterance(real, real_rate, config)
    _write_wav(output_dir / "controlled_vad_extracted.wav", controlled_vad.samples, controlled_rate)
    _write_wav(output_dir / "real_vad_extracted.wav", real_vad.samples, real_rate)

    engine = create_asr_engine(config)
    if not isinstance(engine, Qwen3AsrEngine):
        raise TypeError("this attribution benchmark requires qwen3_asr")
    load_started = time.perf_counter()
    engine.load()
    load_ms = (time.perf_counter() - load_started) * 1000
    warmup_started = time.perf_counter()
    engine.warmup()
    startup_warmup_ms = (time.perf_counter() - warmup_started) * 1000
    fingerprint = _runtime_fingerprint(engine, config, config_path)
    fingerprint["load_ms"] = load_ms
    fingerprint["startup_warmup_ms"] = startup_warmup_ms
    (output_dir / "runtime_fingerprint.json").write_text(
        json.dumps(fingerprint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    payload = {
        "schema_version": 1,
        "title": "Qwen3-ASR direct vs E2E runtime-gap attribution",
        "metric_definitions": {
            "generated_tokens_per_second": (
                "request-level apparent generation throughput: generated tokens divided "
                "by the entire generate stage, including fixed encoder/prefill costs"
            ),
            "estimated_decode_tokens_per_second": (
                "incremental decode throughput: reciprocal of the token-scaling OLS slope"
            ),
        },
        "generated_at": datetime.now().astimezone().isoformat(),
        "runtime_fingerprint": fingerprint,
        "input_preparation": {
            "controlled_source": str(Path(args.controlled_audio).resolve()),
            "real_source": str(Path(args.real_audio).resolve()),
            "scaling_source": str(Path(args.scaling_audio).resolve()),
            "controlled_vad": controlled_vad_info,
            "real_vad": real_vad_info,
        },
        "shapes": {
            "controlled_full": _processor_shapes(engine, controlled),
            "controlled_vad_extracted": _processor_shapes(engine, controlled_vad.samples),
            "real_vad_extracted": _processor_shapes(engine, real_vad.samples),
            "scaling_full": _processor_shapes(engine, scaling),
        },
    }
    try:
        payload["same_input_ab"] = _same_input_ab(
            engine,
            controlled,
            controlled_vad,
            real_vad,
            controlled_rate,
            args.warmup,
            args.runs,
        )
        payload["profiling_overhead_ab"] = _profiling_overhead_ab(
            engine,
            Utterance(real_vad.samples, real_rate, 0, 0),
            args.warmup,
            args.runs,
        )
        targets = [int(value) for value in args.token_targets.split(",")]
        payload["token_scaling"] = _token_scaling(
            engine,
            scaling,
            scaling_rate,
            targets,
            args.scaling_warmup,
            args.scaling_runs,
        )
        payload["shape_trace"] = _shape_trace(
            engine, Utterance(real_vad.samples, real_rate, 0, 0)
        )
    finally:
        engine.close()

    (output_dir / "benchmark.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_rows(output_dir / "same_input_ab.runs.csv", payload["same_input_ab"]["runs"])
    _write_rows(
        output_dir / "profiling_overhead_ab.runs.csv",
        payload["profiling_overhead_ab"]["rows"],
    )
    _write_rows(output_dir / "token_scaling.runs.csv", payload["token_scaling"]["runs"])
    (output_dir / "shape_trace.json").write_text(
        json.dumps(payload["shape_trace"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
