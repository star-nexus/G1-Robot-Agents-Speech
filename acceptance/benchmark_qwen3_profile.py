"""Run synchronized Qwen3-ASR latency decomposition and length scans."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

from g1_speech.asr import create_asr_engine
from g1_speech.config import load_config
from g1_speech.contracts import Utterance
from g1_speech.qwen3_asr import Qwen3AsrEngine

from qwen3_benchmarking import load_audio, summarize_profiles, write_benchmark_outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--audio", action="append", required=True)
    parser.add_argument(
        "--factors",
        help="Comma-separated repetition factors; requires exactly one --audio",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--long-runs", type=int, help="Runs for the longest case")
    parser.add_argument("--record-ranges", action="store_true")
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="Bracket measured requests with cudaProfilerStart/Stop for Nsight capture",
    )
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--title", default="Qwen3-ASR profiling benchmark")
    args = parser.parse_args()
    failure_path = Path(args.output_prefix).with_suffix(".failure.txt")
    failure_path.unlink(missing_ok=True)

    def save_failure(exception_type, exception, exception_traceback):
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(
            "".join(
                traceback.format_exception(
                    exception_type, exception, exception_traceback
                )
            ),
            encoding="utf-8",
        )
        sys.__excepthook__(exception_type, exception, exception_traceback)

    sys.excepthook = save_failure
    if args.warmup < 0 or args.runs < 1:
        raise ValueError("warmup must be >= 0 and runs must be >= 1")
    if args.cuda_profiler_range and len(args.audio) != 1:
        raise ValueError("--cuda-profiler-range requires exactly one --audio")

    config = load_config(args.config)
    engine = create_asr_engine(config)
    if not isinstance(engine, Qwen3AsrEngine):
        raise TypeError("this benchmark requires the qwen3_asr backend")

    cases: list[tuple[str, np.ndarray, int]] = []
    if args.factors:
        if len(args.audio) != 1:
            raise ValueError("--factors requires exactly one --audio")
        samples, sample_rate = load_audio(args.audio[0], config.audio.sample_rate)
        for factor in (int(value) for value in args.factors.split(",")):
            if factor < 1:
                raise ValueError("factors must be positive")
            cases.append((f"{factor}x", np.tile(samples, factor), sample_rate))
    else:
        for audio_path in args.audio:
            samples, sample_rate = load_audio(audio_path, config.audio.sample_rate)
            cases.append((Path(audio_path).stem, samples, sample_rate))

    started = time.perf_counter()
    engine.load()
    load_ms = (time.perf_counter() - started) * 1000
    output_cases = []
    try:
        for case_index, (name, samples, sample_rate) in enumerate(cases):
            if sample_rate != config.audio.sample_rate:
                raise ValueError(
                    f"{name} is {sample_rate} Hz; "
                    f"expected {config.audio.sample_rate} Hz"
                )
            utterance = Utterance(samples, sample_rate, 0, 0)
            warmups = []
            for _ in range(args.warmup):
                warmup_result, warmup_profile = engine.transcribe_profiled(utterance)
                warmups.append(
                    {"text": warmup_result.text, **warmup_profile.as_dict()}
                )
            run_count = (
                args.long_runs
                if args.long_runs is not None and case_index == len(cases) - 1
                else args.runs
            )
            measurements = []
            texts = []
            profiler_started = False
            try:
                if args.cuda_profiler_range:
                    import torch

                    torch.cuda.cudart().cudaProfilerStart()
                    profiler_started = True
                for _ in range(run_count):
                    result, profile = engine.transcribe_profiled(
                        utterance, record_ranges=args.record_ranges
                    )
                    measurements.append(profile.as_dict())
                    texts.append(result.text)
            finally:
                if profiler_started:
                    torch.cuda.cudart().cudaProfilerStop()
            if len(set(texts)) != 1:
                raise RuntimeError(f"non-deterministic transcription in {name}: {texts}")
            output_cases.append(
                {
                    "name": name,
                    "audio_seconds": len(samples) / sample_rate,
                    "warmup": args.warmup,
                    "runs": run_count,
                    "first_request_ms": warmups[0]["total_ms"] if warmups else None,
                    "warmup_measurements": warmups,
                    "text": texts[0],
                    "summary": summarize_profiles(measurements),
                    "measurements": measurements,
                }
            )
    finally:
        engine.close()

    import torch

    payload = {
        "schema_version": 1,
        "title": args.title,
        "generated_at": datetime.now().astimezone().isoformat(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "config": {
            "backend": config.asr.backend,
            "model": Path(config.qwen3_asr.model_dir).name,
            "device": config.qwen3_asr.device,
            "dtype": config.qwen3_asr.dtype,
            "attention_implementation": config.qwen3_asr.attention_implementation,
            "compile": config.qwen3_asr.compile,
            "compile_mode": config.qwen3_asr.compile_mode,
            "cache_implementation": config.qwen3_asr.cache_implementation,
            "quantization": config.qwen3_asr.quantization,
            "max_new_tokens": config.qwen3_asr.max_new_tokens,
        },
        "load_ms": load_ms,
        "cases": output_cases,
    }
    write_benchmark_outputs(payload, args.output_prefix)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
