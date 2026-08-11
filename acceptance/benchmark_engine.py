"""Benchmark one configured ASR backend with a fixed WAV."""

from __future__ import annotations

import argparse
import json
import statistics
import time

from g1_speech.asr import create_asr_engine
from g1_speech.cli import _read_wav
from g1_speech.config import load_config
from g1_speech.contracts import Utterance
from g1_speech.evaluation import error_rate, normalize_characters


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--reference", help="Human reference text for CER")
    args = parser.parse_args()
    if args.warmup < 0 or args.runs < 1:
        raise ValueError("warmup must be >= 0 and runs must be >= 1")

    config = load_config(args.config)
    samples, sample_rate = _read_wav(args.wav)
    if sample_rate != config.audio.sample_rate:
        raise ValueError(f"WAV is {sample_rate} Hz; expected {config.audio.sample_rate} Hz")

    utterance = Utterance(samples, sample_rate, 0, 0)
    engine = create_asr_engine(config)

    try:
        started = time.perf_counter()
        engine.load()
        load_ms = (time.perf_counter() - started) * 1000
        for _ in range(args.warmup):
            engine.transcribe(utterance)

        results = [engine.transcribe(utterance) for _ in range(args.runs)]
        times = [result.inference_ms for result in results]
        audio_ms = len(samples) / sample_rate * 1000
        payload = {
            "backend": config.asr.backend,
            "engine": results[0].engine,
            "device": (
                config.qwen3_asr.device
                if config.asr.backend in {"qwen3_asr", "qwen3-asr"}
                else config.sensevoice.device
            ),
            "attention": (
                config.qwen3_asr.attention_implementation
                if config.asr.backend in {"qwen3_asr", "qwen3-asr"}
                else None
            ),
            "dtype": (
                config.qwen3_asr.dtype
                if config.asr.backend in {"qwen3_asr", "qwen3-asr"}
                else None
            ),
            "text": results[0].text,
            "runs": args.runs,
            "audio_ms": round(audio_ms, 1),
            "load_ms": round(load_ms, 1),
            "mean_ms": round(statistics.fmean(times), 1),
            "median_ms": round(statistics.median(times), 1),
            "p95_ms": round(percentile(times, 0.95), 1),
            "min_ms": round(min(times), 1),
            "max_ms": round(max(times), 1),
            "rtf": round(statistics.fmean(times) / audio_ms, 4),
        }
        if args.reference is not None:
            payload["reference"] = args.reference
            cer = error_rate(
                normalize_characters(args.reference),
                normalize_characters(results[0].text),
            )
            payload["cer"] = round(cer, 4) if cer is not None else None
    finally:
        engine.close()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
