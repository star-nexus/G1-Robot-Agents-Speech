"""Benchmark one configured SenseVoice backend with a fixed WAV."""

from __future__ import annotations

import argparse
import json
import statistics
import time

from g1_speech.cli import _read_wav
from g1_speech.config import load_config
from g1_speech.contracts import Utterance
from g1_speech.engine import SenseVoiceEngine


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
    args = parser.parse_args()
    if args.warmup < 0 or args.runs < 1:
        raise ValueError("warmup 必须 >= 0，runs 必须 >= 1")

    config = load_config(args.config)
    samples, sample_rate = _read_wav(args.wav)
    if sample_rate != config.audio.sample_rate:
        raise ValueError(f"WAV 是 {sample_rate} Hz，要求 {config.audio.sample_rate} Hz")

    utterance = Utterance(samples, sample_rate, 0, 0)
    engine = SenseVoiceEngine(
        model_dir=config.sensevoice.model_dir,
        model_file=config.sensevoice.model_file,
        device=config.sensevoice.device,
        sample_rate=sample_rate,
        language=config.sensevoice.language,
        use_itn=config.sensevoice.use_itn,
        num_threads=config.sensevoice.num_threads,
    )

    started = time.perf_counter()
    engine.load()
    load_ms = (time.perf_counter() - started) * 1000
    for _ in range(args.warmup):
        engine.transcribe(utterance)

    times = [engine.transcribe(utterance).inference_ms for _ in range(args.runs)]
    audio_ms = len(samples) / sample_rate * 1000
    payload = {
        "device": config.sensevoice.device,
        "model_file": config.sensevoice.model_file,
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
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
