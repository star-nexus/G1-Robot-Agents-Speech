"""Measure acoustic speech-end -> VAD-ready -> final text for any ASR backend."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

from g1_speech.asr import create_asr_engine
from g1_speech.config import load_config
from g1_speech.contracts import AudioChunk
from g1_speech.vad import SileroVadSegmenter


def read_audio(path: str, sample_rate: int) -> tuple[np.ndarray, int]:
    source = Path(path)
    if source.suffix.lower() == ".wav":
        with wave.open(str(source), "rb") as handle:
            if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
                raise ValueError("only mono 16-bit PCM WAV files are supported")
            rate = handle.getframerate()
            raw = handle.readframes(handle.getnframes())
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, rate
    # Jetson's multimedia ffmpeg plugins may write an EGL diagnostic to stdout
    # even with `-v error`, corrupting a pipe:1 PCM stream. Decode to a temporary
    # file so stdout can never become part of the audio payload.
    with tempfile.TemporaryDirectory(prefix="g1-speech-audio-") as directory:
        decoded = Path(directory) / "decoded.f32"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(source),
                "-f",
                "f32le",
                "-acodec",
                "pcm_f32le",
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "-y",
                str(decoded),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        samples = np.fromfile(decoded, dtype="<f4")
    if not samples.size:
        raise ValueError(f"decoded audio is empty: {source}")
    return samples, sample_rate


def trim_trailing_silence(samples: np.ndarray, threshold: float = 0.008) -> np.ndarray:
    non_silent = np.flatnonzero(np.abs(samples) >= threshold)
    if non_silent.size == 0:
        raise ValueError("no speech was detected in the WAV file")
    return samples[: non_silent[-1] + 1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--target-ms", type=float, default=800.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    samples, rate = read_audio(args.wav, config.audio.sample_rate)
    if rate != config.audio.sample_rate:
        raise ValueError(f"WAV is {rate} Hz; expected {config.audio.sample_rate} Hz")
    samples = trim_trailing_silence(samples)

    vad = SileroVadSegmenter(
        settings=config.vad,
        sample_rate=rate,
    )
    engine = create_asr_engine(config)
    engine.load()  # The production service also warms the model before listening.
    warmup = getattr(engine, "warmup", None)
    if warmup is not None:
        warmup()

    block = round(rate * config.audio.block_ms / 1000)
    utterance = None
    for offset in range(0, samples.size, block):
        frame = samples[offset : offset + block]
        # Model a PortAudio callback: the timestamp is captured when this block
        # has arrived, then VAD consumes it. Stop at the first complete utterance
        # exactly as the production pipeline would.
        time.sleep(frame.size / rate)
        segments = vad.accept(AudioChunk(frame, rate, time.monotonic_ns()))
        if segments:
            utterance = segments[0]
            break

    deadline = time.monotonic() + 3.0
    silence = np.zeros(block, dtype=np.float32)
    while utterance is None and time.monotonic() < deadline:
        time.sleep(block / rate)
        segments = vad.accept(AudioChunk(silence, rate, time.monotonic_ns()))
        if segments:
            utterance = segments[0]
    if utterance is None:
        raise RuntimeError("VAD did not complete the utterance within 3 seconds")

    recognition_started_ns = time.monotonic_ns()
    profile = None
    profiled_transcribe = getattr(engine, "transcribe_profiled", None)
    if config.qwen3_asr.log_profile and profiled_transcribe is not None:
        result, profile = profiled_transcribe(utterance)
    else:
        result = engine.transcribe(utterance)
    final_ns = time.monotonic_ns()
    latency_ms = (final_ns - utterance.ended_monotonic_ns) / 1_000_000
    metrics = {
        "backend": config.asr.backend,
        "text": result.text,
        "audio_duration_ms": utterance.duration_ms,
        "configured_vad_silence_ms": config.vad.min_silence_seconds * 1000,
        "observed_vad_wait_ms": max(
            0, utterance.vad_ready_monotonic_ns - utterance.ended_monotonic_ns
        )
        / 1_000_000,
        "asr_queue_ms": max(
            0, recognition_started_ns - utterance.vad_ready_monotonic_ns
        )
        / 1_000_000,
        "inference_ms": result.inference_ms,
        "speech_end_to_final_ms": latency_ms,
        "rtf": result.inference_ms / max(1, utterance.duration_ms),
    }
    if profile is not None:
        metrics.update(
            {
                "processor_ms": profile.processor_ms,
                "h2d_ms": profile.h2d_ms,
                "generate_ms": profile.generate_ms,
                "decode_ms": profile.decode_ms,
                "generated_tokens": profile.generated_tokens,
                "generated_tokens_per_second": profile.generated_tokens_per_second,
            }
        )
    failure = None
    if not result.text:
        failure = "recognition result is empty"
    elif latency_ms > args.target_ms:
        failure = f"{latency_ms:.1f}ms > {args.target_ms:.1f}ms"
    metrics["passed"] = failure is None
    if failure is not None:
        metrics["failure"] = failure
    if args.json:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    else:
        for name, value in metrics.items():
            print(f"{name}={value!r}" if isinstance(value, str) else f"{name}={value}")
        print("PASS" if failure is None else f"FAIL: {failure}")
    return 0 if failure is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
