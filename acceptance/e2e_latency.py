"""Measure speech-end -> final text latency with real VAD and SenseVoice."""

from __future__ import annotations

import argparse
import time
import wave
from pathlib import Path

import numpy as np

from g1_speech.config import load_config
from g1_speech.contracts import AudioChunk
from g1_speech.engine import SenseVoiceEngine
from g1_speech.vad import SileroVadSegmenter


def read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(str(Path(path)), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError("只支持 mono 16-bit PCM WAV")
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, rate


def trim_trailing_silence(samples: np.ndarray, threshold: float = 0.008) -> np.ndarray:
    non_silent = np.flatnonzero(np.abs(samples) >= threshold)
    if non_silent.size == 0:
        raise ValueError("WAV 没有检测到语音")
    return samples[: non_silent[-1] + 1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--target-ms", type=float, default=800.0)
    args = parser.parse_args()
    config = load_config(args.config)
    samples, rate = read_wav(args.wav)
    if rate != config.audio.sample_rate:
        raise ValueError(f"WAV 是 {rate} Hz，要求 {config.audio.sample_rate} Hz")
    samples = trim_trailing_silence(samples)

    vad = SileroVadSegmenter(
        model=config.vad.model,
        sample_rate=rate,
        threshold=config.vad.threshold,
        speech_pre_roll_seconds=config.vad.speech_pre_roll_seconds,
        min_silence_seconds=config.vad.min_silence_seconds,
        min_speech_seconds=config.vad.min_speech_seconds,
        max_speech_seconds=config.vad.max_speech_seconds,
        buffer_seconds=config.vad.buffer_seconds,
    )
    engine = SenseVoiceEngine(
        model_dir=config.sensevoice.model_dir,
        model_file=config.sensevoice.model_file,
        device=config.sensevoice.device,
        sample_rate=rate,
        language=config.sensevoice.language,
        use_itn=config.sensevoice.use_itn,
        num_threads=config.sensevoice.num_threads,
    )
    engine.load()  # The production service also warms the model before listening.

    block = round(rate * config.audio.block_ms / 1000)
    period = block / rate
    for offset in range(0, samples.size, block):
        frame = samples[offset : offset + block]
        vad.accept(AudioChunk(frame, rate, time.monotonic_ns()))
        time.sleep(period)
    speech_ended = time.monotonic()

    utterance = None
    deadline = speech_ended + 3.0
    silence = np.zeros(block, dtype=np.float32)
    while utterance is None and time.monotonic() < deadline:
        time.sleep(period)
        segments = vad.accept(AudioChunk(silence, rate, time.monotonic_ns()))
        if segments:
            utterance = segments[0]
    if utterance is None:
        raise RuntimeError("VAD 在 3 秒内没有结束当前语句")

    result = engine.transcribe(utterance)
    latency_ms = (time.monotonic() - speech_ended) * 1000
    print(f"text={result.text!r}")
    print(f"inference_ms={result.inference_ms:.1f}")
    print(f"speech_end_to_final_ms={latency_ms:.1f}")
    if not result.text:
        print("FAIL: 识别结果为空")
        return 1
    if latency_ms > args.target_ms:
        print(f"FAIL: {latency_ms:.1f}ms > {args.target_ms:.1f}ms")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
