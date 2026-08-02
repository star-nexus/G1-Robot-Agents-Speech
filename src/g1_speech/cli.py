"""Command line entrypoint for deployment and diagnostics."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
import time
import wave
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .app import SpeechService
from .config import load_config
from .engine import SenseVoiceEngine, find_model_files
from .contracts import Utterance
from .dds import DdsSpeechSubscriber, initialize_dds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline SenseVoice DDS service for robot Agents")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Start microphone, VAD, SenseVoice, and DDS")
    serve.add_argument("--config", required=True)
    serve.add_argument("--run-seconds", type=float, default=0.0, help="0 runs continuously")

    doctor = sub.add_parser("doctor", help="Check models, microphone, runtime, and DDS")
    doctor.add_argument("--config", required=True)
    doctor.add_argument("--load-model", action="store_true")
    doctor.add_argument("--skip-audio", action="store_true", help="Skip microphone checks")

    listen = sub.add_parser("listen", help="Print DDS speech results as JSON")
    listen.add_argument("--config", required=True)
    listen.add_argument("--timeout", type=float, default=0.0, help="0 listens continuously")
    listen.add_argument("--once", action="store_true", help="Exit after the first result")

    transcribe = sub.add_parser("transcribe", help="Transcribe a WAV with the service engine")
    transcribe.add_argument("--config", required=True)
    transcribe.add_argument("wav")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    if args.command == "serve":
        return _serve(args.config, args.run_seconds)
    if args.command == "doctor":
        return _doctor(args.config, args.load_model, args.skip_audio)
    if args.command == "listen":
        return _listen(args.config, args.timeout, args.once)
    if args.command == "transcribe":
        return _transcribe(args.config, args.wav)
    parser.error("unknown command")
    return 2


def _serve(config_path: str, run_seconds: float) -> int:
    config = load_config(config_path)
    service = SpeechService(config)
    stopped = threading.Event()

    def stop(*_args) -> None:
        stopped.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    service.start()
    started = time.monotonic()
    next_metrics = started + config.metrics_interval_seconds
    try:
        while not stopped.wait(0.2):
            now = time.monotonic()
            if run_seconds > 0 and now - started >= run_seconds:
                break
            if now >= next_metrics:
                payload = asdict(service.pipeline.metrics())
                payload.update(
                    dds_outbox_size=service.sink.queue_size,
                    dds_delivered=service.sink.delivered,
                    dds_retries=service.sink.retries,
                    dds_expired=service.sink.expired,
                    dds_dropped=service.sink.dropped,
                )
                logging.getLogger(__name__).info("metrics=%s", json.dumps(payload))
                next_metrics = now + config.metrics_interval_seconds
    finally:
        service.close()
    return 0


def _doctor(config_path: str, load_model: bool, skip_audio: bool = False) -> int:
    config = load_config(config_path)
    checks: list[tuple[str, bool, str]] = []

    def check(name, callback) -> None:
        try:
            detail = callback()
            checks.append((name, True, str(detail or "OK")))
        except Exception as exc:  # noqa: BLE001
            checks.append((name, False, f"{type(exc).__name__}: {exc}"))

    check(
        "SenseVoice model files",
        lambda: find_model_files(
            config.sensevoice.model_dir, config.sensevoice.model_file
        ),
    )
    check("Silero VAD model", lambda: _require_file(config.vad.model))
    check("sherpa_onnx", lambda: _module_path("sherpa_onnx"))
    if not skip_audio:
        check("sounddevice/microphone", _audio_devices)
    check("cyclonedds", lambda: _module_path("cyclonedds"))
    if load_model:
        engine = SenseVoiceEngine(
            model_dir=config.sensevoice.model_dir,
            model_file=config.sensevoice.model_file,
            device=config.sensevoice.device,
            sample_rate=config.audio.sample_rate,
            language=config.sensevoice.language,
            use_itn=config.sensevoice.use_itn,
            num_threads=config.sensevoice.num_threads,
        )
        check("SenseVoice runtime load", lambda: engine.load() or "loaded")

    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return 0 if all(ok for _, ok, _ in checks) else 1


def _listen(config_path: str, timeout: float, once: bool) -> int:
    if timeout < 0:
        raise ValueError("timeout must not be negative")
    config = load_config(config_path)
    initialize_dds(config.dds.domain_id, config.dds.network_interface)
    stopped = threading.Event()

    def on_speech(event) -> None:
        print(json.dumps(asdict(event), ensure_ascii=False), flush=True)
        if once:
            stopped.set()

    def stop(*_args) -> None:
        stopped.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    subscriber = DdsSpeechSubscriber(on_speech, topic=config.dds.speech_topic)
    subscriber.start()
    started = time.monotonic()
    try:
        while not stopped.wait(0.2):
            if timeout > 0 and time.monotonic() - started >= timeout:
                return 1 if once else 0
    finally:
        subscriber.close()
    return 0


def _transcribe(config_path: str, wav_path: str) -> int:
    config = load_config(config_path)
    samples, sample_rate = _read_wav(wav_path)
    now = time.monotonic_ns()
    utterance = Utterance(samples, sample_rate, now, now)
    engine = SenseVoiceEngine(
        model_dir=config.sensevoice.model_dir,
        model_file=config.sensevoice.model_file,
        device=config.sensevoice.device,
        sample_rate=config.audio.sample_rate,
        language=config.sensevoice.language,
        use_itn=config.sensevoice.use_itn,
        num_threads=config.sensevoice.num_threads,
    )
    result = engine.transcribe(utterance)
    print(json.dumps(asdict(result), ensure_ascii=False))
    return 0 if result.text else 1


def _read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(str(Path(path).expanduser()), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError("only mono 16-bit PCM WAV files are supported")
        sample_rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, sample_rate


def _module_path(name: str) -> str:
    module = __import__(name)
    return str(getattr(module, "__file__", "built-in"))


def _require_file(path: str) -> str:
    candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return str(candidate)


def _audio_devices() -> str:
    import sounddevice as sd

    devices = sd.query_devices()
    inputs = [device["name"] for device in devices if device["max_input_channels"] > 0]
    if not inputs:
        raise RuntimeError("no audio input device found")
    return ", ".join(inputs)


if __name__ == "__main__":
    sys.exit(main())
