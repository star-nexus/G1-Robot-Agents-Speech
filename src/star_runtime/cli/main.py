"""Command line entrypoint for deployment and diagnostics."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
import wave
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np

from star_runtime.agent import load_role_package
from star_runtime.apps.local_voice_agent import (
    DEFAULT_REASONING_BUDGET_MESSAGE,
    DEFAULT_SYSTEM_PROMPT,
    LocalAgentSettings,
    load_role_prompt,
    run_local_voice_agent,
)
from star_runtime.apps.integrated_runtime import (
    build_integrated_runtime,
    run_integrated_runtime,
)
from star_runtime.speech.asr.factory import asr_diagnostic_checks, create_asr_engine
from star_runtime.speech.audio.input import resolve_alsa_input_device
from star_runtime.speech.config import AudioConfig, load_config, write_config
from star_runtime.speech.contracts import Utterance
from star_runtime.perception.vision import (
    LatestFrameCamera,
    LlamaVisionClassifier,
    LlamaVisionClassifierSettings,
    RoutedVisionInput,
    VisionRouteRecorder,
    VisionRouter,
)
from star_runtime.apps.speech_runtime import build_speech_runtime
from star_runtime.transports.dds.voice import (
    DdsSpeechSubscriber,
    DdsTtsPublisher,
    initialize_dds,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pluggable offline ASR service for robot Agents"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    config = sub.add_parser("config", help="Create a validated service configuration")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    config_init = config_sub.add_parser("init", help="Generate a service configuration")
    config_init.add_argument("--output", required=True)
    config_init.add_argument("--base", help="Start from an existing JSON configuration")
    config_init.add_argument(
        "--from-env",
        action="store_true",
        help="Apply supported deployment environment variables",
    )
    config_init.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.FIELD=VALUE",
        help="Apply an explicit JSON or string override; may be repeated",
    )
    config_init.add_argument("--force", action="store_true", help="Overwrite output")

    serve = sub.add_parser("serve", help="Start microphone, VAD, ASR, and transport")
    serve.add_argument("--config", required=True)
    serve.add_argument("--transport", choices=("dds", "ros2"))
    serve.add_argument("--run-seconds", type=float, default=0.0, help="0 runs continuously")

    doctor = sub.add_parser(
        "doctor", help="Check models, microphone, and selected transport runtime"
    )
    doctor.add_argument("--config", required=True)
    doctor.add_argument("--transport", choices=("dds", "ros2"))
    doctor.add_argument("--load-model", action="store_true")
    doctor.add_argument("--skip-audio", action="store_true", help="Skip microphone checks")

    listen = sub.add_parser("listen", help="Print DDS speech results as JSON")
    listen.add_argument("--config", required=True)
    listen.add_argument("--timeout", type=float, default=0.0, help="0 listens continuously")
    listen.add_argument("--once", action="store_true", help="Exit after the first result")

    transcribe = sub.add_parser("transcribe", help="Transcribe a WAV with the service engine")
    transcribe.add_argument("--config", required=True)
    transcribe.add_argument("wav")

    speak = sub.add_parser("speak", help="Publish one complete TTS request over DDS")
    speak.add_argument("--config", required=True)
    speak.add_argument("--language", default="")
    speak.add_argument("--voice", default="")
    speak.add_argument("--instructions", default="")
    speak.add_argument("--request-id")
    speak.add_argument("text")

    agent = sub.add_parser(
        "agent",
        help="Bridge speech through local llama.cpp into streaming TTS",
    )
    _add_agent_arguments(agent)

    runtime = sub.add_parser(
        "runtime",
        help="Run microphone → ASR → Agent → TTS in one process",
    )
    _add_agent_arguments(runtime)
    return parser


def _add_agent_arguments(parser: argparse.ArgumentParser) -> None:
    """Keep standalone and integrated Agent options identical."""

    agent = parser
    agent.add_argument("--config", required=True)
    agent.add_argument(
        "--robot-adapter",
        help="Explicit star-robot-* adapter ID; required when several bodies match",
    )
    agent.add_argument(
        "--url",
        default="http://127.0.0.1:8080/v1/chat/completions",
    )
    agent.add_argument("--model", default="local-qwen3-4b")
    persona = agent.add_mutually_exclusive_group()
    persona.add_argument("--system-prompt")
    persona.add_argument(
        "--role-file",
        help="Legacy UTF-8 role prompt file",
    )
    persona.add_argument(
        "--role-package",
        help="Versioned Role Package directory or role.json manifest",
    )
    agent.add_argument("--max-tokens", type=int)
    agent.add_argument("--temperature", type=float)
    agent.add_argument("--history-turns", type=int)
    agent.add_argument(
        "--thinking",
        dest="enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable Qwen3 reasoning; reasoning text is not sent to TTS",
    )
    agent.add_argument(
        "--thinking-budget-tokens",
        type=int,
        default=None,
        help="Maximum reasoning tokens when thinking is enabled (-1 is unlimited)",
    )
    agent.add_argument(
        "--thinking-budget-message",
        default=None,
        help="Instruction inserted when llama.cpp exhausts the thinking budget",
    )
    agent.add_argument("--request-timeout", type=float, default=120.0)
    agent.add_argument("--max-speech-age", type=float, default=5.0)
    agent.add_argument(
        "--tts-voice",
        default=None,
        help="Override the TTS preset voice for this Agent character",
    )
    agent.add_argument(
        "--tts-language",
        default=None,
        help="Override the TTS language for this Agent character",
    )
    agent.add_argument(
        "--tts-instructions",
        default=None,
        help="Optional TTS delivery/style instructions",
    )
    agent.add_argument(
        "--vision",
        choices=("off", "auto", "always"),
        default="off",
        help="Attach camera frames never, only when routed, or on every turn",
    )
    agent.add_argument("--camera-device", default="/dev/video0")
    agent.add_argument("--camera-width", type=int, default=640)
    agent.add_argument("--camera-height", type=int, default=480)
    agent.add_argument("--camera-fps", type=int, default=5)
    agent.add_argument("--camera-start-timeout", type=float, default=5.0)
    agent.add_argument("--camera-frame-max-age", type=float, default=2.0)
    agent.add_argument(
        "--vision-route-log",
        help="Optional private JSONL path for visual routing diagnostics",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    if args.command == "config":
        return _config_init(
            args.output,
            args.base,
            args.from_env,
            args.overrides,
            args.force,
        )
    if args.command == "serve":
        return _serve(args.config, args.run_seconds, args.transport)
    if args.command == "doctor":
        return _doctor(args.config, args.load_model, args.skip_audio, args.transport)
    if args.command == "listen":
        return _listen(args.config, args.timeout, args.once)
    if args.command == "transcribe":
        return _transcribe(args.config, args.wav)
    if args.command == "speak":
        return _speak(
            args.config,
            args.text,
            args.request_id,
            args.language,
            args.voice,
            args.instructions,
        )
    if args.command in {"agent", "runtime"}:
        config, role_package, settings = _load_agent_configuration(args)
        vision, camera = _build_vision_input(args)
        if args.command == "runtime":
            runtime = build_integrated_runtime(
                config,
                settings,
                role_package=role_package,
                robot_adapter_id=args.robot_adapter,
                vision=vision,
            )
            if camera is None:
                return run_integrated_runtime(runtime)
            try:
                camera.start(args.camera_start_timeout)
                return run_integrated_runtime(runtime)
            finally:
                runtime.close()
                camera.close()
        try:
            if camera is not None:
                camera.start(args.camera_start_timeout)
            return run_local_voice_agent(
                config,
                settings,
                role_package=role_package,
                robot_adapter_id=args.robot_adapter,
                vision=vision,
            )
        finally:
            if camera is not None:
                camera.close()
    parser.error("unknown command")
    return 2


def _load_agent_configuration(args):
    """Resolve role-package defaults once for both Runtime launch modes."""

    config = load_config(args.config, runtime_environment=os.environ)
    role_package = load_role_package(args.role_package) if args.role_package else None
    if role_package:
        system_prompt = role_package.prompt
        role_model = role_package.model
        role_voice = role_package.voice
        role_history = (
            role_package.memory.max_turns
            if role_package.memory.provider == "window"
            else 0
        )
    else:
        system_prompt = (
            load_role_prompt(args.role_file)
            if args.role_file
            else args.system_prompt or DEFAULT_SYSTEM_PROMPT
        )
        role_model = None
        role_voice = None
        role_history = 4
    settings = LocalAgentSettings(
        url=args.url,
        model=args.model,
        system_prompt=system_prompt,
        max_tokens=(
            args.max_tokens
            if args.max_tokens is not None
            else role_model.max_tokens if role_model else 64
        ),
        temperature=(
            args.temperature
            if args.temperature is not None
            else role_model.temperature if role_model else 0.2
        ),
        history_turns=(
            args.history_turns if args.history_turns is not None else role_history
        ),
        enable_thinking=(
            args.enable_thinking
            if args.enable_thinking is not None
            else role_model.thinking if role_model else False
        ),
        reasoning_budget_tokens=(
            args.thinking_budget_tokens
            if args.thinking_budget_tokens is not None
            else role_model.thinking_budget_tokens if role_model else -1
        ),
        reasoning_budget_message=(
            args.thinking_budget_message
            if args.thinking_budget_message is not None
            else DEFAULT_REASONING_BUDGET_MESSAGE
        ),
        request_timeout_seconds=args.request_timeout,
        max_speech_age_seconds=args.max_speech_age,
        tts_voice=(
            args.tts_voice
            if args.tts_voice is not None
            else role_voice.voice if role_voice else ""
        ),
        tts_language=(
            args.tts_language
            if args.tts_language is not None
            else role_voice.language if role_voice else ""
        ),
        tts_instructions=(
            args.tts_instructions
            if args.tts_instructions is not None
            else role_voice.instructions if role_voice else ""
        ),
    )
    return config, role_package, settings


def _build_vision_input(args):
    """Create visual ports only when selected so text deployments stay lightweight."""

    if args.vision == "off":
        return None, None
    camera = LatestFrameCamera(
        device=args.camera_device,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
    )
    classifier = None
    if args.vision == "auto":
        classifier = LlamaVisionClassifier(
            LlamaVisionClassifierSettings(
                url=args.url,
                model=args.model,
                timeout_seconds=args.request_timeout,
            )
        )
    router = VisionRouter(mode=args.vision, classifier=classifier)
    recorder = (
        VisionRouteRecorder(args.vision_route_log)
        if args.vision_route_log
        else None
    )
    return (
        RoutedVisionInput(
            router,
            camera,
            max_frame_age_seconds=args.camera_frame_max_age,
            recorder=recorder,
        ),
        camera,
    )


def _config_init(
    output: str,
    base: str | None,
    from_env: bool,
    overrides: list[str],
    force: bool,
) -> int:
    path = write_config(
        output,
        base=base,
        environment=os.environ if from_env else None,
        overrides=overrides,
        overwrite=force,
    )
    print(path)
    return 0


def _serve(config_path: str, run_seconds: float, transport: str | None = None) -> int:
    config = load_config(config_path, runtime_environment=os.environ)
    if (transport or config.transport.backend) == "inprocess":
        raise ValueError("use the runtime command for the in-process backend")
    service = build_speech_runtime(config, transport_backend=transport)
    stopped = threading.Event()

    def stop(*_args) -> None:
        stopped.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    started = time.monotonic()
    next_metrics = started + config.metrics_interval_seconds
    try:
        service.start()
        while not stopped.wait(0.2):
            now = time.monotonic()
            if run_seconds > 0 and now - started >= run_seconds:
                break
            if now >= next_metrics:
                payload = service.metrics()
                logging.getLogger(__name__).info("metrics=%s", json.dumps(payload))
                next_metrics = now + config.metrics_interval_seconds
    finally:
        service.close()
    return 0


def _doctor(
    config_path: str,
    load_model: bool,
    skip_audio: bool = False,
    transport: str | None = None,
) -> int:
    config = load_config(config_path, runtime_environment=os.environ)
    selected_transport = transport or config.transport.backend
    checks: list[tuple[str, bool, str]] = []

    def check(name, callback) -> None:
        try:
            detail = callback()
            checks.append((name, True, str(detail or "OK")))
        except Exception as exc:  # noqa: BLE001
            checks.append((name, False, f"{type(exc).__name__}: {exc}"))

    for name, callback in asr_diagnostic_checks(config):
        check(name, callback)
    check("Silero VAD model", lambda: _require_file(config.vad.model))
    if not skip_audio:
        check("audio input policy", lambda: _audio_input_policy(config.audio))
    if selected_transport == "inprocess":
        checks.append(("transport", True, "in-process callbacks; no middleware"))
    elif selected_transport == "dds":
        check("cyclonedds", lambda: _module_path("cyclonedds"))
    else:
        check("rclpy", lambda: _module_path("rclpy"))
        check("g1_speech_msgs", lambda: _module_path("g1_speech_msgs"))
    if load_model:
        try:
            engine = create_asr_engine(config)
        except Exception as exc:  # noqa: BLE001
            checks.append(("ASR adapter creation", False, f"{type(exc).__name__}: {exc}"))
        else:
            check(
                f"{config.asr.backend} runtime load",
                lambda: engine.load() or "loaded",
            )

    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return 0 if all(ok for _, ok, _ in checks) else 1


def _listen(config_path: str, timeout: float, once: bool) -> int:
    if timeout < 0:
        raise ValueError("timeout must not be negative")
    config = load_config(config_path, runtime_environment=os.environ)
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
    config = load_config(config_path, runtime_environment=os.environ)
    samples, sample_rate = _read_wav(wav_path)
    now = time.monotonic_ns()
    utterance = Utterance(samples, sample_rate, now, now)
    engine = create_asr_engine(config)
    try:
        result = engine.transcribe(utterance)
        print(json.dumps(asdict(result), ensure_ascii=False))
        return 0 if result.text else 1
    finally:
        engine.close()


def _speak(
    config_path: str,
    text: str,
    request_id: str | None,
    language: str,
    voice: str,
    instructions: str,
) -> int:
    config = load_config(config_path, runtime_environment=os.environ)
    initialize_dds(config.dds.domain_id, config.dds.network_interface)
    publisher = DdsTtsPublisher(topic=config.dds.tts_topic, source="g1-speech-cli")
    publisher.start()
    try:
        delivered = publisher.publish(
            request_id=request_id or uuid.uuid4().hex,
            sequence=0,
            text=text,
            is_final=True,
            language=language,
            voice=voice,
            instructions=instructions,
            timeout=1.0,
        )
        return 0 if delivered else 1
    finally:
        publisher.close()


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


def _audio_input_policy(settings: AudioConfig) -> str:
    import sounddevice as sd

    if settings.input_backend == "alsa":
        try:
            index, name = resolve_alsa_input_device(
                sd,
                card_id=settings.alsa_card,
                pcm_device=settings.alsa_device,
            )
            sd.check_input_settings(
                device=index,
                samplerate=settings.alsa_sample_rate,
                channels=settings.alsa_channels,
                dtype=settings.alsa_dtype,
            )
        except Exception as exc:  # noqa: BLE001
            if settings.fallback_backend != "pulse":
                raise
            pulse = sd.query_devices(settings.pulse_device, "input")
            sd.check_input_settings(
                device=settings.pulse_device,
                samplerate=settings.sample_rate,
                channels=1,
                dtype="float32",
            )
            return (
                f"ALSA unavailable ({exc}); PulseAudio fallback available: "
                f"{pulse['name']}"
            )
        return f"ALSA direct: {name}"
    pulse = sd.query_devices(settings.pulse_device, "input")
    sd.check_input_settings(
        device=settings.pulse_device,
        samplerate=settings.sample_rate,
        channels=1,
        dtype="float32",
    )
    return f"PulseAudio: {pulse['name']}"


if __name__ == "__main__":
    sys.exit(main())
