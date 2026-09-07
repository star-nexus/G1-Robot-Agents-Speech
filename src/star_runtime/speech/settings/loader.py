"""JSON loading, migration, path resolution, and host overrides."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ...transports.config import DdsConfig, Ros2Config, TransportConfig
from .models import (
    AsrConfig,
    AudioConfig,
    AudioOutputConfig,
    AudioProcessingConfig,
    PlaybackConfig,
    Qwen3AsrConfig,
    SenseVoiceConfig,
    ServiceConfig,
    TtsConfig,
    VadConfig,
)


def default_config_dict() -> dict[str, Any]:
    """Return the canonical, JSON-serializable service defaults."""
    return asdict(ServiceConfig())


def write_config(
    path: str | Path,
    *,
    base: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
    overrides: Sequence[str] = (),
    overwrite: bool = False,
) -> Path:
    """Generate a validated config from canonical defaults and explicit overrides."""
    target = Path(path).expanduser().resolve()
    if target.exists() and not overwrite:
        raise FileExistsError(f"configuration already exists: {target}")

    if base is None:
        raw = default_config_dict()
    else:
        base_path = Path(base).expanduser().resolve()
        raw = json.loads(base_path.read_text(encoding="utf-8"))

    raw = deepcopy(raw)
    _migrate_legacy_config(raw)
    _fill_missing_section_defaults(raw)
    if environment is not None:
        _apply_environment_overrides(raw, environment)
    for assignment in overrides:
        _apply_assignment(raw, assignment)

    _config_from_raw(deepcopy(raw), target.parent)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(raw, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def _merge_dataclass(cls, raw: dict[str, Any]):
    allowed = cls.__dataclass_fields__.keys()
    unknown = set(raw) - set(allowed)
    if unknown:
        raise ValueError(f"{cls.__name__} contains unknown settings: {sorted(unknown)}")
    return cls(**raw)


def load_config(
    path: str | Path,
    *,
    runtime_environment: Mapping[str, str] | None = None,
) -> ServiceConfig:
    """Load a model profile and optionally apply backend-neutral host overrides.

    Model selection and model-specific tuning always come from the JSON profile.
    The runtime environment is deliberately restricted to settings that apply to
    every ASR backend, such as the microphone, VAD, transport, and playback gate.
    """
    config_path = Path(path).expanduser().resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    _migrate_legacy_config(raw)
    _fill_missing_section_defaults(raw)
    if runtime_environment is not None:
        _apply_environment_overrides(
            raw,
            runtime_environment,
            allowed_names=_RUNTIME_ENV_OVERRIDE_NAMES,
        )
    return _config_from_raw(raw, config_path.parent)


def _fill_missing_section_defaults(raw: dict[str, Any]) -> None:
    """Make intentionally small model profiles safe for section-level overrides."""
    for section, value in default_config_dict().items():
        if not isinstance(value, dict):
            continue
        target_section = raw.setdefault(section, {})
        for field_name, default_value in value.items():
            target_section.setdefault(field_name, deepcopy(default_value))


def _migrate_legacy_config(raw: dict[str, Any]) -> None:
    """Preserve the old PortAudio/Pulse behavior while upgrading audio.device."""
    audio = raw.get("audio")
    if not isinstance(audio, dict) or "device" not in audio:
        return
    if "pulse_device" in audio:
        raise ValueError("audio.device and audio.pulse_device cannot both be set")
    audio["pulse_device"] = audio.pop("device")
    audio.setdefault("input_backend", "pulse")
    audio.setdefault("fallback_backend", None)


def _config_from_raw(raw: dict[str, Any], base: Path) -> ServiceConfig:
    allowed_top = ServiceConfig.__dataclass_fields__.keys()
    unknown_top = set(raw) - set(allowed_top)
    if unknown_top:
        raise ValueError(f"ServiceConfig contains unknown settings: {sorted(unknown_top)}")

    audio = _merge_dataclass(AudioConfig, raw.pop("audio", {}))
    audio_processing = _merge_dataclass(
        AudioProcessingConfig, raw.pop("audio_processing", {})
    )
    audio_output = _merge_dataclass(AudioOutputConfig, raw.pop("audio_output", {}))
    vad_raw = raw.pop("vad", {})
    vad_raw["model"] = str(_resolve_path(base, vad_raw.get("model", VadConfig.model)))
    vad = _merge_dataclass(VadConfig, vad_raw)
    asr = _merge_dataclass(AsrConfig, raw.pop("asr", {}))
    sense_raw = raw.pop("sensevoice", {})
    sense_raw["model_dir"] = str(
        _resolve_path(base, sense_raw.get("model_dir", SenseVoiceConfig.model_dir))
    )
    model_file = sense_raw.get("model_file")
    sense_raw["model_file"] = (
        str(_resolve_path(base, model_file)) if model_file else None
    )
    sensevoice = _merge_dataclass(SenseVoiceConfig, sense_raw)
    qwen_raw = raw.pop("qwen3_asr", {})
    qwen_raw["model_dir"] = str(
        _resolve_path(base, qwen_raw.get("model_dir", Qwen3AsrConfig.model_dir))
    )
    qwen3_asr = _merge_dataclass(Qwen3AsrConfig, qwen_raw)
    transport = _merge_dataclass(TransportConfig, raw.pop("transport", {}))
    dds = _merge_dataclass(DdsConfig, raw.pop("dds", {}))
    ros2 = _merge_dataclass(Ros2Config, raw.pop("ros2", {}))
    playback = _merge_dataclass(PlaybackConfig, raw.pop("playback", {}))
    tts_raw = raw.pop("tts", {})
    reference_audio = tts_raw.get("reference_audio")
    tts_raw["reference_audio"] = (
        str(_resolve_path(base, reference_audio)) if reference_audio else None
    )
    tts = _merge_dataclass(TtsConfig, tts_raw)
    config = ServiceConfig(
        audio=audio,
        audio_processing=audio_processing,
        audio_output=audio_output,
        vad=vad,
        asr=asr,
        sensevoice=sensevoice,
        qwen3_asr=qwen3_asr,
        transport=transport,
        dds=dds,
        ros2=ros2,
        playback=playback,
        tts=tts,
        **raw,
    )
    config.validate()
    return config


def _parse_optional_device(value: str) -> int | str | None:
    value = value.strip()
    if not value or value.lower() == "null":
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _parse_optional_string(value: str) -> str | None:
    value = value.strip()
    return value or None


def _parse_optional_backend(value: str) -> str | None:
    value = value.strip().lower()
    return value or None


def _parse_audio_latency(value: str) -> str | float:
    normalized = value.strip().lower()
    if normalized in {"low", "high"}:
        return normalized
    return float(normalized)


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("expected a boolean value")


_ENV_OVERRIDES: dict[str, tuple[str, str, Callable[[str], Any]]] = {
    "AUDIO_INPUT_BACKEND": ("audio", "input_backend", str),
    "AUDIO_INPUT_FALLBACK": ("audio", "fallback_backend", _parse_optional_backend),
    "ALSA_INPUT_CARD": ("audio", "alsa_card", _parse_optional_string),
    "ALSA_INPUT_DEVICE": ("audio", "alsa_device", int),
    "ALSA_INPUT_SAMPLE_RATE": ("audio", "alsa_sample_rate", int),
    "ALSA_INPUT_CHANNELS": ("audio", "alsa_channels", int),
    "ALSA_INPUT_DTYPE": ("audio", "alsa_dtype", str),
    "PULSE_INPUT_DEVICE": ("audio", "pulse_device", _parse_optional_device),
    "AUDIO_INPUT_BLOCK_MS": ("audio", "block_ms", int),
    "AUDIO_INPUT_LATENCY": ("audio", "latency", _parse_audio_latency),
    "AUDIO_PROCESSING_MODE": ("audio_processing", "mode", str),
    "WEBRTC_AEC_STREAM_DELAY_MS": (
        "audio_processing",
        "stream_delay_ms",
        int,
    ),
    "WEBRTC_AEC_ENABLED": (
        "audio_processing",
        "echo_cancellation",
        _parse_bool,
    ),
    "WEBRTC_NS_ENABLED": ("audio_processing", "noise_suppression", _parse_bool),
    "WEBRTC_NS_LEVEL": ("audio_processing", "noise_suppression_level", int),
    "WEBRTC_AGC_ENABLED": (
        "audio_processing",
        "automatic_gain_control",
        _parse_bool,
    ),
    "ALSA_OUTPUT_CARD": ("audio_output", "alsa_card", _parse_optional_string),
    "ALSA_OUTPUT_DEVICE": ("audio_output", "alsa_device", int),
    "ALSA_OUTPUT_SAMPLE_RATE": ("audio_output", "sample_rate", int),
    "ALSA_OUTPUT_CHANNELS": ("audio_output", "channels", int),
    "AUDIO_OUTPUT_BLOCK_MS": ("audio_output", "block_ms", int),
    "AUDIO_OUTPUT_LATENCY": ("audio_output", "latency", _parse_audio_latency),
    "AUDIO_OUTPUT_VOLUME": ("audio_output", "volume", float),
    "AUDIO_OUTPUT_BUFFER_SECONDS": ("audio_output", "buffer_seconds", float),
    "AUDIO_OUTPUT_INTERRUPT_STRATEGY": (
        "audio_output",
        "interrupt_strategy",
        str,
    ),
    # Compatibility with deploy.env files created before the input-backend split.
    "MICROPHONE_DEVICE": ("audio", "pulse_device", _parse_optional_device),
    "SPEECH_TRANSPORT": ("transport", "backend", str),
    "ROS2_NODE_NAME": ("ros2", "node_name", str),
    "ROS2_SPEECH_TOPIC": ("ros2", "speech_topic", str),
    "ROS2_PLAYBACK_TOPIC": ("ros2", "playback_topic", str),
    "ROS2_TTS_TOPIC": ("ros2", "tts_topic", str),
    "ROS2_CONTROL_TOPIC": ("ros2", "control_topic", str),
    "ROS2_QOS_DEPTH": ("ros2", "qos_depth", int),
    "VAD_THRESHOLD": ("vad", "threshold", float),
    "VAD_PRE_ROLL_SECONDS": ("vad", "speech_pre_roll_seconds", float),
    "VAD_MIN_SILENCE_SECONDS": ("vad", "min_silence_seconds", float),
    "VAD_MIN_SPEECH_SECONDS": ("vad", "min_speech_seconds", float),
    "VAD_MAX_SPEECH_SECONDS": ("vad", "max_speech_seconds", float),
    "DDS_DOMAIN_ID": ("dds", "domain_id", int),
    "DDS_NETWORK_INTERFACE": ("dds", "network_interface", _parse_optional_string),
    "SPEECH_TOPIC": ("dds", "speech_topic", str),
    "PLAYBACK_TOPIC": ("dds", "playback_topic", str),
    "TTS_TOPIC": ("dds", "tts_topic", str),
    "CONTROL_TOPIC": ("dds", "control_topic", str),
    "DDS_DELIVERY_TTL_SECONDS": ("dds", "delivery_ttl_seconds", float),
    "DDS_OUTBOX_CAPACITY": ("dds", "outbox_capacity", int),
    "PLAYBACK_RESUME_DELAY_MS": ("playback", "resume_delay_ms", int),
    "PLAYBACK_MAX_ACTIVE_SECONDS": ("playback", "max_active_seconds", float),
    "TTS_ENABLED": ("tts", "enabled", _parse_bool),
    "TTS_WEBSOCKET_URL": ("tts", "websocket_url", str),
    "TTS_VOICE": ("tts", "voice", str),
    "TTS_LANGUAGE": ("tts", "language", str),
}


# These are deployment/host settings, not model settings.  Keeping this list
# separate prevents deploy.env from silently replacing the backend, checkpoint,
# dtype, attention implementation, or cache policy selected by a JSON profile.
_RUNTIME_ENV_OVERRIDE_NAMES = frozenset(
    {
        "MICROPHONE_DEVICE",
        "AUDIO_INPUT_BACKEND",
        "AUDIO_INPUT_FALLBACK",
        "ALSA_INPUT_CARD",
        "ALSA_INPUT_DEVICE",
        "ALSA_INPUT_SAMPLE_RATE",
        "ALSA_INPUT_CHANNELS",
        "ALSA_INPUT_DTYPE",
        "PULSE_INPUT_DEVICE",
        "AUDIO_INPUT_BLOCK_MS",
        "AUDIO_INPUT_LATENCY",
        "AUDIO_PROCESSING_MODE",
        "WEBRTC_AEC_STREAM_DELAY_MS",
        "WEBRTC_AEC_ENABLED",
        "WEBRTC_NS_ENABLED",
        "WEBRTC_NS_LEVEL",
        "WEBRTC_AGC_ENABLED",
        "ALSA_OUTPUT_CARD",
        "ALSA_OUTPUT_DEVICE",
        "ALSA_OUTPUT_SAMPLE_RATE",
        "ALSA_OUTPUT_CHANNELS",
        "AUDIO_OUTPUT_BLOCK_MS",
        "AUDIO_OUTPUT_LATENCY",
        "AUDIO_OUTPUT_VOLUME",
        "AUDIO_OUTPUT_BUFFER_SECONDS",
        "AUDIO_OUTPUT_INTERRUPT_STRATEGY",
        "SPEECH_TRANSPORT",
        "ROS2_NODE_NAME",
        "ROS2_SPEECH_TOPIC",
        "ROS2_PLAYBACK_TOPIC",
        "ROS2_TTS_TOPIC",
        "ROS2_QOS_DEPTH",
        "VAD_THRESHOLD",
        "VAD_PRE_ROLL_SECONDS",
        "VAD_MIN_SILENCE_SECONDS",
        "VAD_MIN_SPEECH_SECONDS",
        "VAD_MAX_SPEECH_SECONDS",
        "DDS_DOMAIN_ID",
        "DDS_NETWORK_INTERFACE",
        "SPEECH_TOPIC",
        "PLAYBACK_TOPIC",
        "TTS_TOPIC",
        "DDS_DELIVERY_TTL_SECONDS",
        "DDS_OUTBOX_CAPACITY",
        "PLAYBACK_RESUME_DELAY_MS",
        "PLAYBACK_MAX_ACTIVE_SECONDS",
        "TTS_ENABLED",
        "TTS_WEBSOCKET_URL",
        "TTS_VOICE",
        "TTS_LANGUAGE",
    }
)


def _apply_environment_overrides(
    raw: dict[str, Any],
    environment: Mapping[str, str],
    *,
    allowed_names: frozenset[str] | None = None,
) -> None:
    if (
        "MICROPHONE_DEVICE" in environment
        and "AUDIO_INPUT_BACKEND" not in environment
        and (allowed_names is None or "MICROPHONE_DEVICE" in allowed_names)
    ):
        raw["audio"]["input_backend"] = "pulse"
        raw["audio"]["fallback_backend"] = None
    for name, (section, field_name, parser) in _ENV_OVERRIDES.items():
        if allowed_names is not None and name not in allowed_names:
            continue
        if name not in environment:
            continue
        try:
            value = parser(environment[name])
        except ValueError as exc:
            raise ValueError(f"invalid {name}: {environment[name]!r}") from exc
        raw[section][field_name] = value


def _apply_assignment(raw: dict[str, Any], assignment: str) -> None:
    key, separator, encoded = assignment.partition("=")
    if not separator or not key:
        raise ValueError(f"override must use section.field=value: {assignment!r}")
    parts = key.split(".")
    if len(parts) != 2:
        raise ValueError(f"override must target section.field: {key!r}")
    section, field_name = parts
    if section not in raw or not isinstance(raw[section], dict):
        raise ValueError(f"unknown configuration section: {section!r}")
    if field_name not in raw[section]:
        raise ValueError(f"unknown configuration field: {key!r}")
    try:
        value = json.loads(encoded)
    except json.JSONDecodeError:
        value = encoded
    raw[section][field_name] = value


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()
