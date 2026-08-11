"""JSON configuration with path resolution relative to the config file."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    block_ms: int = 100
    device: int | str | None = None
    queue_seconds: float = 5.0
    heartbeat_timeout_seconds: float = 2.0
    reconnect_initial_seconds: float = 0.5
    reconnect_max_seconds: float = 10.0


@dataclass(frozen=True)
class VadConfig:
    model: str = "models/silero_vad.onnx"
    threshold: float = 0.35
    speech_pre_roll_seconds: float = 0.5
    min_silence_seconds: float = 0.35
    min_speech_seconds: float = 0.15
    max_speech_seconds: float = 10.0
    buffer_seconds: float = 30.0


@dataclass(frozen=True)
class SenseVoiceConfig:
    model_dir: str = "models"
    model_file: str | None = None
    device: str = "cpu"
    language: str = "zh"
    use_itn: bool = True
    num_threads: int = 4


@dataclass(frozen=True)
class AsrConfig:
    """Select an ASR adapter without coupling the pipeline to its runtime."""

    backend: str = "sensevoice"


@dataclass(frozen=True)
class Qwen3AsrConfig:
    model_dir: str = "models/Qwen3-ASR-0.6B-hf"
    device: str = "auto"
    dtype: str = "auto"
    language: str | None = "zh"
    prompt: str | None = None
    max_new_tokens: int = 256
    attention_implementation: str | None = "sdpa"
    compile: bool = False
    compile_mode: str = "reduce-overhead"
    compile_dynamic: bool = False
    startup_warmup_seconds: float = 0.0
    cache_implementation: str | None = None
    quantization: str | None = None
    log_profile: bool = False


@dataclass(frozen=True)
class DdsConfig:
    domain_id: int = 0
    network_interface: str | None = None
    speech_topic: str = "rt/g1/hri/speech/final"
    playback_topic: str = "rt/g1/hri/playback/state"
    write_timeout_seconds: float = 0.5
    retry_interval_seconds: float = 0.2
    delivery_ttl_seconds: float = 120.0
    outbox_capacity: int = 128


@dataclass(frozen=True)
class Ros2Config:
    node_name: str = "g1_speech"
    speech_topic: str = "hri/speech/final"
    playback_topic: str = "hri/playback/state"
    qos_depth: int = 10


@dataclass(frozen=True)
class TransportConfig:
    backend: str = "dds"


@dataclass(frozen=True)
class PlaybackConfig:
    resume_delay_ms: int = 250
    max_active_seconds: float = 30.0


@dataclass(frozen=True)
class ServiceConfig:
    source_name: str = "g1_speech_mic"
    utterance_queue_capacity: int = 4
    metrics_interval_seconds: float = 60.0
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    sensevoice: SenseVoiceConfig = field(default_factory=SenseVoiceConfig)
    qwen3_asr: Qwen3AsrConfig = field(default_factory=Qwen3AsrConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    dds: DdsConfig = field(default_factory=DdsConfig)
    ros2: Ros2Config = field(default_factory=Ros2Config)
    playback: PlaybackConfig = field(default_factory=PlaybackConfig)

    def validate(self) -> None:
        if self.audio.sample_rate != 16000:
            raise ValueError("The bundled ASR backends and Silero VAD require 16000 Hz audio")
        if self.audio.block_ms <= 0 or self.audio.block_ms > 500:
            raise ValueError("audio.block_ms must be between 1 and 500")
        if self.audio.queue_seconds <= 0:
            raise ValueError("audio.queue_seconds must be greater than zero")
        if self.audio.heartbeat_timeout_seconds <= 0:
            raise ValueError("audio.heartbeat_timeout_seconds must be greater than zero")
        if self.audio.reconnect_initial_seconds <= 0:
            raise ValueError("audio.reconnect_initial_seconds must be greater than zero")
        if self.audio.reconnect_max_seconds < self.audio.reconnect_initial_seconds:
            raise ValueError(
                "audio.reconnect_max_seconds must be at least reconnect_initial_seconds"
            )
        if self.utterance_queue_capacity < 1:
            raise ValueError("utterance_queue_capacity must be greater than zero")
        if not 0 < self.vad.threshold < 1:
            raise ValueError("vad.threshold must be between 0 and 1")
        if self.vad.min_silence_seconds <= 0:
            raise ValueError("vad.min_silence_seconds must be greater than zero")
        if self.vad.min_speech_seconds <= 0:
            raise ValueError("vad.min_speech_seconds must be greater than zero")
        if not 0 <= self.vad.speech_pre_roll_seconds <= 1:
            raise ValueError("vad.speech_pre_roll_seconds must be between 0 and 1")
        if self.sensevoice.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("sensevoice.device must be cpu, cuda, or auto")
        if not self.asr.backend.strip():
            raise ValueError("asr.backend must not be empty")
        if self.qwen3_asr.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("qwen3_asr.device must be cpu, cuda, or auto")
        if self.qwen3_asr.dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError(
                "qwen3_asr.dtype must be auto, float32, float16, or bfloat16"
            )
        if self.qwen3_asr.attention_implementation not in {
            None,
            "eager",
            "sdpa",
            "fa2",
            "flash_attention_2",
        }:
            raise ValueError(
                "qwen3_asr.attention_implementation must be eager, sdpa, "
                "fa2, flash_attention_2, or null"
            )
        if self.qwen3_asr.max_new_tokens < 1:
            raise ValueError("qwen3_asr.max_new_tokens must be greater than zero")
        if not self.qwen3_asr.compile_mode.strip():
            raise ValueError("qwen3_asr.compile_mode must not be empty")
        if self.qwen3_asr.compile_dynamic and self.qwen3_asr.cache_implementation not in {
            "static",
            "offloaded_static",
        }:
            raise ValueError(
                "qwen3_asr.compile_dynamic requires a static cache implementation"
            )
        if not 0.0 <= self.qwen3_asr.startup_warmup_seconds <= 30.0:
            raise ValueError(
                "qwen3_asr.startup_warmup_seconds must be between 0 and 30"
            )
        if self.qwen3_asr.quantization not in {None, "bnb_nf4"}:
            raise ValueError("qwen3_asr.quantization must be bnb_nf4 or null")
        if self.vad.max_speech_seconds <= self.vad.min_speech_seconds:
            raise ValueError("vad.max_speech_seconds must exceed min_speech_seconds")
        if self.vad.buffer_seconds <= 0:
            raise ValueError("vad.buffer_seconds must be greater than zero")
        if self.dds.outbox_capacity < 1:
            raise ValueError("dds.outbox_capacity must be greater than zero")
        if self.transport.backend not in {"dds", "ros2"}:
            raise ValueError("transport.backend must be dds or ros2")
        if not self.ros2.node_name:
            raise ValueError("ros2.node_name must not be empty")
        if not self.ros2.speech_topic or not self.ros2.playback_topic:
            raise ValueError("ROS 2 topic names must not be empty")
        if self.ros2.qos_depth < 1:
            raise ValueError("ros2.qos_depth must be greater than zero")


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


def _config_from_raw(raw: dict[str, Any], base: Path) -> ServiceConfig:
    allowed_top = ServiceConfig.__dataclass_fields__.keys()
    unknown_top = set(raw) - set(allowed_top)
    if unknown_top:
        raise ValueError(f"ServiceConfig contains unknown settings: {sorted(unknown_top)}")

    audio = _merge_dataclass(AudioConfig, raw.pop("audio", {}))
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
    config = ServiceConfig(
        audio=audio,
        vad=vad,
        asr=asr,
        sensevoice=sensevoice,
        qwen3_asr=qwen3_asr,
        transport=transport,
        dds=dds,
        ros2=ros2,
        playback=playback,
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


_ENV_OVERRIDES: dict[str, tuple[str, str, Callable[[str], Any]]] = {
    "MICROPHONE_DEVICE": ("audio", "device", _parse_optional_device),
    "SPEECH_TRANSPORT": ("transport", "backend", str),
    "ROS2_NODE_NAME": ("ros2", "node_name", str),
    "ROS2_SPEECH_TOPIC": ("ros2", "speech_topic", str),
    "ROS2_PLAYBACK_TOPIC": ("ros2", "playback_topic", str),
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
    "DDS_DELIVERY_TTL_SECONDS": ("dds", "delivery_ttl_seconds", float),
    "DDS_OUTBOX_CAPACITY": ("dds", "outbox_capacity", int),
    "PLAYBACK_RESUME_DELAY_MS": ("playback", "resume_delay_ms", int),
    "PLAYBACK_MAX_ACTIVE_SECONDS": ("playback", "max_active_seconds", float),
}


# These are deployment/host settings, not model settings.  Keeping this list
# separate prevents deploy.env from silently replacing the backend, checkpoint,
# dtype, attention implementation, or cache policy selected by a JSON profile.
_RUNTIME_ENV_OVERRIDE_NAMES = frozenset(
    {
        "MICROPHONE_DEVICE",
        "SPEECH_TRANSPORT",
        "ROS2_NODE_NAME",
        "ROS2_SPEECH_TOPIC",
        "ROS2_PLAYBACK_TOPIC",
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
        "DDS_DELIVERY_TTL_SECONDS",
        "DDS_OUTBOX_CAPACITY",
        "PLAYBACK_RESUME_DELAY_MS",
        "PLAYBACK_MAX_ACTIVE_SECONDS",
    }
)


def _apply_environment_overrides(
    raw: dict[str, Any],
    environment: Mapping[str, str],
    *,
    allowed_names: frozenset[str] | None = None,
) -> None:
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
