"""Validated speech and inference configuration value objects."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...transports.config import DdsConfig, Ros2Config, TransportConfig


@dataclass(frozen=True)
class AudioConfig:
    input_backend: str = "alsa"
    fallback_backend: str | None = None
    alsa_card: str | None = None
    alsa_device: int = 0
    alsa_sample_rate: int = 48000
    alsa_channels: int = 2
    alsa_dtype: str = "int16"
    pulse_device: int | str | None = "pulse"
    sample_rate: int = 16000
    block_ms: int = 20
    latency: str | float = "low"
    queue_seconds: float = 5.0
    heartbeat_timeout_seconds: float = 2.0
    reconnect_initial_seconds: float = 0.5
    reconnect_max_seconds: float = 10.0


@dataclass(frozen=True)
class AudioProcessingConfig:
    """Microphone enhancement policy independent of the ASR backend."""

    mode: str = "off"
    sample_rate: int = 16000
    frame_ms: int = 10
    echo_cancellation: bool = True
    noise_suppression: bool = True
    automatic_gain_control: bool = False
    stream_delay_ms: int = 80
    noise_suppression_level: int = 2


@dataclass(frozen=True)
class AudioOutputConfig:
    """Low-latency ALSA output selected by stable card ID."""

    backend: str = "alsa"
    alsa_card: str | None = None
    alsa_device: int = 0
    sample_rate: int = 48000
    channels: int = 2
    dtype: str = "int16"
    block_ms: int = 10
    latency: str | float = "low"
    volume: float = 1.0
    buffer_seconds: float = 8.0
    interrupt_strategy: str = "persistent"


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
class PlaybackConfig:
    resume_delay_ms: int = 250
    max_active_seconds: float = 30.0


@dataclass(frozen=True)
class TtsConfig:
    """Streaming TTS client and sentence scheduler configuration."""

    enabled: bool = False
    backend: str = "vllm_omni"
    websocket_url: str = "ws://127.0.0.1:8091/v1/audio/speech/stream"
    model: str = "Qwen3-TTS-12Hz-0.6B-CustomVoice"
    task_type: str = "CustomVoice"
    voice: str = "Ryan"
    language: str = "Auto"
    instructions: str = ""
    reference_audio: str | None = None
    reference_text: str | None = None
    max_new_tokens: int = 256
    initial_codec_chunk_frames: int | None = None
    connect_timeout_seconds: float = 5.0
    receive_timeout_seconds: float = 0.1
    request_queue_capacity: int = 16
    min_chunk_speech_units: float = 8.0
    preferred_chunk_speech_units: float = 14.0
    max_chunk_speech_units: float = 24.0
    max_buffer_characters: int = 240
    style_tokens: dict[str, str] = field(
        default_factory=lambda: {
            "happy": "Speak in a bright, delighted and energetic tone.",
            "sad": "Speak in a subdued, sad and gentle tone.",
            "angry": "Speak firmly with controlled anger.",
            "calm": "Speak calmly, clearly and reassuringly.",
        }
    )


@dataclass(frozen=True)
class ServiceConfig:
    source_name: str = "g1_speech_mic"
    utterance_queue_capacity: int = 4
    metrics_interval_seconds: float = 60.0
    audio: AudioConfig = field(default_factory=AudioConfig)
    audio_processing: AudioProcessingConfig = field(default_factory=AudioProcessingConfig)
    audio_output: AudioOutputConfig = field(default_factory=AudioOutputConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    sensevoice: SenseVoiceConfig = field(default_factory=SenseVoiceConfig)
    qwen3_asr: Qwen3AsrConfig = field(default_factory=Qwen3AsrConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    dds: DdsConfig = field(default_factory=DdsConfig)
    ros2: Ros2Config = field(default_factory=Ros2Config)
    playback: PlaybackConfig = field(default_factory=PlaybackConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)

    def validate(self) -> None:
        if self.audio.input_backend not in {"alsa", "pulse"}:
            raise ValueError("audio.input_backend must be alsa or pulse")
        if self.audio.fallback_backend not in {None, "pulse"}:
            raise ValueError("audio.fallback_backend must be pulse or null")
        if self.audio.alsa_card is not None:
            if not self.audio.alsa_card.strip():
                raise ValueError(
                    "audio.alsa_card must be null or a non-empty ALSA card ID"
                )
            if self.audio.alsa_card.isdecimal():
                raise ValueError(
                    "audio.alsa_card must be a stable ALSA card ID, not a numeric card index"
                )
        if self.audio.alsa_device < 0:
            raise ValueError("audio.alsa_device must not be negative")
        if self.audio.alsa_sample_rate < self.audio.sample_rate:
            raise ValueError(
                "audio.alsa_sample_rate must be at least audio.sample_rate"
            )
        if self.audio.alsa_sample_rate % self.audio.sample_rate:
            raise ValueError(
                "audio.alsa_sample_rate must be an integer multiple of audio.sample_rate"
            )
        if self.audio.alsa_channels not in {1, 2}:
            raise ValueError("audio.alsa_channels must be 1 or 2")
        if self.audio.alsa_dtype != "int16":
            raise ValueError("audio.alsa_dtype currently supports only int16")
        if self.audio.sample_rate != 16000:
            raise ValueError("The bundled ASR backends and Silero VAD require 16000 Hz audio")
        if self.audio.block_ms <= 0 or self.audio.block_ms > 500:
            raise ValueError("audio.block_ms must be between 1 and 500")
        if self.audio.latency not in {"low", "high"}:
            if (
                not isinstance(self.audio.latency, (int, float))
                or self.audio.latency <= 0
            ):
                raise ValueError("audio.latency must be low, high, or positive seconds")
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
        if self.audio_processing.mode not in {"off", "hardware", "webrtc"}:
            raise ValueError("audio_processing.mode must be off, hardware, or webrtc")
        if self.audio_processing.mode == "webrtc":
            if self.audio_processing.sample_rate != self.audio.sample_rate:
                raise ValueError(
                    "audio_processing.sample_rate must match the ASR pipeline sample rate"
                )
            if self.audio_processing.frame_ms != 10:
                raise ValueError("WebRTC APM requires audio_processing.frame_ms=10")
            frame_samples = (
                self.audio_processing.sample_rate
                * self.audio_processing.frame_ms
                // 1000
            )
            if self.audio.block_ms * self.audio.sample_rate // 1000 % frame_samples:
                raise ValueError("audio block size must contain complete WebRTC APM frames")
        if not 0 <= self.audio_processing.stream_delay_ms <= 500:
            raise ValueError("audio_processing.stream_delay_ms must be between 0 and 500")
        if self.audio_processing.noise_suppression_level not in {0, 1, 2, 3}:
            raise ValueError("audio_processing.noise_suppression_level must be 0..3")
        if self.audio_output.backend != "alsa":
            raise ValueError("audio_output.backend currently supports only alsa")
        if self.audio_output.alsa_card is not None:
            if not self.audio_output.alsa_card.strip():
                raise ValueError("audio_output.alsa_card must be null or non-empty")
            if self.audio_output.alsa_card.isdecimal():
                raise ValueError("audio_output.alsa_card must be a stable ID, not an index")
        if self.audio_output.sample_rate not in {16000, 24000, 32000, 44100, 48000}:
            raise ValueError("audio_output.sample_rate is not a supported PCM rate")
        if self.audio_output.channels not in {1, 2}:
            raise ValueError("audio_output.channels must be 1 or 2")
        if self.audio_output.dtype != "int16":
            raise ValueError("audio_output.dtype currently supports only int16")
        if self.audio_output.block_ms not in {10, 20}:
            raise ValueError("audio_output.block_ms must be 10 or 20")
        if (
            self.audio_processing.mode == "webrtc"
            and self.audio_output.interrupt_strategy == "persistent"
            and self.audio_output.block_ms != 10
        ):
            raise ValueError(
                "persistent WebRTC AEC requires audio_output.block_ms=10 "
                "for one continuous speaker/render clock"
            )
        if not 0 < self.audio_output.volume <= 1:
            raise ValueError("audio_output.volume must be greater than 0 and at most 1")
        if not 0.1 <= self.audio_output.buffer_seconds <= 60:
            raise ValueError("audio_output.buffer_seconds must be between 0.1 and 60")
        if self.audio_output.interrupt_strategy not in {"persistent", "hard_abort"}:
            raise ValueError(
                "audio_output.interrupt_strategy must be persistent or hard_abort"
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
        if self.transport.backend not in {"inprocess", "dds", "ros2"}:
            raise ValueError("transport.backend must be inprocess, dds, or ros2")
        if not self.ros2.node_name:
            raise ValueError("ros2.node_name must not be empty")
        if not all(
            (
                self.ros2.speech_topic,
                self.ros2.playback_topic,
                self.ros2.tts_topic,
                self.ros2.control_topic,
            )
        ):
            raise ValueError("ROS 2 topic names must not be empty")
        if self.ros2.qos_depth < 1:
            raise ValueError("ros2.qos_depth must be greater than zero")
        if self.tts.backend != "vllm_omni":
            raise ValueError("tts.backend currently supports only vllm_omni")
        if self.tts.task_type not in {"CustomVoice", "VoiceDesign", "Base"}:
            raise ValueError("tts.task_type must be CustomVoice, VoiceDesign, or Base")
        if not self.tts.websocket_url.startswith(("ws://", "wss://")):
            raise ValueError("tts.websocket_url must use ws:// or wss://")
        if self.tts.task_type == "Base" and self.tts.enabled:
            if not self.tts.reference_audio:
                raise ValueError("Base TTS requires tts.reference_audio")
            if not self.tts.reference_text:
                raise ValueError("Base TTS requires tts.reference_text")
        if self.tts.task_type != "Base" and (
            self.tts.reference_audio or self.tts.reference_text
        ):
            raise ValueError(
                "reference audio/text are only valid with a Qwen3-TTS Base model"
            )
        if self.tts.max_new_tokens < 1:
            raise ValueError("tts.max_new_tokens must be greater than zero")
        if self.tts.request_queue_capacity < 1:
            raise ValueError("tts.request_queue_capacity must be greater than zero")
        chunk_units = (
            self.tts.min_chunk_speech_units,
            self.tts.preferred_chunk_speech_units,
            self.tts.max_chunk_speech_units,
        )
        if not 0 < chunk_units[0] <= chunk_units[1] <= chunk_units[2]:
            raise ValueError(
                "tts chunk speech units must satisfy 0 < min <= preferred <= max"
            )
        if self.tts.max_buffer_characters < 8:
            raise ValueError("tts.max_buffer_characters must be at least 8")
