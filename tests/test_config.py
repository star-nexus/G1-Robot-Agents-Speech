from __future__ import annotations

import json
from pathlib import Path

import pytest

from g1_speech.config import (
    AudioConfig,
    Qwen3AsrConfig,
    ServiceConfig,
    default_config_dict,
    load_config,
    write_config,
)


def test_config_example_matches_canonical_defaults():
    example = Path(__file__).parents[1] / "config.example.json"

    assert json.loads(example.read_text(encoding="utf-8")) == default_config_dict()


def test_write_config_applies_only_explicit_environment_overrides(tmp_path):
    config_path = tmp_path / "config.json"

    write_config(
        config_path,
        environment={
            "VAD_THRESHOLD": "0.42",
            "AUDIO_INPUT_BACKEND": "pulse",
            "PULSE_INPUT_DEVICE": "7",
            "AUDIO_INPUT_BLOCK_MS": "20",
            "AUDIO_INPUT_LATENCY": "0.015",
        },
    )
    config = load_config(config_path)

    assert config.vad.threshold == 0.42
    assert config.vad.speech_pre_roll_seconds == default_config_dict()["vad"][
        "speech_pre_roll_seconds"
    ]
    assert config.audio.input_backend == "pulse"
    assert config.audio.pulse_device == 7
    assert config.audio.block_ms == 20
    assert config.audio.latency == 0.015


def test_write_config_derives_gpu_config_without_copying_defaults(tmp_path):
    cpu_path = write_config(tmp_path / "config.json")
    gpu_path = write_config(
        tmp_path / "config.gpu.json",
        base=cpu_path,
        overrides=("sensevoice.device=cuda", "sensevoice.num_threads=1"),
    )

    cpu = load_config(cpu_path)
    gpu = load_config(gpu_path)

    assert gpu.sensevoice.device == "cuda"
    assert gpu.sensevoice.num_threads == 1
    assert gpu.vad == cpu.vad


def test_write_config_upgrades_an_older_base_with_new_default_fields(tmp_path):
    base = tmp_path / "old.json"
    base.write_text(
        json.dumps({"qwen3_asr": {"model_dir": "/models/qwen3-asr"}}),
        encoding="utf-8",
    )

    output = write_config(
        tmp_path / "new.json",
        base=base,
        overrides=("qwen3_asr.compile=true",),
    )

    assert load_config(output).qwen3_asr.compile is True
    assert load_config(output).qwen3_asr.attention_implementation == "sdpa"


def test_paths_are_relative_to_config_file(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "vad": {"model": "assets/vad.onnx"},
                "sensevoice": {
                    "model_dir": "assets/sensevoice",
                    "model_file": "assets/sensevoice/model.onnx",
                },
                "qwen3_asr": {"model_dir": "assets/qwen3-asr"},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.vad.model == str((tmp_path / "assets/vad.onnx").resolve())
    assert config.sensevoice.model_dir == str((tmp_path / "assets/sensevoice").resolve())
    assert config.sensevoice.model_file == str(
        (tmp_path / "assets/sensevoice/model.onnx").resolve()
    )
    assert config.qwen3_asr.model_dir == str((tmp_path / "assets/qwen3-asr").resolve())


def test_model_selection_is_not_accepted_from_deployment_environment(tmp_path):
    path = write_config(
        tmp_path / "config.json",
        environment={
            "ASR_BACKEND": "qwen3_asr",
            "QWEN3_ASR_MODEL_DIR": "/models/qwen3-asr",
            "QWEN3_ASR_DTYPE": "bfloat16",
            "QWEN3_ASR_QUANTIZATION": "bnb_nf4",
            "QWEN3_ASR_LOG_PROFILE": "1",
        },
    )

    config = load_config(path)
    assert config.asr.backend == "sensevoice"
    assert config.qwen3_asr.model_dir.endswith("models/Qwen3-ASR-0.6B-hf")
    assert config.qwen3_asr.dtype == "auto"
    assert config.qwen3_asr.quantization is None
    assert config.qwen3_asr.log_profile is False


def test_runtime_environment_only_overrides_backend_neutral_settings(tmp_path):
    path = tmp_path / "qwen.json"
    path.write_text(
        json.dumps(
            {
                "asr": {"backend": "qwen3_asr"},
                "qwen3_asr": {
                    "model_dir": "/models/selected-qwen",
                    "dtype": "float16",
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_config(
        path,
        runtime_environment={
            "ASR_BACKEND": "sensevoice",
            "QWEN3_ASR_MODEL_DIR": "/models/should-not-win",
            "QWEN3_ASR_DTYPE": "bfloat16",
            "AUDIO_INPUT_BACKEND": "alsa",
            "ALSA_INPUT_CARD": "Microphone",
            "ALSA_INPUT_DEVICE": "2",
            "ALSA_INPUT_SAMPLE_RATE": "48000",
            "ALSA_INPUT_CHANNELS": "2",
            "ALSA_INPUT_DTYPE": "int16",
            "PULSE_INPUT_DEVICE": "pulse",
            "VAD_THRESHOLD": "0.42",
            "DDS_DOMAIN_ID": "7",
            "PLAYBACK_RESUME_DELAY_MS": "175",
        },
    )

    assert config.asr.backend == "qwen3_asr"
    assert config.qwen3_asr.model_dir == "/models/selected-qwen"
    assert config.qwen3_asr.dtype == "float16"
    assert config.audio.input_backend == "alsa"
    assert config.audio.alsa_card == "Microphone"
    assert config.audio.alsa_device == 2
    assert config.audio.alsa_sample_rate == 48000
    assert config.audio.alsa_channels == 2
    assert config.audio.alsa_dtype == "int16"
    assert config.audio.pulse_device == "pulse"
    assert config.vad.threshold == 0.42
    assert config.dds.domain_id == 7
    assert config.playback.resume_delay_ms == 175


def test_qwen3_asr_defaults_to_sdpa_attention():
    assert ServiceConfig().qwen3_asr.attention_implementation == "sdpa"
    assert ServiceConfig().qwen3_asr.compile is False
    assert ServiceConfig().qwen3_asr.compile_dynamic is False
    assert ServiceConfig().qwen3_asr.startup_warmup_seconds == 0.0
    assert ServiceConfig().qwen3_asr.cache_implementation is None
    assert ServiceConfig().qwen3_asr.quantization is None
    assert ServiceConfig().qwen3_asr.log_profile is False


def test_legacy_audio_device_is_migrated_to_pulse_selector(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text('{"audio": {"device": "pulse"}}', encoding="utf-8")

    config = load_config(path)

    assert config.audio.pulse_device == "pulse"
    assert config.audio.input_backend == "pulse"
    assert config.audio.fallback_backend is None


def test_legacy_microphone_environment_keeps_pulse_behavior(tmp_path):
    path = write_config(tmp_path / "config.json")

    config = load_config(
        path,
        runtime_environment={"MICROPHONE_DEVICE": "USB Microphone"},
    )

    assert config.audio.input_backend == "pulse"
    assert config.audio.fallback_backend is None
    assert config.audio.pulse_device == "USB Microphone"


def test_numeric_alsa_card_index_is_rejected():
    config = ServiceConfig(audio=AudioConfig(alsa_card="1"))

    with pytest.raises(ValueError, match="stable ALSA card ID"):
        config.validate()


def test_qwen3_asr_rejects_unknown_attention_backend():
    config = ServiceConfig(
        qwen3_asr=Qwen3AsrConfig(attention_implementation="magic")
    )
    with pytest.raises(ValueError, match="attention_implementation"):
        config.validate()


def test_qwen3_asr_rejects_unknown_quantization():
    config = ServiceConfig(qwen3_asr=Qwen3AsrConfig(quantization="awq-ish"))
    with pytest.raises(ValueError, match="quantization"):
        config.validate()


def test_qwen3_asr_dynamic_compile_requires_static_cache():
    config = ServiceConfig(qwen3_asr=Qwen3AsrConfig(compile_dynamic=True))
    with pytest.raises(ValueError, match="static cache"):
        config.validate()


def test_qwen3_asr_startup_warmup_is_bounded():
    config = ServiceConfig(
        qwen3_asr=Qwen3AsrConfig(startup_warmup_seconds=31.0)
    )
    with pytest.raises(ValueError, match="startup_warmup_seconds"):
        config.validate()


def test_unknown_config_key_is_rejected(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"online_provider": "forbidden"}', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown settings"):
        load_config(path)


def test_transport_defaults_to_dds_and_ros2_has_independent_topics(tmp_path):
    path = write_config(tmp_path / "config.json")
    config = load_config(path)

    assert config.transport.backend == "dds"
    assert config.dds.speech_topic == "rt/g1/hri/speech/final"
    assert config.ros2.speech_topic == "hri/speech/final"


def test_transport_backend_can_be_selected_from_environment(tmp_path):
    path = write_config(
        tmp_path / "config.json",
        environment={"SPEECH_TRANSPORT": "ros2"},
    )

    assert load_config(path).transport.backend == "ros2"


def test_invalid_transport_backend_is_rejected(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"transport": {"backend": "mqtt"}}', encoding="utf-8")

    with pytest.raises(ValueError, match="transport.backend"):
        load_config(path)
