from __future__ import annotations

import json
from pathlib import Path

import pytest

from g1_speech.config import default_config_dict, load_config, write_config


def test_config_example_matches_canonical_defaults():
    example = Path(__file__).parents[1] / "config.example.json"

    assert json.loads(example.read_text(encoding="utf-8")) == default_config_dict()


def test_write_config_applies_only_explicit_environment_overrides(tmp_path):
    config_path = tmp_path / "config.json"

    write_config(
        config_path,
        environment={"VAD_THRESHOLD": "0.42", "MICROPHONE_DEVICE": "7"},
    )
    config = load_config(config_path)

    assert config.vad.threshold == 0.42
    assert config.vad.speech_pre_roll_seconds == default_config_dict()["vad"][
        "speech_pre_roll_seconds"
    ]
    assert config.audio.device == 7


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


def test_unknown_config_key_is_rejected(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"online_provider": "forbidden"}', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown settings"):
        load_config(path)
