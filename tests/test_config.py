from __future__ import annotations

import json

import pytest

from g1_speech.config import load_config


def test_vad_defaults_are_tuned_for_far_field_speech(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")

    config = load_config(config_path)

    assert config.vad.threshold == 0.35
    assert config.vad.speech_pre_roll_seconds == 0.5
    assert config.vad.min_silence_seconds == 0.35
    assert config.vad.min_speech_seconds == 0.15
    assert config.vad.max_speech_seconds == 10.0


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
