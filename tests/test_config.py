from __future__ import annotations

import json

import pytest

from g1_speech.config import load_config


def test_paths_are_relative_to_config_file(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "vad": {"model": "assets/vad.onnx"},
                "sensevoice": {"model_dir": "assets/sensevoice"},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.vad.model == str((tmp_path / "assets/vad.onnx").resolve())
    assert config.sensevoice.model_dir == str((tmp_path / "assets/sensevoice").resolve())


def test_unknown_config_key_is_rejected(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"online_provider": "forbidden"}', encoding="utf-8")
    with pytest.raises(ValueError, match="未知配置项"):
        load_config(path)
