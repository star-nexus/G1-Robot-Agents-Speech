from pathlib import Path

import pytest

from g1_speech.engine import find_model_files


def test_find_model_files_accepts_parent_directory(tmp_path: Path):
    model_dir = tmp_path / "sherpa-onnx-sense-voice-test"
    model_dir.mkdir()
    model = model_dir / "model.int8.onnx"
    tokens = model_dir / "tokens.txt"
    model.write_bytes(b"model")
    tokens.write_text("tokens", encoding="utf-8")

    assert find_model_files(tmp_path) == (model, tokens)


def test_find_model_files_rejects_incomplete_model(tmp_path: Path):
    (tmp_path / "model.int8.onnx").write_bytes(b"model")
    with pytest.raises(FileNotFoundError):
        find_model_files(tmp_path)
