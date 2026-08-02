from pathlib import Path

import pytest

from g1_speech.engine import ensure_provider_available, find_model_files


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


def test_explicit_model_file_overrides_int8_preference(tmp_path: Path):
    model_dir = tmp_path / "sensevoice"
    model_dir.mkdir()
    (model_dir / "model.int8.onnx").write_bytes(b"int8")
    fp32_model = model_dir / "model.onnx"
    fp32_model.write_bytes(b"fp32")
    tokens = model_dir / "tokens.txt"
    tokens.write_text("tokens", encoding="utf-8")

    assert find_model_files(tmp_path, fp32_model) == (fp32_model, tokens)


def test_cuda_rejects_cpu_only_sherpa_wheel():
    class CpuSherpa:
        __version__ = "1.13.4"

    with pytest.raises(RuntimeError, match="refusing to silently fall back"):
        ensure_provider_available(CpuSherpa, "cuda")


def test_cuda_accepts_cuda_sherpa_wheel():
    class CudaSherpa:
        __version__ = "1.13.4+cuda"

    ensure_provider_available(CudaSherpa, "cuda")
