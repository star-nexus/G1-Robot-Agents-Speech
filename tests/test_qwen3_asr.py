from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from g1_speech.contracts import Utterance
from g1_speech.qwen3_asr import (
    Qwen3AsrEngine,
    _sdpa_supports_enable_gqa,
    validate_qwen3_asr_model_dir,
)


def _model_dir(tmp_path: Path) -> Path:
    for name in ("config.json", "model.safetensors", "processor_config.json", "tokenizer.json"):
        (tmp_path / name).write_bytes(b"test")
    return tmp_path


class Batch(dict):
    def to(self, device, dtype):
        self.target = (device, dtype)
        return self


class Processor:
    def __init__(self):
        self.request = None

    def apply_transcription_request(self, **request):
        self.request = request
        return Batch(input_ids=np.zeros((1, 2), dtype=np.int64))

    def decode(self, generated, return_format):
        assert return_format == "parsed"
        return [{"language": "Chinese", "transcription": "你好，机器人。"}]


class Model:
    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False

    def generate(self, **inputs):
        assert inputs["do_sample"] is False
        return np.zeros((1, 4), dtype=np.int64)


class Torch:
    float32 = "float32"
    float16 = "float16"
    bfloat16 = "bfloat16"
    cuda = SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None)

    @staticmethod
    def device(name):
        return SimpleNamespace(type=name.split(":", 1)[0], name=name)

    @staticmethod
    def from_numpy(array):
        return array

    @staticmethod
    def inference_mode():
        return nullcontext()

def test_qwen3_engine_uses_in_memory_audio_and_preserves_selected_language(tmp_path, caplog):
    caplog.set_level("INFO", logger="g1_speech.qwen3_asr")
    processor = Processor()
    model = Model()
    model_options = {}

    def load_model(*args, **kwargs):
        model_options.update(kwargs)
        return model

    transformers = SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: processor),
        AutoModelForMultimodalLM=SimpleNamespace(from_pretrained=load_model),
    )
    engine = Qwen3AsrEngine(
        model_dir=_model_dir(tmp_path),
        device="cuda",
        dtype="bfloat16",
        language="zh",
        prompt="Vocabulary: robot model R1",
        attention_implementation="eager",
        torch_module=Torch,
        transformers_module=transformers,
    )
    utterance = Utterance(np.zeros(16000, dtype=np.float32), 16000, 0, 0)

    result = engine.transcribe(utterance)

    assert result.text == "你好，机器人。"
    assert result.language == "zh"
    assert result.engine == "qwen3_asr"
    assert np.shares_memory(processor.request["audio"], utterance.samples)
    assert processor.request["prompt"] == "Vocabulary: robot model R1"
    assert processor.request["language"] == "zh"
    assert model_options["device_map"] == "cuda:0"
    assert model_options["dtype"] == "bfloat16"
    assert model_options["attn_implementation"] == "eager"
    assert model.training is False
    assert "audio=1.00s RTF=" in caplog.text


def test_sdpa_gqa_capability_detection_uses_runtime_documentation():
    legacy = SimpleNamespace(
        nn=SimpleNamespace(
            functional=SimpleNamespace(
                scaled_dot_product_attention=SimpleNamespace(__doc__="scale=None")
            )
        )
    )
    current = SimpleNamespace(
        nn=SimpleNamespace(
            functional=SimpleNamespace(
                scaled_dot_product_attention=SimpleNamespace(__doc__="enable_gqa=False")
            )
        )
    )

    assert not _sdpa_supports_enable_gqa(legacy)
    assert _sdpa_supports_enable_gqa(current)


def test_qwen3_model_validation_reports_missing_artifacts(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        validate_qwen3_asr_model_dir(tmp_path)
