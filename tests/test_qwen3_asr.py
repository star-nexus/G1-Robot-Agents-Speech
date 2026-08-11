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
    validate_flash_attention_2_environment,
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
        self.inputs = None

    def eval(self):
        self.training = False

    def forward(self, **inputs):
        return inputs

    def generate(self, **inputs):
        self.inputs = inputs
        assert inputs["do_sample"] is False
        return np.zeros((1, 4), dtype=np.int64)


class Torch:
    float32 = "float32"
    float16 = "float16"
    bfloat16 = "bfloat16"
    cuda = SimpleNamespace(
        is_available=lambda: True,
        empty_cache=lambda: None,
        get_device_capability=lambda device: (8, 7),
    )

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


def test_bnb_nf4_quantizes_only_the_language_decoder(tmp_path):
    processor = Processor()
    model = Model()
    model_options = {}

    class QuantizationConfig:
        def __init__(self, **kwargs):
            self.options = kwargs

    def load_model(*args, **kwargs):
        model_options.update(kwargs)
        return model

    transformers = SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: processor),
        AutoModelForMultimodalLM=SimpleNamespace(from_pretrained=load_model),
        BitsAndBytesConfig=QuantizationConfig,
    )
    engine = Qwen3AsrEngine(
        model_dir=_model_dir(tmp_path),
        device="cuda",
        dtype="float16",
        quantization="bnb_nf4",
        torch_module=Torch,
        transformers_module=transformers,
    )

    engine.load()

    options = model_options["quantization_config"].options
    assert options["load_in_4bit"] is True
    assert options["bnb_4bit_compute_dtype"] == "float16"
    assert options["bnb_4bit_quant_type"] == "nf4"
    assert options["llm_int8_skip_modules"] == [
        "model.audio_tower",
        "model.multi_modal_projector",
        "lm_head",
    ]


def test_profiled_transcription_keeps_result_api_and_reports_all_stages(tmp_path):
    processor = Processor()
    model = Model()
    transformers = SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: processor),
        AutoModelForMultimodalLM=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: model
        ),
    )
    engine = Qwen3AsrEngine(
        model_dir=_model_dir(tmp_path),
        device="cuda",
        dtype="bfloat16",
        cache_implementation="static",
        torch_module=Torch,
        transformers_module=transformers,
    )

    result, profile = engine.transcribe_profiled(
        Utterance(np.zeros(16000, dtype=np.float32), 16000, 0, 0)
    )

    assert result.text == "你好，机器人。"
    assert result.inference_ms == profile.total_ms
    assert profile.audio_seconds == 1.0
    assert profile.generated_tokens == 2
    assert profile.rtf == profile.total_ms / 1000
    assert set(profile.as_dict()) == {
        "audio_seconds",
        "processor_ms",
        "h2d_ms",
        "generate_ms",
        "decode_ms",
        "total_ms",
        "generated_tokens",
        "generated_tokens_per_second",
        "rtf",
    }
    assert model.inputs["cache_implementation"] == "static"


def test_live_profile_logging_is_opt_in(tmp_path, caplog):
    caplog.set_level("INFO", logger="g1_speech.qwen3_asr")
    processor = Processor()
    model = Model()
    transformers = SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: processor),
        AutoModelForMultimodalLM=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: model
        ),
    )
    engine = Qwen3AsrEngine(
        model_dir=_model_dir(tmp_path),
        device="cuda",
        dtype="float16",
        log_profile=True,
        torch_module=Torch,
        transformers_module=transformers,
    )

    engine.transcribe(Utterance(np.zeros(16000, dtype=np.float32), 16000, 0, 0))

    assert "Qwen3-ASR profile tokens=2 generate=" in caplog.text
    assert "tokens_per_second=" in caplog.text


def test_torch_compile_is_an_explicit_load_option(tmp_path):
    processor = Processor()
    model = Model()
    compiled = SimpleNamespace(model=None, mode=None)
    events = []

    class CompileTorch(Torch):
        compiler = SimpleNamespace(
            cudagraph_mark_step_begin=lambda: events.append("step")
        )

        @staticmethod
        def compile(selected_model, *, mode):
            compiled.model = selected_model
            compiled.mode = mode

            def compiled_forward(**inputs):
                events.append("forward")
                return selected_model(**inputs)

            return compiled_forward

    transformers = SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: processor),
        AutoModelForMultimodalLM=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: model
        ),
    )
    engine = Qwen3AsrEngine(
        model_dir=_model_dir(tmp_path),
        compile_model=True,
        compile_mode="reduce-overhead",
        torch_module=CompileTorch,
        transformers_module=transformers,
    )

    engine.load()
    assert engine._model.forward(value=1) == {"value": 1}

    assert compiled.model.__self__ is model
    assert compiled.mode == "reduce-overhead"
    assert events == ["step", "forward"]


def test_torch_compile_prefers_autoregressive_language_model(tmp_path):
    processor = Processor()
    decoder = Model()
    outer = Model()
    outer.model = SimpleNamespace(language_model=decoder)
    compiled = SimpleNamespace(model=None)

    class CompileTorch(Torch):
        @staticmethod
        def compile(selected_model, *, mode):
            compiled.model = selected_model
            return selected_model

    transformers = SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: processor),
        AutoModelForMultimodalLM=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: outer
        ),
    )
    engine = Qwen3AsrEngine(
        model_dir=_model_dir(tmp_path),
        compile_model=True,
        torch_module=CompileTorch,
        transformers_module=transformers,
    )

    engine.load()

    assert compiled.model.__self__ is decoder


def test_dynamic_compile_config_is_forwarded_to_generate(tmp_path):
    processor = Processor()
    model = Model()

    class CompileConfig:
        def __init__(self, **options):
            self.options = options

    transformers = SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: processor),
        AutoModelForMultimodalLM=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: model
        ),
        CompileConfig=CompileConfig,
    )
    engine = Qwen3AsrEngine(
        model_dir=_model_dir(tmp_path),
        cache_implementation="static",
        compile_dynamic=True,
        torch_module=Torch,
        transformers_module=transformers,
    )

    engine.transcribe(Utterance(np.zeros(16_000, dtype=np.float32), 16_000, 0, 1))

    compile_config = model.inputs["compile_config"]
    assert compile_config.options == {"dynamic": True}


def test_startup_warmup_is_idempotent_and_forces_decode_tokens(tmp_path):
    processor = Processor()
    model = Model()
    calls = []
    original_generate = model.generate

    def generate(**inputs):
        calls.append(inputs)
        return original_generate(**inputs)

    model.generate = generate
    transformers = SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: processor),
        AutoModelForMultimodalLM=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: model
        ),
    )
    engine = Qwen3AsrEngine(
        model_dir=_model_dir(tmp_path),
        device="cuda",
        dtype="float16",
        startup_warmup_seconds=1.25,
        torch_module=Torch,
        transformers_module=transformers,
    )

    engine.warmup()
    engine.warmup()

    assert len(calls) == 1
    assert calls[0]["min_new_tokens"] == 8
    assert processor.request["audio"].shape == (20_000,)


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


def test_fa2_alias_is_normalized_and_passed_to_transformers(tmp_path, monkeypatch):
    processor = Processor()
    model = Model()
    model_options = {}
    transformers = SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: processor),
        AutoModelForMultimodalLM=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: model_options.update(kwargs) or model
        ),
    )
    real_import = __import__

    def import_module(name):
        if name == "flash_attn":
            return SimpleNamespace(__version__="2.test")
        if name == "flash_attn.flash_attn_interface":
            return SimpleNamespace()
        return real_import(name)

    monkeypatch.setattr("g1_speech.qwen3_asr.importlib.import_module", import_module)
    engine = Qwen3AsrEngine(
        model_dir=_model_dir(tmp_path),
        device="cuda",
        dtype="bfloat16",
        attention_implementation="fa2",
        torch_module=Torch,
        transformers_module=transformers,
    )

    engine.load()

    assert model_options["attn_implementation"] == "flash_attention_2"


def test_fa2_rejects_cpu_before_importing_extension():
    with pytest.raises(RuntimeError, match="device=cuda"):
        validate_flash_attention_2_environment(
            Torch, SimpleNamespace(type="cpu"), Torch.bfloat16
        )


def test_fa2_rejects_float32():
    with pytest.raises(RuntimeError, match="float16 or bfloat16"):
        validate_flash_attention_2_environment(
            Torch, SimpleNamespace(type="cuda"), Torch.float32
        )
