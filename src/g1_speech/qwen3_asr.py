"""Transformers-native Qwen3-ASR adapter for 16 kHz offline utterances."""

from __future__ import annotations

import gc
import importlib
import logging
import os
import threading
import time
from contextlib import ExitStack, nullcontext
from dataclasses import asdict, dataclass
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import RecognitionResult, Utterance

logger = logging.getLogger(__name__)


_LANGUAGE_CODES = {
    "chinese": "zh",
    "english": "en",
    "cantonese": "yue",
    "japanese": "ja",
    "korean": "ko",
    "french": "fr",
    "german": "de",
    "spanish": "es",
    "portuguese": "pt",
    "russian": "ru",
}


@dataclass(frozen=True)
class Qwen3AsrProfile:
    """Per-request diagnostic timings; intentionally separate from RecognitionResult."""

    audio_seconds: float
    processor_ms: float
    h2d_ms: float
    generate_ms: float
    decode_ms: float
    total_ms: float
    generated_tokens: int
    generated_tokens_per_second: float
    rtf: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def validate_qwen3_asr_model_dir(model_dir: str | os.PathLike) -> Path:
    root = Path(model_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Qwen3-ASR model directory not found: {root}")
    required = ("config.json", "processor_config.json", "tokenizer.json")
    missing = [name for name in required if not (root / name).is_file()]
    has_weights = (root / "model.safetensors").is_file() or (
        root / "model.safetensors.index.json"
    ).is_file()
    if not has_weights:
        missing.append("model.safetensors[.index.json]")
    if missing:
        raise FileNotFoundError(
            f"Qwen3-ASR model directory is incomplete ({', '.join(missing)}): {root}"
        )
    return root


def ensure_torch_numpy_compatible(torch_module: Any) -> None:
    """Catch Jetson PyTorch builds compiled against NumPy 1 while NumPy 2 is active."""
    try:
        torch_module.from_numpy(np.zeros(1, dtype=np.float32))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "PyTorch cannot exchange arrays with the installed NumPy; on Jetson install "
            "the qwen3-asr extra (which pins a compatible NumPy 1.x) in the ASR environment"
        ) from exc


def _model_name(model_dir: str | os.PathLike) -> str:
    name = Path(model_dir).name.lower().removesuffix("-hf").replace("-", "_")
    return name if name.startswith("qwen3_asr") else "qwen3_asr"


def _sdpa_supports_enable_gqa(torch_module: Any) -> bool:
    try:
        implementation = torch_module.nn.functional.scaled_dot_product_attention
    except AttributeError:
        return True
    return "enable_gqa" in (implementation.__doc__ or "")


def validate_flash_attention_2_environment(
    torch_module: Any,
    device: Any,
    dtype: Any,
) -> str:
    """Validate the constraints that FA2 cannot safely infer or repair for us."""
    if device.type != "cuda":
        raise RuntimeError("FlashAttention 2 requires qwen3_asr.device=cuda")
    if dtype not in {torch_module.float16, torch_module.bfloat16}:
        raise RuntimeError(
            "FlashAttention 2 requires qwen3_asr.dtype=float16 or bfloat16"
        )
    capability = torch_module.cuda.get_device_capability(device)
    if capability < (8, 0):
        raise RuntimeError(
            "FlashAttention 2 requires an Ampere-or-newer NVIDIA GPU; "
            f"detected compute capability {capability[0]}.{capability[1]}"
        )
    try:
        module = importlib.import_module("flash_attn")
        importlib.import_module("flash_attn.flash_attn_interface")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "FlashAttention 2 is selected but flash-attn cannot be imported. "
            "On Jetson, run scripts/setup-flash-attention-2.sh in this repository."
        ) from exc
    return str(getattr(module, "__version__", "installed"))


def _register_sdpa_compatibility(transformers_module: Any, torch_module: Any) -> str:
    """Register SDPA with explicit KV expansion for PyTorch builds without enable_gqa."""
    interface_name = "g1_sdpa_compat"
    interfaces = transformers_module.modeling_utils.ALL_ATTENTION_FUNCTIONS
    if interface_name in interfaces:
        return interface_name

    def repeat_kv(hidden_states, repetitions: int):
        if repetitions == 1:
            return hidden_states
        batch, heads, length, head_dim = hidden_states.shape
        expanded = hidden_states[:, :, None, :, :].expand(
            batch, heads, repetitions, length, head_dim
        )
        return expanded.reshape(batch, heads * repetitions, length, head_dim)

    def sdpa_compatibility_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout: float = 0.0,
        scaling: float | None = None,
        is_causal: bool | None = None,
        **_kwargs,
    ):
        groups = getattr(module, "num_key_value_groups", 1)
        if groups > 1:
            key = repeat_kv(key, groups)
            value = repeat_kv(value, groups)
        is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
        is_causal = query.shape[2] > 1 and attention_mask is None and is_causal
        output = torch_module.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout,
            scale=scaling,
            is_causal=is_causal,
        )
        return output.transpose(1, 2).contiguous(), None

    interfaces.register(interface_name, sdpa_compatibility_forward)
    return interface_name


class Qwen3AsrEngine:
    def __init__(
        self,
        *,
        model_dir: str | os.PathLike,
        device: str = "auto",
        dtype: str = "auto",
        sample_rate: int = 16000,
        language: str | None = "zh",
        prompt: str | None = None,
        max_new_tokens: int = 256,
        attention_implementation: str | None = None,
        compile_model: bool = False,
        compile_mode: str = "reduce-overhead",
        compile_dynamic: bool = False,
        startup_warmup_seconds: float = 0.0,
        cache_implementation: str | None = None,
        quantization: str | None = None,
        log_profile: bool = False,
        torch_module: Any | None = None,
        transformers_module: Any | None = None,
    ) -> None:
        self.name = _model_name(model_dir)
        self._model_dir = model_dir
        self._device_setting = device
        self._dtype_setting = dtype
        self._sample_rate = sample_rate
        self._language = language
        self._prompt = prompt
        self._max_new_tokens = max_new_tokens
        self._attention_implementation = attention_implementation
        self._compile_model = compile_model
        self._compile_mode = compile_mode
        self._compile_dynamic = compile_dynamic
        self._startup_warmup_seconds = startup_warmup_seconds
        self._cache_implementation = cache_implementation
        self._quantization = quantization
        self._log_profile = log_profile
        self._torch = torch_module
        self._transformers = transformers_module
        self._processor = None
        self._model = None
        self._generation_compile_config = None
        self._device = None
        self._dtype = None
        self._effective_attention_implementation = None
        self._lock = threading.Lock()
        self._warmup_lock = threading.Lock()
        self._warmup_done = False

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            model_dir = validate_qwen3_asr_model_dir(self._model_dir)
            if self._torch is None:
                import torch

                self._torch = torch
            if self._transformers is None:
                # Avoid importing the system TensorFlow stack through Transformers.
                os.environ.setdefault("USE_TF", "0")
                os.environ.setdefault("USE_FLAX", "0")
                import transformers

                self._transformers = transformers
            ensure_torch_numpy_compatible(self._torch)
            self._device = self._resolve_device()
            self._dtype = self._resolve_dtype()
            self._effective_attention_implementation = self._resolve_attention_implementation()
            logger.info(
                "Loading Qwen3-ASR: model=%s device=%s dtype=%s attention=%s",
                model_dir,
                self._device,
                self._dtype,
                self._effective_attention_implementation or "auto",
            )
            processor = self._transformers.AutoProcessor.from_pretrained(
                str(model_dir), local_files_only=True
            )
            model_kwargs: dict[str, Any] = {
                "local_files_only": True,
                "dtype": self._dtype,
                "device_map": "cuda:0" if self._device.type == "cuda" else "cpu",
            }
            if self._effective_attention_implementation:
                model_kwargs["attn_implementation"] = self._effective_attention_implementation
            if self._quantization == "bnb_nf4":
                if self._device.type != "cuda" or self._dtype != self._torch.float16:
                    raise ValueError("bnb_nf4 requires CUDA with float16 compute")
                # Keep the acoustic frontend and multimodal projection in FP16.
                # Only the repeated language decoder uses W4A16; this avoids
                # introducing an uncontrolled acoustic-accuracy change.
                model_kwargs["quantization_config"] = self._transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self._torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=False,
                    llm_int8_skip_modules=[
                        "model.audio_tower",
                        "model.multi_modal_projector",
                        "lm_head",
                    ],
                )
            model = self._transformers.AutoModelForMultimodalLM.from_pretrained(
                str(model_dir), **model_kwargs
            )
            model.eval()
            if self._compile_dynamic:
                self._generation_compile_config = self._transformers.CompileConfig(
                    dynamic=True
                )
            if self._compile_model:
                compile_target = getattr(
                    getattr(model, "model", None), "language_model", None
                )
                if compile_target is None:
                    compile_target = model
                    compile_scope = "model"
                else:
                    compile_scope = "language_model"
                logger.info(
                    "Compiling Qwen3-ASR %s forward(mode=%s)",
                    compile_scope,
                    self._compile_mode,
                )
                # Compile the autoregressive decoder when the model exposes it.
                # Compiling the full multimodal forward mixes data-dependent audio
                # control flow with decode and creates graph breaks between CUDA
                # Graph partitions.  The decoder is the repeated, latency-dominant
                # portion and has stable per-token execution.
                compiled_forward = self._torch.compile(
                    compile_target.forward, mode=self._compile_mode
                )
                mark_step_begin = getattr(
                    getattr(self._torch, "compiler", None),
                    "cudagraph_mark_step_begin",
                    None,
                )
                if mark_step_begin is None:
                    compile_target.forward = compiled_forward
                else:
                    # Transformers generate() invokes forward once per token.  In
                    # reduce-overhead mode PyTorch may reuse CUDA Graph output
                    # buffers across those invocations, so each call must declare
                    # a new iteration before the previous output is consumed by
                    # the next decode step.
                    @wraps(compiled_forward)
                    def compiled_forward_with_step(*args, **kwargs):
                        mark_step_begin()
                        return compiled_forward(*args, **kwargs)

                    compile_target.forward = compiled_forward_with_step
            self._processor = processor
            self._model = model
            logger.info("Qwen3-ASR loaded")

    def warmup(self) -> None:
        """Optionally compile generation before live microphone capture starts."""
        if self._startup_warmup_seconds <= 0:
            return
        with self._warmup_lock:
            if self._warmup_done:
                return
            self.load()
            sample_count = max(
                1, round(self._startup_warmup_seconds * self._sample_rate)
            )
            logger.info(
                "Warming Qwen3-ASR generation with %.2fs synthetic audio; "
                "the first compile can take several minutes",
                self._startup_warmup_seconds,
            )
            started = time.perf_counter()
            self._transcribe(
                Utterance(
                    np.zeros(sample_count, dtype=np.float32),
                    self._sample_rate,
                    0,
                    0,
                ),
                collect_profile=False,
                min_new_tokens=8,
            )
            self._warmup_done = True
            logger.info(
                "Qwen3-ASR startup warm-up complete in %.1fs",
                time.perf_counter() - started,
            )

    def _resolve_attention_implementation(self) -> str | None:
        requested = self._attention_implementation
        if requested == "fa2":
            requested = "flash_attention_2"
        if requested == "flash_attention_2":
            version = validate_flash_attention_2_environment(
                self._torch, self._device, self._dtype
            )
            logger.info("Using FlashAttention 2: flash-attn=%s", version)
            return requested
        if requested != "sdpa" or _sdpa_supports_enable_gqa(self._torch):
            return requested
        effective = _register_sdpa_compatibility(self._transformers, self._torch)
        logger.info(
            "PyTorch SDPA lacks enable_gqa; using explicit KV expansion via %s",
            effective,
        )
        return effective

    def _resolve_device(self):
        if self._device_setting == "cuda":
            if not self._torch.cuda.is_available():
                raise RuntimeError("qwen3_asr.device=cuda, but CUDA is not available")
            return self._torch.device("cuda:0")
        if self._device_setting == "auto":
            selected = "cuda:0" if self._torch.cuda.is_available() else "cpu"
            return self._torch.device(selected)
        return self._torch.device("cpu")

    def _resolve_dtype(self):
        if self._dtype_setting == "auto":
            if self._device.type == "cuda":
                return self._torch.bfloat16
            return self._torch.float32
        return getattr(self._torch, self._dtype_setting)

    def transcribe(self, utterance: Utterance) -> RecognitionResult:
        result, profile = self._transcribe(
            utterance, collect_profile=self._log_profile
        )
        if profile is not None:
            logger.info(
                "Qwen3-ASR profile tokens=%d generate=%.1fms tokens_per_second=%.2f "
                "processor=%.1fms h2d=%.1fms decode=%.1fms",
                profile.generated_tokens,
                profile.generate_ms,
                profile.generated_tokens_per_second,
                profile.processor_ms,
                profile.h2d_ms,
                profile.decode_ms,
            )
        return result

    def transcribe_profiled(
        self,
        utterance: Utterance,
        *,
        record_ranges: bool = False,
    ) -> tuple[RecognitionResult, Qwen3AsrProfile]:
        """Transcribe with synchronized stage timings for benchmarks and profilers."""
        result, profile = self._transcribe(
            utterance,
            collect_profile=True,
            record_ranges=record_ranges,
        )
        assert profile is not None
        return result, profile

    def _transcribe(
        self,
        utterance: Utterance,
        *,
        collect_profile: bool,
        record_ranges: bool = False,
        min_new_tokens: int | None = None,
    ) -> tuple[RecognitionResult, Qwen3AsrProfile | None]:
        if utterance.sample_rate != self._sample_rate:
            raise ValueError(
                f"Audio sample rate {utterance.sample_rate} does not match model rate "
                f"{self._sample_rate}"
            )
        self.load()
        samples = np.asarray(utterance.samples, dtype=np.float32).reshape(-1)
        request: dict[str, Any] = {"audio": samples}
        if self._language:
            request["language"] = self._language
        if self._prompt:
            request["prompt"] = self._prompt
        self._synchronize_device()
        started = time.perf_counter()
        timings: dict[str, float] = {}
        with (
            self._range("qwen3_asr_total", record_ranges),
            self._lock,
            self._torch.inference_mode(),
        ):
            inputs = self._run_stage(
                "processor",
                lambda: self._processor.apply_transcription_request(**request),
                timings,
                collect_profile,
                record_ranges,
            )
            inputs = self._run_stage(
                "h2d",
                lambda: inputs.to(self._device, self._dtype),
                timings,
                collect_profile,
                record_ranges,
            )
            generation_options: dict[str, Any] = {
                "max_new_tokens": self._max_new_tokens,
                "do_sample": False,
            }
            if min_new_tokens is not None:
                generation_options["min_new_tokens"] = min_new_tokens
            if self._cache_implementation:
                generation_options["cache_implementation"] = self._cache_implementation
            if self._generation_compile_config is not None:
                generation_options["compile_config"] = self._generation_compile_config
            output_ids = self._run_stage(
                "generate",
                lambda: self._model.generate(**inputs, **generation_options),
                timings,
                collect_profile,
                record_ranges,
            )
            input_tokens = inputs["input_ids"].shape[1]

            def decode_output():
                generated = output_ids[:, input_tokens:]
                decoded = self._processor.decode(generated, return_format="parsed")[0]
                return generated, decoded

            generated_ids, parsed = self._run_stage(
                "decode",
                decode_output,
                timings,
                collect_profile,
                record_ranges,
            )
        self._synchronize_device()
        inference_ms = (time.perf_counter() - started) * 1000
        audio_seconds = samples.size / self._sample_rate
        rtf = inference_ms / 1000 / audio_seconds if audio_seconds else float("inf")
        if isinstance(parsed, dict):
            text = str(parsed.get("transcription", "")).strip()
            detected = str(parsed.get("language", "")).strip()
        else:
            text = str(parsed).strip()
            detected = ""
        language = self._language or _LANGUAGE_CODES.get(detected.lower(), detected or "unknown")
        logger.info(
            "Qwen3-ASR recognition %.1fms audio=%.2fs RTF=%.4f: %r",
            inference_ms,
            audio_seconds,
            rtf,
            text,
        )
        result = RecognitionResult(
            text=text,
            language=language,
            inference_ms=inference_ms,
            engine=self.name,
        )
        profile = None
        if collect_profile:
            generated_tokens = int(generated_ids.shape[-1])
            generate_ms = timings["generate"]
            profile = Qwen3AsrProfile(
                audio_seconds=audio_seconds,
                processor_ms=timings["processor"],
                h2d_ms=timings["h2d"],
                generate_ms=generate_ms,
                decode_ms=timings["decode"],
                total_ms=inference_ms,
                generated_tokens=generated_tokens,
                generated_tokens_per_second=(
                    generated_tokens / (generate_ms / 1000) if generate_ms else float("inf")
                ),
                rtf=rtf,
            )
        return result, profile

    def _run_stage(
        self,
        name: str,
        operation,
        timings: dict[str, float],
        collect_profile: bool,
        record_ranges: bool,
    ):
        context = self._range(name, record_ranges)
        if not collect_profile:
            with context:
                return operation()
        self._synchronize_device()
        started = time.perf_counter()
        with context:
            value = operation()
        self._synchronize_device()
        timings[name] = (time.perf_counter() - started) * 1000
        return value

    def _range(self, name: str, enabled: bool):
        if not enabled:
            return nullcontext()
        stack = ExitStack()
        profiler = getattr(self._torch, "profiler", None)
        if profiler is not None and hasattr(profiler, "record_function"):
            stack.enter_context(profiler.record_function(name))
        cuda = getattr(self._torch, "cuda", None)
        nvtx = getattr(cuda, "nvtx", None)
        if self._device.type == "cuda" and nvtx is not None and hasattr(nvtx, "range"):
            stack.enter_context(nvtx.range(name))
        return stack

    def _synchronize_device(self) -> None:
        if self._device.type == "cuda" and hasattr(self._torch.cuda, "synchronize"):
            self._torch.cuda.synchronize(self._device)

    def close(self) -> None:
        with self._lock:
            model = self._model
            self._model = None
            self._processor = None
            self._warmup_done = False
        if model is None:
            return
        del model
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
