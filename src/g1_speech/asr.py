"""ASR adapter composition, discovery, and selected-backend diagnostics."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
from collections.abc import Callable
from typing import Any

from .config import ServiceConfig
from .contracts import AsrEngine
from .engine import SenseVoiceEngine, find_model_files
from .qwen3_asr import Qwen3AsrEngine, validate_qwen3_asr_model_dir

AsrFactory = Callable[[ServiceConfig], AsrEngine]


def _sensevoice(config: ServiceConfig) -> AsrEngine:
    settings = config.sensevoice
    return SenseVoiceEngine(
        model_dir=settings.model_dir,
        model_file=settings.model_file,
        device=settings.device,
        sample_rate=config.audio.sample_rate,
        language=settings.language,
        use_itn=settings.use_itn,
        num_threads=settings.num_threads,
    )


def _qwen3_asr(config: ServiceConfig) -> AsrEngine:
    settings = config.qwen3_asr
    return Qwen3AsrEngine(
        model_dir=settings.model_dir,
        device=settings.device,
        dtype=settings.dtype,
        sample_rate=config.audio.sample_rate,
        language=settings.language,
        prompt=settings.prompt,
        max_new_tokens=settings.max_new_tokens,
        attention_implementation=settings.attention_implementation,
    )


_BUILTINS: dict[str, AsrFactory] = {
    "sensevoice": _sensevoice,
    "qwen3_asr": _qwen3_asr,
    "qwen3-asr": _qwen3_asr,
}


def resolve_asr_factory(backend: str) -> AsrFactory:
    normalized = backend.strip().lower()
    if normalized in _BUILTINS:
        return _BUILTINS[normalized]
    if ":" in backend:
        module_name, attribute = backend.rsplit(":", 1)
        factory = getattr(importlib.import_module(module_name), attribute)
        if not callable(factory):
            raise TypeError(f"ASR factory is not callable: {backend}")
        return factory
    entries = importlib.metadata.entry_points()
    selected = entries.select(group="g1_speech.asr_backends", name=backend)
    for entry in selected:
        factory = entry.load()
        if not callable(factory):
            raise TypeError(f"ASR entry point is not callable: {backend}")
        return factory
    raise ValueError(
        f"unknown ASR backend {backend!r}; use sensevoice, qwen3_asr, a "
        "g1_speech.asr_backends entry point, or module:factory"
    )


def create_asr_engine(config: ServiceConfig) -> AsrEngine:
    return resolve_asr_factory(config.asr.backend)(config)


def asr_diagnostic_checks(config: ServiceConfig) -> list[tuple[str, Callable[[], Any]]]:
    backend = config.asr.backend.strip().lower()
    if backend == "sensevoice":
        return [
            (
                "SenseVoice model files",
                lambda: find_model_files(
                    config.sensevoice.model_dir, config.sensevoice.model_file
                ),
            ),
            ("sherpa_onnx", lambda: _module_path("sherpa_onnx")),
        ]
    if backend in {"qwen3_asr", "qwen3-asr"}:
        checks = [
            (
                "Qwen3-ASR model files",
                lambda: validate_qwen3_asr_model_dir(config.qwen3_asr.model_dir),
            ),
            ("torch", lambda: _module_path("torch")),
            ("transformers", _qwen_module_path),
        ]
        if config.qwen3_asr.attention_implementation in {
            "fa2",
            "flash_attention_2",
        }:
            checks.append(("flash_attn", lambda: _module_path("flash_attn")))
        return checks
    return [("ASR adapter", lambda: resolve_asr_factory(config.asr.backend))]


def _module_path(name: str) -> str:
    module = importlib.import_module(name)
    return str(getattr(module, "__file__", "built-in"))


def _qwen_module_path() -> str:
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    return _module_path("transformers")
