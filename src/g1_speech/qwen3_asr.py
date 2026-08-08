"""Transformers-native Qwen3-ASR adapter for 16 kHz offline utterances."""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
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
        self._torch = torch_module
        self._transformers = transformers_module
        self._processor = None
        self._model = None
        self._device = None
        self._dtype = None
        self._lock = threading.Lock()

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
            logger.info(
                "Loading Qwen3-ASR: model=%s device=%s dtype=%s",
                model_dir,
                self._device,
                self._dtype,
            )
            processor = self._transformers.AutoProcessor.from_pretrained(
                str(model_dir), local_files_only=True
            )
            model_kwargs: dict[str, Any] = {
                "local_files_only": True,
                "dtype": self._dtype,
                "device_map": "cuda:0" if self._device.type == "cuda" else "cpu",
            }
            if self._attention_implementation:
                model_kwargs["attn_implementation"] = self._attention_implementation
            model = self._transformers.AutoModelForMultimodalLM.from_pretrained(
                str(model_dir), **model_kwargs
            )
            model.eval()
            self._processor = processor
            self._model = model
            logger.info("Qwen3-ASR loaded")

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
        started = time.perf_counter()
        with self._lock, self._torch.inference_mode():
            inputs = self._processor.apply_transcription_request(**request)
            inputs = inputs.to(self._device, self._dtype)
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            )
            generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
            parsed = self._processor.decode(generated_ids, return_format="parsed")[0]
        inference_ms = (time.perf_counter() - started) * 1000
        if isinstance(parsed, dict):
            text = str(parsed.get("transcription", "")).strip()
            detected = str(parsed.get("language", "")).strip()
        else:
            text = str(parsed).strip()
            detected = ""
        language = self._language or _LANGUAGE_CODES.get(detected.lower(), detected or "unknown")
        logger.info("Qwen3-ASR recognition %.1fms: %r", inference_ms, text)
        return RecognitionResult(
            text=text,
            language=language,
            inference_ms=inference_ms,
            engine=self.name,
        )

    def close(self) -> None:
        with self._lock:
            model = self._model
            self._model = None
            self._processor = None
        if model is None:
            return
        del model
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
