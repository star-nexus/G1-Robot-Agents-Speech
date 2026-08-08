"""Pure SenseVoice inference engine; no microphone, UI, network, or business logic."""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import RecognitionResult, Utterance

logger = logging.getLogger(__name__)


def find_model_files(
    model_dir: str | os.PathLike,
    model_file: str | os.PathLike | None = None,
) -> tuple[Path, Path]:
    root = Path(model_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"SenseVoice model directory not found: {root}")

    if model_file:
        model = Path(model_file).expanduser().resolve()
        if not model.is_file():
            raise FileNotFoundError(f"SenseVoice model file not found: {model}")
        tokens = model.parent / "tokens.txt"
        if not tokens.is_file():
            raise FileNotFoundError(f"SenseVoice tokens.txt not found: {tokens}")
        return model, tokens

    candidates: list[Path] = []
    if (root / "model.int8.onnx").is_file() or (root / "model.onnx").is_file():
        candidates.append(root)
    candidates.extend(sorted(root.glob("sherpa-onnx-sense-voice-*")))
    for candidate in candidates:
        model = candidate / "model.int8.onnx"
        if not model.is_file():
            model = candidate / "model.onnx"
        tokens = candidate / "tokens.txt"
        if model.is_file() and tokens.is_file():
            return model, tokens
    raise FileNotFoundError(
        f"SenseVoice model.int8.onnx/model.onnx and tokens.txt were not found under {root}"
    )


def ensure_provider_available(sherpa_module: Any, device: str) -> None:
    """Reject sherpa CPU wheels that would silently fall back from CUDA to CPU."""
    if device != "cuda":
        return
    version = str(getattr(sherpa_module, "__version__", ""))
    if "+cuda" not in version:
        raise RuntimeError(
            "sensevoice.device=cuda, but sherpa-onnx is not a CUDA build "
            f"(version={version or 'unknown'}); refusing to silently fall back to CPU"
        )


class SenseVoiceEngine:
    name = "sensevoice"

    def __init__(
        self,
        *,
        model_dir: str | os.PathLike,
        model_file: str | os.PathLike | None = None,
        device: str = "cpu",
        sample_rate: int = 16000,
        language: str = "zh",
        use_itn: bool = True,
        num_threads: int = 4,
        sherpa_module: Any | None = None,
    ) -> None:
        self._model_dir = model_dir
        self._model_file = model_file
        self._device = device
        self._sample_rate = sample_rate
        self._language = language
        self._use_itn = use_itn
        self._num_threads = num_threads
        self._sherpa = sherpa_module
        self._recognizer = None
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self._recognizer is not None:
                return
            if self._sherpa is None:
                import sherpa_onnx

                self._sherpa = sherpa_onnx
            ensure_provider_available(self._sherpa, self._device)
            model, tokens = find_model_files(self._model_dir, self._model_file)
            provider = "cpu" if self._device in ("cpu", "auto") else self._device
            logger.info("Loading SenseVoice: model=%s provider=%s", model, provider)
            self._recognizer = self._sherpa.OfflineRecognizer.from_sense_voice(
                model=str(model),
                tokens=str(tokens),
                num_threads=self._num_threads,
                use_itn=self._use_itn,
                language=self._language,
                provider=provider,
                debug=False,
            )
            logger.info("SenseVoice loaded")

    def transcribe(self, utterance: Utterance) -> RecognitionResult:
        if utterance.sample_rate != self._sample_rate:
            raise ValueError(
                f"Audio sample rate {utterance.sample_rate} does not match model rate "
                f"{self._sample_rate}"
            )
        self.load()
        samples = np.asarray(utterance.samples, dtype=np.float32).reshape(-1)
        started = time.perf_counter()
        with self._lock:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(self._sample_rate, samples)
            self._recognizer.decode_stream(stream)
            text = (stream.result.text or "").strip()
        inference_ms = (time.perf_counter() - started) * 1000
        logger.info("SenseVoice recognition %.1fms: %r", inference_ms, text)
        return RecognitionResult(
            text=text,
            language=self._language,
            inference_ms=inference_ms,
            engine=self.name,
        )

    def close(self) -> None:
        with self._lock:
            self._recognizer = None
