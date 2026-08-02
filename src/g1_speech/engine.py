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
        raise FileNotFoundError(f"SenseVoice 模型目录不存在: {root}")

    if model_file:
        model = Path(model_file).expanduser().resolve()
        if not model.is_file():
            raise FileNotFoundError(f"SenseVoice 模型文件不存在: {model}")
        tokens = model.parent / "tokens.txt"
        if not tokens.is_file():
            raise FileNotFoundError(f"SenseVoice tokens.txt 不存在: {tokens}")
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
    raise FileNotFoundError(f"在 {root} 下未找到 SenseVoice model.int8.onnx/model.onnx 和 tokens.txt")


def ensure_provider_available(sherpa_module: Any, device: str) -> None:
    """Reject sherpa CPU wheels that would silently fall back from CUDA to CPU."""
    if device != "cuda":
        return
    version = str(getattr(sherpa_module, "__version__", ""))
    if "+cuda" not in version:
        raise RuntimeError(
            "sensevoice.device=cuda，但当前 sherpa-onnx 不是 CUDA 构建"
            f"（version={version or 'unknown'}）；拒绝静默回退到 CPU"
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
            logger.info("加载 SenseVoice: model=%s provider=%s", model, provider)
            self._recognizer = self._sherpa.OfflineRecognizer.from_sense_voice(
                model=str(model),
                tokens=str(tokens),
                num_threads=self._num_threads,
                use_itn=self._use_itn,
                language=self._language,
                provider=provider,
                debug=False,
            )
            logger.info("SenseVoice 加载完成")

    def transcribe(self, utterance: Utterance) -> RecognitionResult:
        if utterance.sample_rate != self._sample_rate:
            raise ValueError(
                f"音频采样率 {utterance.sample_rate} 与模型采样率 {self._sample_rate} 不一致"
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
        logger.info("SenseVoice 识别 %.1fms: %r", inference_ms, text)
        return RecognitionResult(
            text=text,
            language=self._language,
            inference_ms=inference_ms,
            engine=self.name,
        )
