"""Interruptible low-latency ALSA playback with an AEC render-reference tee."""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np

from .audio import AudioInputUnavailable, read_alsa_card_numbers
from .audio_processing import AudioProcessor, Pcm16Resampler
from .config import AudioOutputConfig

logger = logging.getLogger(__name__)

_ALSA_HARDWARE_NAME = re.compile(r"\(hw:(\d+),(\d+)\)\s*$")


def resolve_alsa_output_device(
    sounddevice_module: Any,
    *,
    card_id: str | None,
    pcm_device: int,
    proc_root: Path = Path("/proc/asound"),
) -> tuple[int, str]:
    candidates: list[tuple[int, str, int, int]] = []
    for index, device in enumerate(sounddevice_module.query_devices()):
        if int(device.get("max_output_channels", 0)) < 1:
            continue
        name = str(device.get("name", ""))
        match = _ALSA_HARDWARE_NAME.search(name)
        if match is not None:
            candidates.append((index, name, int(match.group(1)), int(match.group(2))))

    if card_id is None:
        matching = [item for item in candidates if item[3] == pcm_device]
        if len(matching) != 1:
            names = ", ".join(item[1] for item in matching) or "none"
            raise AudioInputUnavailable(
                "audio_output.alsa_card is unset and hardware output auto-detection "
                f"is ambiguous (candidates: {names})"
            )
        index, name, _card_number, _device_number = matching[0]
        return index, name

    cards = read_alsa_card_numbers(proc_root)
    if card_id not in cards:
        available = ", ".join(sorted(cards)) or "none"
        raise AudioInputUnavailable(
            f"ALSA output card ID {card_id!r} is unavailable (available: {available})"
        )
    card_number = cards[card_id]
    for index, name, candidate_card, candidate_device in candidates:
        if candidate_card == card_number and candidate_device == pcm_device:
            return index, name
    raise AudioInputUnavailable(
        f"ALSA output {card_id!r} maps to hw:{card_number},{pcm_device}, but "
        "PortAudio cannot see that hardware output"
    )


class AlsaOutputPlayer:
    """Persistent RawOutputStream that can be aborted from the VAD thread."""

    def __init__(
        self,
        settings: AudioOutputConfig,
        *,
        render_sink: AudioProcessor,
        sounddevice_module: Any | None = None,
        proc_asound_root: Path = Path("/proc/asound"),
    ) -> None:
        self._settings = settings
        self._render_sink = render_sink
        self._sd = sounddevice_module
        self._proc_root = proc_asound_root
        self._stream: Any | None = None
        self._lock = threading.RLock()
        self._source_rate: int | None = None
        self._resampler: Pcm16Resampler | None = None
        self._pending = bytearray()
        self._aborted = False
        self.active_device: str | None = None
        self.actual_latency_seconds: float | None = None
        self.blocks_written = 0
        self.bytes_written = 0
        self.interruptions = 0

    def start(self) -> None:
        with self._lock:
            if self._stream is not None:
                return
            if self._sd is None:
                import sounddevice as sd

                self._sd = sd
            index, name = resolve_alsa_output_device(
                self._sd,
                card_id=self._settings.alsa_card,
                pcm_device=self._settings.alsa_device,
                proc_root=self._proc_root,
            )
            blocksize = self._settings.sample_rate * self._settings.block_ms // 1000
            stream = self._sd.RawOutputStream(
                device=index,
                samplerate=self._settings.sample_rate,
                blocksize=blocksize,
                channels=self._settings.channels,
                dtype=self._settings.dtype,
                latency=self._settings.latency,
            )
            stream.start()
            self._stream = stream
            self.active_device = name
            latency = getattr(stream, "latency", None)
            self.actual_latency_seconds = float(latency) if latency is not None else None
            logger.info(
                "Speaker started: backend=alsa device=%r latency_ms=%s "
                "output=%dHz/%dch/%s block_ms=%d",
                name,
                "unknown" if latency is None else f"{float(latency) * 1000:.1f}",
                self._settings.sample_rate,
                self._settings.channels,
                self._settings.dtype,
                self._settings.block_ms,
            )

    def begin(self, source_sample_rate: int) -> None:
        with self._lock:
            if self._stream is None:
                raise RuntimeError("audio output is not started")
            self._source_rate = source_sample_rate
            self._resampler = Pcm16Resampler(
                source_sample_rate,
                self._settings.sample_rate,
            )
            self._pending.clear()
            self._aborted = False
            if not getattr(self._stream, "active", True):
                self._stream.start()

    def write(self, pcm16_mono: bytes) -> bool:
        with self._lock:
            if self._aborted:
                return False
            if self._stream is None or self._resampler is None:
                raise RuntimeError("begin() must be called before playback")
            self._pending.extend(self._resampler.process(pcm16_mono))
            stream = self._stream

        frame_count = self._settings.sample_rate * self._settings.block_ms // 1000
        mono_bytes = frame_count * 2
        while True:
            with self._lock:
                if self._aborted:
                    return False
                if len(self._pending) < mono_bytes:
                    return True
                block = bytes(self._pending[:mono_bytes])
                del self._pending[:mono_bytes]
            output, reference = self._prepare_block(block)
            self._render_sink.push_render(reference, self._settings.sample_rate)
            stream.write(output)
            self.blocks_written += 1
            self.bytes_written += len(output)

    def finish(self) -> None:
        with self._lock:
            if self._aborted or not self._pending:
                self._pending.clear()
                return
            frame_count = self._settings.sample_rate * self._settings.block_ms // 1000
            mono_bytes = frame_count * 2
            self._pending.extend(b"\0" * (mono_bytes - len(self._pending)))
        self.write(b"")

    def abort(self) -> None:
        with self._lock:
            self._aborted = True
            self._pending.clear()
            stream = self._stream
            self.interruptions += 1
        if stream is not None:
            try:
                stream.abort()
            except Exception:  # noqa: BLE001
                logger.exception("ALSA playback abort failed")

    def close(self) -> None:
        with self._lock:
            stream, self._stream = self._stream, None
            self._pending.clear()
        if stream is not None:
            try:
                stream.abort()
            finally:
                stream.close()

    def metrics(self) -> dict[str, int | float | str | None]:
        return {
            "audio_output_device": self.active_device,
            "audio_output_latency_ms": (
                None
                if self.actual_latency_seconds is None
                else round(self.actual_latency_seconds * 1000, 3)
            ),
            "audio_output_blocks": self.blocks_written,
            "audio_output_bytes": self.bytes_written,
            "audio_output_interruptions": self.interruptions,
        }

    def _prepare_block(self, pcm: bytes) -> tuple[bytes, np.ndarray]:
        mono = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
        mono *= self._settings.volume
        mono_i16 = np.rint(np.clip(mono, -32768, 32767)).astype("<i2")
        reference = mono_i16.astype(np.float32) / 32768.0
        if self._settings.channels == 1:
            return mono_i16.tobytes(), reference
        stereo = np.repeat(mono_i16[:, None], self._settings.channels, axis=1)
        return stereo.astype("<i2", copy=False).tobytes(), reference
