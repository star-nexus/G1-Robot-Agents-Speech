"""Interruptible low-latency ALSA playback with an AEC render-reference tee."""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from .input import AudioInputUnavailable, read_alsa_card_numbers
from .processing import AudioProcessor, Pcm16Resampler
from ..config import AudioOutputConfig
from ...core.control import ControlStamp
from ...core.timing import RuntimeTimingAudit

logger = logging.getLogger(__name__)

_ALSA_HARDWARE_NAME = re.compile(r"\(hw:(\d+),(\d+)\)\s*$")


class _PcmRingBuffer:
    """Fixed-capacity mono PCM16 ring in the hardware output clock domain."""

    def __init__(self, capacity_samples: int) -> None:
        if capacity_samples < 1:
            raise ValueError("ring buffer capacity must be positive")
        self._samples = np.empty(capacity_samples, dtype="<i2")
        self._read = 0
        self._write = 0
        self.size = 0

    @property
    def capacity(self) -> int:
        return int(self._samples.size)

    @property
    def writable(self) -> int:
        return self.capacity - self.size

    def write(self, samples: np.ndarray) -> int:
        count = min(int(samples.size), self.writable)
        if count == 0:
            return 0
        first = min(count, self.capacity - self._write)
        self._samples[self._write : self._write + first] = samples[:first]
        remaining = count - first
        if remaining:
            self._samples[:remaining] = samples[first : first + remaining]
        self._write = (self._write + count) % self.capacity
        self.size += count
        return count

    def read(self, count: int) -> np.ndarray:
        count = min(count, self.size)
        result = np.empty(count, dtype="<i2")
        if count == 0:
            return result
        first = min(count, self.capacity - self._read)
        result[:first] = self._samples[self._read : self._read + first]
        remaining = count - first
        if remaining:
            result[first:] = self._samples[:remaining]
        self._read = (self._read + count) % self.capacity
        self.size -= count
        return result

    def clear(self) -> None:
        self._read = 0
        self._write = 0
        self.size = 0


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
    """Continuous bounded PCM timeline consumed by a dedicated ALSA thread."""

    def __init__(
        self,
        settings: AudioOutputConfig,
        *,
        render_sink: AudioProcessor,
        sounddevice_module: Any | None = None,
        proc_asound_root: Path = Path("/proc/asound"),
        timing_audit: RuntimeTimingAudit | None = None,
    ) -> None:
        self._settings = settings
        self._render_sink = render_sink
        self._sd = sounddevice_module
        self._proc_root = proc_asound_root
        self._timing = timing_audit
        self._stream: Any | None = None
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._source_rate: int | None = None
        self._resampler: Pcm16Resampler | None = None
        capacity = max(1, round(settings.sample_rate * settings.buffer_seconds))
        self._ring = _PcmRingBuffer(capacity)
        self._timing_spans: deque[tuple[int, ControlStamp | None]] = deque()
        self._enqueue_stamp: ControlStamp | None = None
        self._timing_dequeued_stamps: set[ControlStamp] = set()
        self._timing_committed_stamps: set[ControlStamp] = set()
        self._timing_written_stamps: set[ControlStamp] = set()
        self._block_samples = settings.sample_rate * settings.block_ms // 1000
        # WebRTC reverse processing needs a continuous far-end timebase.  The
        # persistent PortAudio writer is the one speaker-clock authority: every
        # hardware block carries either queued speaker PCM or explicit silence
        # to both the device and the AEC render path.
        self._continuous_render_clock = bool(
            getattr(render_sink, "mode", None) == "webrtc"
            and settings.interrupt_strategy == "persistent"
        )
        silence_pcm = np.zeros(self._block_samples, dtype="<i2").tobytes()
        self._silence_output, self._silence_reference = self._prepare_block(
            silence_pcm
        )
        self._thread: threading.Thread | None = None
        self._stop = False
        self._drain_requested = False
        self._inflight_samples = 0
        self._abort_epoch = 0
        self._aborted = False
        self._write_in_progress_epoch: int | None = None
        self._active_write_commit_sequence: int | None = None
        self._active_write_has_payload = False
        self._next_write_commit_sequence = 0
        self._hardware_write_started_ns: int | None = None
        self._hardware_write_kind: str | None = None
        self._pending_recovery_epoch: int | None = None
        self._interrupt_started_ns: int | None = None
        self._first_replacement_pcm_ns: int | None = None
        self._first_replacement_enqueue_ns: int | None = None
        self._first_replacement_dequeue_ns: int | None = None
        self._last_stream_active: bool | None = None
        self.active_device: str | None = None
        self.actual_latency_seconds: float | None = None
        self.blocks_written = 0
        self.bytes_written = 0
        self.render_clock_silence_blocks = 0
        self.interruptions = 0
        self.backpressure_waits = 0
        self.peak_buffered_samples = 0
        self.playback_errors = 0
        self.stream_abort_calls = 0
        self.stream_abort_errors = 0
        self.close_abort_fallbacks = 0
        self.stream_restart_attempts = 0
        self.stream_restart_successes = 0
        self.stream_restart_errors = 0
        self.stream_active_transitions = 0
        self.write_underflows = 0
        self.write_errors = 0
        self.stale_blocks_dropped_before_write = 0
        self.forbidden_stale_write_attempts = 0
        self.residual_writes_committed = 0
        self.residual_write_completions = 0
        self.recoveries = 0
        self.replacement_pcm_arrivals = 0
        self.replacement_enqueues = 0
        self.replacement_dequeues = 0
        self.last_write_ms: float | None = None
        self.max_write_ms = 0.0
        self.last_abort_ms: float | None = None
        self.max_abort_ms = 0.0
        self.last_recovery_ms: float | None = None
        self.max_recovery_ms = 0.0

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
            stream = self._sd.RawOutputStream(
                device=index,
                samplerate=self._settings.sample_rate,
                blocksize=self._block_samples,
                channels=self._settings.channels,
                dtype=self._settings.dtype,
                latency=self._settings.latency,
            )
            stream.start()
            self._stream = stream
            self._record_stream_active_locked(stream, "initial-start")
            self._stop = False
            self._thread = threading.Thread(
                target=self._playback_loop,
                name="speech-pcm-playback",
                daemon=True,
            )
            self._thread.start()
            self.active_device = name
            latency = getattr(stream, "latency", None)
            self.actual_latency_seconds = float(latency) if latency is not None else None
            logger.info(
                "Speaker started: backend=alsa device=%r latency_ms=%s "
                "output=%dHz/%dch/%s block_ms=%d interrupt_strategy=%s",
                name,
                "unknown" if latency is None else f"{float(latency) * 1000:.1f}",
                self._settings.sample_rate,
                self._settings.channels,
                self._settings.dtype,
                self._settings.block_ms,
                self._settings.interrupt_strategy,
            )

    def note_pcm_ready(self) -> None:
        """Mark the first provider PCM made available after interruption."""
        with self._condition:
            if (
                self._pending_recovery_epoch != self._abort_epoch
                or self._first_replacement_pcm_ns is not None
            ):
                return
            now_ns = time.monotonic_ns()
            self._first_replacement_pcm_ns = now_ns
            self.replacement_pcm_arrivals += 1
            latency_ms = self._since_interrupt_ms_locked(now_ns)
            logger.info(
                "PLAYBACK_RECOVERY_PCM_READY generation=%d latency_ms=%.3f",
                self._abort_epoch,
                latency_ms,
            )

    def set_control_stamp(self, stamp: ControlStamp) -> None:
        """Associate subsequent enqueue samples with one immutable turn stamp."""

        with self._condition:
            selected = stamp if stamp.scoped else None
            if selected != self._enqueue_stamp:
                self._timing_dequeued_stamps.clear()
                self._timing_committed_stamps.clear()
                self._timing_written_stamps.clear()
            self._enqueue_stamp = selected

    def enqueue(
        self,
        pcm16_mono: bytes,
        source_sample_rate: int,
        *,
        cancel: threading.Event | None = None,
    ) -> bool:
        """Append PCM to the timeline, blocking only when bounded look-ahead is full."""
        if not pcm16_mono:
            return True
        with self._condition:
            if self._stream is None or self._stop:
                raise RuntimeError("audio output is not started")
            if cancel is not None and cancel.is_set():
                return False
            enqueue_stamp = self._enqueue_stamp
            if (
                self._pending_recovery_epoch == self._abort_epoch
                and self._first_replacement_enqueue_ns is None
            ):
                now_ns = time.monotonic_ns()
                self._first_replacement_enqueue_ns = now_ns
                self.replacement_enqueues += 1
                logger.info(
                    "PLAYBACK_RECOVERY_ENQUEUE generation=%d latency_ms=%.3f bytes=%d",
                    self._abort_epoch,
                    self._since_interrupt_ms_locked(now_ns),
                    len(pcm16_mono),
                )
            if self._source_rate != source_sample_rate:
                self._source_rate = source_sample_rate
                self._resampler = Pcm16Resampler(
                    source_sample_rate,
                    self._settings.sample_rate,
                )
            assert self._resampler is not None
            converted = self._resampler.process(pcm16_mono)
            samples = np.frombuffer(converted, dtype="<i2")
            self._aborted = False
            self._drain_requested = False

        offset = 0
        while offset < samples.size:
            with self._condition:
                if self._stop or self._aborted:
                    return False
                if cancel is not None and cancel.is_set():
                    return False
                written = self._ring.write(samples[offset:])
                if written:
                    self._append_timing_span_locked(written, enqueue_stamp)
                    offset += written
                    self.peak_buffered_samples = max(
                        self.peak_buffered_samples,
                        self._ring.size,
                    )
                    self._condition.notify_all()
                    continue
                self.backpressure_waits += 1
                self._condition.wait(timeout=0.05)
        return True

    def wait_until_idle(
        self,
        *,
        cancel: threading.Event | None = None,
        timeout: float | None = None,
    ) -> bool:
        """Flush a partial final block and wait until the PCM timeline is consumed."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            self._drain_requested = True
            self._condition.notify_all()
            while self._ring.size or self._inflight_samples:
                if cancel is not None and cancel.is_set():
                    return False
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._condition.wait(timeout=min(remaining, 0.05))
                else:
                    self._condition.wait(timeout=0.05)
            self._drain_requested = False
            return True

    def abort(self) -> None:
        started_ns = time.monotonic_ns()
        with self._condition:
            self._aborted = True
            self._ring.clear()
            self._timing_spans.clear()
            self._drain_requested = False
            self._abort_epoch += 1
            generation = self._abort_epoch
            self._pending_recovery_epoch = generation
            self._interrupt_started_ns = started_ns
            self._first_replacement_pcm_ns = None
            self._first_replacement_enqueue_ns = None
            self._first_replacement_dequeue_ns = None
            residual_write = bool(
                self._active_write_commit_sequence is not None
                and self._active_write_has_payload
            )
            if residual_write:
                self.residual_writes_committed += 1
            stream = self._stream
            self.interruptions += 1
            self._condition.notify_all()
        logger.info(
            "PLAYBACK_INTERRUPT generation=%d strategy=%s stream_active=%s "
            "residual_write=%s",
            generation,
            self._settings.interrupt_strategy,
            self._stream_active(stream),
            residual_write,
        )
        if stream is not None and self._settings.interrupt_strategy == "hard_abort":
            try:
                self.stream_abort_calls += 1
                stream.abort()
            except Exception:  # noqa: BLE001
                self.stream_abort_errors += 1
                logger.exception("ALSA playback abort failed")
            finally:
                finished_ns = time.monotonic_ns()
                elapsed_ms = (finished_ns - started_ns) / 1_000_000
                with self._condition:
                    self.last_abort_ms = elapsed_ms
                    self.max_abort_ms = max(self.max_abort_ms, elapsed_ms)
                    self._record_stream_active_locked(stream, "interrupt-abort")
                logger.info(
                    "PLAYBACK_STREAM_ABORT generation=%d duration_ms=%.3f active=%s",
                    generation,
                    elapsed_ms,
                    self._stream_active(stream),
                )

    def close(self) -> None:
        with self._condition:
            stream = self._stream
            thread = self._thread
            self._stop = True
            self._ring.clear()
            self._timing_spans.clear()
            self._abort_epoch += 1
            self._condition.notify_all()
        # A healthy persistent writer finishes its one bounded block and then
        # observes _stop. Avoid recreating the JP6.2 abort/write race at normal
        # shutdown; retain abort as a bounded fallback for a genuinely stuck IO.
        if (
            thread is not None
            and self._settings.interrupt_strategy == "persistent"
        ):
            thread.join(timeout=0.25)
        needs_close_abort = bool(
            stream is not None
            and (
                self._settings.interrupt_strategy == "hard_abort"
                or (thread is not None and thread.is_alive())
            )
        )
        if needs_close_abort and stream is not None:
            if self._settings.interrupt_strategy == "persistent":
                self.close_abort_fallbacks += 1
                logger.warning("PLAYBACK_CLOSE_ABORT_FALLBACK writer did not stop")
            try:
                self.stream_abort_calls += 1
                stream.abort()
            except Exception:  # noqa: BLE001
                self.stream_abort_errors += 1
                logger.exception("ALSA playback close abort failed")
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
            if thread.is_alive():
                logger.warning("ALSA playback thread did not stop within 2s")
        with self._condition:
            self._thread = None
            self._stream = None
            self._inflight_samples = 0
        if stream is not None:
            stream.close()

    def metrics(self) -> dict[str, int | float | str | None]:
        with self._condition:
            buffered_samples = self._ring.size + self._inflight_samples
            stream = self._stream
            now_ns = time.monotonic_ns()
            payload = {
                "audio_output_device": self.active_device,
                "audio_output_latency_ms": (
                    None
                    if self.actual_latency_seconds is None
                    else round(self.actual_latency_seconds * 1000, 3)
                ),
                "audio_output_blocks": self.blocks_written,
                "audio_output_bytes": self.bytes_written,
                "audio_output_continuous_render_clock": (
                    self._continuous_render_clock
                ),
                "audio_output_render_clock_silence_blocks": (
                    self.render_clock_silence_blocks
                ),
                "audio_output_interruptions": self.interruptions,
                "audio_output_buffered_ms": round(
                    buffered_samples * 1000 / self._settings.sample_rate,
                    3,
                ),
                "audio_output_peak_buffered_ms": round(
                    self.peak_buffered_samples * 1000 / self._settings.sample_rate,
                    3,
                ),
                "audio_output_buffer_capacity_ms": round(
                    self._ring.capacity * 1000 / self._settings.sample_rate,
                    3,
                ),
                "audio_output_backpressure_waits": self.backpressure_waits,
                "audio_output_playback_errors": self.playback_errors,
                "audio_output_interrupt_strategy": self._settings.interrupt_strategy,
                "audio_output_stream_active": self._stream_active(stream),
                "audio_output_playback_thread_alive": bool(
                    self._thread is not None and self._thread.is_alive()
                ),
                "audio_output_stream_abort_calls": self.stream_abort_calls,
                "audio_output_stream_abort_errors": self.stream_abort_errors,
                "audio_output_close_abort_fallbacks": self.close_abort_fallbacks,
                "audio_output_stream_restart_attempts": self.stream_restart_attempts,
                "audio_output_stream_restart_successes": self.stream_restart_successes,
                "audio_output_stream_restart_errors": self.stream_restart_errors,
                "audio_output_stream_active_transitions": (
                    self.stream_active_transitions
                ),
                "audio_output_write_underflows": self.write_underflows,
                "audio_output_write_errors": self.write_errors,
                "audio_output_hardware_write_in_progress": (
                    self._hardware_write_started_ns is not None
                ),
                "audio_output_hardware_write_generation": (
                    self._write_in_progress_epoch
                    if self._hardware_write_kind == "pcm"
                    else None
                ),
                "audio_output_hardware_write_kind": self._hardware_write_kind,
                "audio_output_hardware_write_in_progress_ms": (
                    None
                    if self._hardware_write_started_ns is None
                    else round(
                        (now_ns - self._hardware_write_started_ns) / 1_000_000,
                        3,
                    )
                ),
                "audio_output_stale_blocks_dropped_before_write": (
                    self.stale_blocks_dropped_before_write
                ),
                "audio_output_forbidden_stale_write_attempts": (
                    self.forbidden_stale_write_attempts
                ),
                "audio_output_residual_writes_committed": (
                    self.residual_writes_committed
                ),
                "audio_output_residual_write_completions": (
                    self.residual_write_completions
                ),
                "audio_output_recovery_pending": (
                    self._pending_recovery_epoch is not None
                ),
                "audio_output_recoveries": self.recoveries,
                "audio_output_replacement_pcm_arrivals": (
                    self.replacement_pcm_arrivals
                ),
                "audio_output_replacement_enqueues": self.replacement_enqueues,
                "audio_output_replacement_dequeues": self.replacement_dequeues,
                "audio_output_first_replacement_pcm_ms": (
                    self._event_latency_ms_locked(self._first_replacement_pcm_ns)
                ),
                "audio_output_first_replacement_enqueue_ms": (
                    self._event_latency_ms_locked(self._first_replacement_enqueue_ns)
                ),
                "audio_output_first_replacement_dequeue_ms": (
                    self._event_latency_ms_locked(self._first_replacement_dequeue_ns)
                ),
                "audio_output_last_write_ms": self.last_write_ms,
                "audio_output_max_write_ms": round(self.max_write_ms, 3),
                "audio_output_last_abort_ms": self.last_abort_ms,
                "audio_output_max_abort_ms": round(self.max_abort_ms, 3),
                "audio_output_last_recovery_ms": self.last_recovery_ms,
                "audio_output_max_recovery_ms": round(self.max_recovery_ms, 3),
            }
        return payload

    def _playback_loop(self) -> None:
        while True:
            with self._condition:
                while True:
                    if self._stop:
                        return
                    available = self._ring.size
                    if available >= self._block_samples:
                        count = self._block_samples
                        break
                    if self._drain_requested and available:
                        count = available
                        break
                    if self._continuous_render_clock:
                        count = 0
                        break
                    self._condition.wait()
                has_payload = count > 0
                epoch = self._abort_epoch
                if has_payload:
                    block = self._ring.read(count)
                    timing_stamps = self._consume_timing_spans_locked(count)
                    dequeue_audit_stamps = tuple(
                        stamp
                        for stamp in timing_stamps
                        if stamp not in self._timing_dequeued_stamps
                    )
                    self._timing_dequeued_stamps.update(dequeue_audit_stamps)
                    self._inflight_samples = count
                    dequeued_ns = time.monotonic_ns()
                    if self._timing is not None:
                        for stamp in dequeue_audit_stamps:
                            self._timing.mark(
                                stamp,
                                "playback_dequeue",
                                at_ns=dequeued_ns,
                                samples=count,
                                generation=epoch,
                            )
                    recovery_dequeue = self._pending_recovery_epoch == epoch
                    if (
                        recovery_dequeue
                        and self._first_replacement_dequeue_ns is None
                    ):
                        now_ns = time.monotonic_ns()
                        self._first_replacement_dequeue_ns = now_ns
                        self.replacement_dequeues += 1
                        logger.info(
                            "PLAYBACK_RECOVERY_DEQUEUE generation=%d "
                            "latency_ms=%.3f samples=%d",
                            epoch,
                            self._since_interrupt_ms_locked(now_ns),
                            count,
                        )
                else:
                    block = None
                    timing_stamps = ()
                    dequeue_audit_stamps = ()
                    self._inflight_samples = 0
                self._condition.notify_all()

            if not has_payload:
                output = self._silence_output
                reference = self._silence_reference
            else:
                assert block is not None
                if count < self._block_samples:
                    block = np.pad(block, (0, self._block_samples - count))
                pcm = block.astype("<i2", copy=False).tobytes()
                output, reference = self._prepare_block(pcm)
            commit_sequence: int | None = None

            try:
                with self._condition:
                    if has_payload and epoch != self._abort_epoch:
                        self.stale_blocks_dropped_before_write += 1
                        continue
                    if self._stop:
                        continue
                    stream = self._stream
                    if stream is None:
                        return
                    needs_restart = not self._stream_active(stream)
                if needs_restart:
                    with self._condition:
                        self.stream_restart_attempts += 1
                    restart_started_ns = time.monotonic_ns()
                    logger.info(
                        "PLAYBACK_STREAM_RESTART_BEGIN generation=%d",
                        epoch,
                    )
                    try:
                        stream.start()
                    except Exception:  # noqa: BLE001
                        with self._condition:
                            self.stream_restart_errors += 1
                            self._record_stream_active_locked(stream, "restart-error")
                        logger.exception(
                            "PLAYBACK_STREAM_RESTART_ERROR generation=%d duration_ms=%.3f",
                            epoch,
                            (time.monotonic_ns() - restart_started_ns) / 1_000_000,
                        )
                        raise
                    with self._condition:
                        self.stream_restart_successes += 1
                        self._record_stream_active_locked(stream, "restart-success")
                    logger.info(
                        "PLAYBACK_STREAM_RESTART_OK generation=%d duration_ms=%.3f",
                        epoch,
                        (time.monotonic_ns() - restart_started_ns) / 1_000_000,
                    )

                with self._condition:
                    # Restart is deliberately outside the player lock. An
                    # interruption during restart must still prevent submission.
                    if has_payload and epoch != self._abort_epoch:
                        self.stale_blocks_dropped_before_write += 1
                        continue
                    if self._stop:
                        continue
                    if not has_payload:
                        # Silence is unscoped Control Plane data.  Refresh the
                        # bookkeeping generation after any concurrent abort;
                        # the frame advances only the speaker/AEC clock.
                        epoch = self._abort_epoch
                    self._next_write_commit_sequence += 1
                    commit_sequence = self._next_write_commit_sequence
                    self._active_write_commit_sequence = commit_sequence
                    self._write_in_progress_epoch = epoch
                    self._active_write_has_payload = has_payload
                    self._hardware_write_started_ns = time.monotonic_ns()
                    self._hardware_write_kind = (
                        "pcm" if has_payload else "aec_silence"
                    )
                    prewrite_commit_ns = self._hardware_write_started_ns
                    commit_audit_stamps = tuple(
                        stamp
                        for stamp in timing_stamps
                        if stamp not in self._timing_committed_stamps
                    )
                    self._timing_committed_stamps.update(commit_audit_stamps)
                    if self._timing is not None:
                        for stamp in commit_audit_stamps:
                            self._timing.mark(
                                stamp,
                                "playback_prewrite_commit",
                                at_ns=prewrite_commit_ns,
                                generation=epoch,
                                commit_sequence=commit_sequence,
                            )
                    recovery_write = bool(
                        has_payload and self._pending_recovery_epoch == epoch
                    )
                    if recovery_write:
                        logger.info(
                            "PLAYBACK_HW_WRITE_BEGIN generation=%d latency_ms=%.3f "
                            "stream_active=%s bytes=%d",
                            epoch,
                            self._since_interrupt_ms_locked(time.monotonic_ns()),
                            self._stream_active(stream),
                            len(output),
                        )
                submission = self._submit_hardware_write(
                    stream,
                    output,
                    reference,
                    epoch,
                    commit_sequence,
                )
                if submission is None:
                    continue
                underflowed, write_started_ns, write_finished_ns = submission
                write_audit_stamps = tuple(
                    stamp
                    for stamp in timing_stamps
                    if stamp not in self._timing_written_stamps
                )
                self._timing_written_stamps.update(write_audit_stamps)
                if self._timing is not None:
                    for stamp in write_audit_stamps:
                        self._timing.mark(
                            stamp,
                            "hardware_write_begin",
                            at_ns=write_started_ns,
                            generation=epoch,
                            bytes=len(output),
                        )
                        self._timing.mark(
                            stamp,
                            "hardware_write_complete",
                            at_ns=write_finished_ns,
                            generation=epoch,
                            underflow=underflowed,
                        )
                write_ms = (write_finished_ns - write_started_ns) / 1_000_000
                with self._condition:
                    self.last_write_ms = round(write_ms, 3)
                    self.max_write_ms = max(self.max_write_ms, write_ms)
                    if has_payload:
                        self.blocks_written += 1
                        self.bytes_written += len(output)
                    else:
                        self.render_clock_silence_blocks += 1
                    if underflowed:
                        self.write_underflows += 1
                    if has_payload and epoch != self._abort_epoch:
                        self.residual_write_completions += 1
                    if recovery_write and self._pending_recovery_epoch == epoch:
                        recovery_ms = self._since_interrupt_ms_locked(write_finished_ns)
                        self.last_recovery_ms = round(recovery_ms, 3)
                        self.max_recovery_ms = max(self.max_recovery_ms, recovery_ms)
                        self.recoveries += 1
                        self._pending_recovery_epoch = None
                        logger.info(
                            "PLAYBACK_RECOVERED generation=%d latency_ms=%.3f "
                            "write_ms=%.3f underflow=%s stream_active=%s",
                            epoch,
                            recovery_ms,
                            write_ms,
                            underflowed,
                            self._stream_active(stream),
                        )
                    self._record_stream_active_locked(stream, "write-complete")
            except Exception:  # noqa: BLE001
                with self._condition:
                    interrupted = bool(
                        (has_payload and epoch != self._abort_epoch) or self._stop
                    )
                if not interrupted:
                    self.playback_errors += 1
                    self.write_errors += 1
                    logger.exception("ALSA continuous playback failed")
            finally:
                with self._condition:
                    if self._write_in_progress_epoch == epoch:
                        self._write_in_progress_epoch = None
                    if (
                        commit_sequence is not None
                        and self._active_write_commit_sequence == commit_sequence
                    ):
                        self._active_write_commit_sequence = None
                        self._active_write_has_payload = False
                    self._hardware_write_started_ns = None
                    self._hardware_write_kind = None
                    self._inflight_samples = 0
                    if not self._ring.size:
                        self._drain_requested = False
                    self._condition.notify_all()

    def _submit_hardware_write(
        self,
        stream: Any,
        output: bytes,
        reference: np.ndarray,
        generation: int,
        commit_sequence: int,
    ) -> tuple[bool, int, int] | None:
        """Submit only a block holding the exact token minted by the final gate."""

        with self._condition:
            authorized = (
                self._active_write_commit_sequence == commit_sequence
                and self._write_in_progress_epoch == generation
            )
            if not authorized:
                self.forbidden_stale_write_attempts += 1
                logger.error(
                    "PLAYBACK_FORBIDDEN_STALE_WRITE_ATTEMPT generation=%d "
                    "sequence=%d active_commit=%s current_generation=%d",
                    generation,
                    commit_sequence,
                    self._active_write_commit_sequence,
                    self._abort_epoch,
                )
                return None

        # Invalidation after authorization is a committed residual write. The
        # commit remains valid, and its render reference must accompany the PCM.
        self._render_sink.push_render(reference, self._settings.sample_rate)
        write_started_ns = time.monotonic_ns()
        underflowed = bool(stream.write(output))
        write_finished_ns = time.monotonic_ns()
        return underflowed, write_started_ns, write_finished_ns

    @staticmethod
    def _stream_active(stream: Any | None) -> bool | None:
        if stream is None:
            return None
        try:
            return bool(getattr(stream, "active", True))
        except Exception:  # noqa: BLE001
            return None

    def _record_stream_active_locked(self, stream: Any, source: str) -> None:
        active = self._stream_active(stream)
        if active == self._last_stream_active:
            return
        previous = self._last_stream_active
        self._last_stream_active = active
        if previous is not None:
            self.stream_active_transitions += 1
        logger.info(
            "PLAYBACK_STREAM_STATE source=%s previous=%s active=%s",
            source,
            previous,
            active,
        )

    def _since_interrupt_ms_locked(self, now_ns: int) -> float:
        if self._interrupt_started_ns is None:
            return 0.0
        return (now_ns - self._interrupt_started_ns) / 1_000_000

    def _event_latency_ms_locked(self, event_ns: int | None) -> float | None:
        if event_ns is None or self._interrupt_started_ns is None:
            return None
        return round(self._since_interrupt_ms_locked(event_ns), 3)

    def _append_timing_span_locked(
        self,
        sample_count: int,
        stamp: ControlStamp | None,
    ) -> None:
        if sample_count <= 0:
            return
        if self._timing_spans and self._timing_spans[-1][1] == stamp:
            previous_count, _ = self._timing_spans.pop()
            self._timing_spans.append((previous_count + sample_count, stamp))
            return
        self._timing_spans.append((sample_count, stamp))

    def _consume_timing_spans_locked(self, sample_count: int) -> tuple[ControlStamp, ...]:
        remaining = sample_count
        stamps: list[ControlStamp] = []
        while remaining > 0 and self._timing_spans:
            span_count, stamp = self._timing_spans.popleft()
            consumed = min(remaining, span_count)
            remaining -= consumed
            leftover = span_count - consumed
            if leftover:
                self._timing_spans.appendleft((leftover, stamp))
            if stamp is not None and stamp not in stamps:
                stamps.append(stamp)
        return tuple(stamps)

    def _prepare_block(self, pcm: bytes) -> tuple[bytes, np.ndarray]:
        mono = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
        mono *= self._settings.volume
        mono_i16 = np.rint(np.clip(mono, -32768, 32767)).astype("<i2")
        reference = mono_i16.astype(np.float32) / 32768.0
        if self._settings.channels == 1:
            return mono_i16.tobytes(), reference
        stereo = np.repeat(mono_i16[:, None], self._settings.channels, axis=1)
        return stereo.astype("<i2", copy=False).tobytes(), reference
