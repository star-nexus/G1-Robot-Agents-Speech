"""Playback-aware microphone gate used to prevent the robot hearing itself."""

from __future__ import annotations

import threading
import time


class PlaybackGate:
    def __init__(self, *, resume_delay_ms: int = 250, max_active_seconds: float = 30.0) -> None:
        self._resume_delay_ns = resume_delay_ms * 1_000_000
        self._max_active_ns = round(max_active_seconds * 1_000_000_000)
        self._active = False
        self._active_since_ns = 0
        self._muted_until_ns = 0
        self._lock = threading.Lock()

    def set_active(self, active: bool, *, now_ns: int | None = None) -> None:
        now = time.monotonic_ns() if now_ns is None else now_ns
        with self._lock:
            if active:
                self._active = True
                self._active_since_ns = now
                self._muted_until_ns = 0
            else:
                self._active = False
                self._active_since_ns = 0
                self._muted_until_ns = now + self._resume_delay_ns

    def is_muted(self, *, now_ns: int | None = None) -> bool:
        now = time.monotonic_ns() if now_ns is None else now_ns
        with self._lock:
            if self._active and now - self._active_since_ns > self._max_active_ns:
                self._active = False
                self._active_since_ns = 0
                self._muted_until_ns = now + self._resume_delay_ns
            return self._active or now < self._muted_until_ns

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active
