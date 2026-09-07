"""Playback-aware microphone gate used to prevent robot self-hearing."""

from __future__ import annotations

import threading
import time
from typing import Callable

from ..core.control import ControlStamp
from ..core.events import PlaybackState


class PlaybackGate:
    def __init__(self, *, resume_delay_ms: int = 250, max_active_seconds: float = 30.0) -> None:
        self._resume_delay_ns = resume_delay_ms * 1_000_000
        self._max_active_ns = round(max_active_seconds * 1_000_000_000)
        self._active = False
        self._active_since_ns = 0
        self._muted_until_ns = 0
        self._lock = threading.Lock()
        self._active_stamp: ControlStamp | None = None
        self._validator: Callable[[ControlStamp], bool] | None = None

    def set_validator(self, validator: Callable[[ControlStamp], bool]) -> None:
        self._validator = validator

    def handle_state(self, state: PlaybackState | bool) -> None:
        if isinstance(state, bool):
            self.set_active(state)
            return
        stamp = state.control_stamp
        now = time.monotonic_ns()
        if state.active:
            if self._validator is not None and not self._validator(stamp):
                return
            with self._lock:
                self._active = True
                self._active_since_ns = now
                self._muted_until_ns = 0
                self._active_stamp = stamp
            return
        with self._lock:
            if self._active_stamp is not None and self._active_stamp != stamp:
                return
            self._active = False
            self._active_since_ns = 0
            self._muted_until_ns = now + self._resume_delay_ns
            self._active_stamp = None

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
                self._active_stamp = None

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
