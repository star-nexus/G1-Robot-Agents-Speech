"""Canonical Session -> Turn -> Epoch validity authority.

The owner lives on the speech side because VAD is the first component to know
about barge-in.  Other processes keep a replica for prompt cancellation only;
the owner's ``is_current`` check is always the final correctness guard.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, replace

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ControlStamp:
    session_id: str
    turn_id: str
    epoch: int

    @property
    def scoped(self) -> bool:
        return bool(self.session_id and self.turn_id and self.epoch > 0)


@dataclass(frozen=True)
class EpochInvalidated:
    event_id: str
    session_id: str
    turn_id: str
    epoch: int
    next_epoch: int
    reason: str
    created_unix_ns: int
    source: str = "star-runtime-control"

    @property
    def stamp(self) -> ControlStamp:
        return ControlStamp(self.session_id, self.turn_id, self.epoch)


class RuntimeControlPlane:
    """Thread-safe owner of one session's current turn and epoch."""

    def __init__(self, session_id: str | None = None) -> None:
        self.session_id = session_id or uuid.uuid4().hex
        self._current: ControlStamp | None = None
        self._completed: ControlStamp | None = None
        self._lock = threading.RLock()
        self.invalidations = 0
        self.stale_drops = 0

    def begin_turn(self, turn_id: str | None = None) -> ControlStamp:
        with self._lock:
            stamp = ControlStamp(
                self.session_id,
                turn_id or uuid.uuid4().hex,
                1,
            )
            self._current = stamp
            self._completed = None
        logger.info(
            "Control turn started: session_id=%s turn_id=%s epoch=%d",
            stamp.session_id,
            stamp.turn_id,
            stamp.epoch,
        )
        return stamp

    def invalidate(self, reason: str) -> EpochInvalidated | None:
        with self._lock:
            old = self._current
            if old is None or self._completed == old:
                return None
            new = replace(old, epoch=old.epoch + 1)
            self._current = new
            self._completed = None
            self.invalidations += 1
        event = EpochInvalidated(
            event_id=uuid.uuid4().hex,
            session_id=old.session_id,
            turn_id=old.turn_id,
            epoch=old.epoch,
            next_epoch=new.epoch,
            reason=reason,
            created_unix_ns=time.time_ns(),
        )
        logger.info(
            "Control epoch invalidated: session_id=%s turn_id=%s epoch=%d "
            "next_epoch=%d reason=%s",
            old.session_id,
            old.turn_id,
            old.epoch,
            new.epoch,
            reason,
        )
        return event

    def is_current(self, stamp: ControlStamp) -> bool:
        with self._lock:
            return stamp.scoped and self._current == stamp

    def is_active(self, stamp: ControlStamp) -> bool:
        with self._lock:
            return (
                stamp.scoped
                and self._current == stamp
                and self._completed != stamp
            )

    def note_stale(self, stage: str, stamp: ControlStamp) -> None:
        with self._lock:
            self.stale_drops += 1
            current = self._current
        logger.info(
            "Control stale work discarded: stage=%s session_id=%s turn_id=%s "
            "epoch=%d current=%s",
            stage,
            stamp.session_id,
            stamp.turn_id,
            stamp.epoch,
            current,
        )

    def complete(self, stamp: ControlStamp) -> bool:
        """Mark the current epoch quiescent after its final playback edge."""

        with self._lock:
            if self._current != stamp:
                return False
            self._completed = stamp
            return True

    @property
    def active(self) -> bool:
        with self._lock:
            return self._current is not None and self._completed != self._current

    @property
    def current(self) -> ControlStamp | None:
        with self._lock:
            return self._current


class ControlPlaneReplica:
    """Agent-side current-stamp replica used to stop provider work early."""

    def __init__(self) -> None:
        self._current: ControlStamp | None = None
        self._lock = threading.Lock()

    def observe_turn(self, stamp: ControlStamp) -> None:
        with self._lock:
            self._current = stamp

    def apply(self, event: EpochInvalidated) -> bool:
        with self._lock:
            if self._current != event.stamp:
                return False
            self._current = ControlStamp(
                event.session_id,
                event.turn_id,
                event.next_epoch,
            )
            return True

    def is_current(self, stamp: ControlStamp) -> bool:
        with self._lock:
            return self._current == stamp

    @property
    def current(self) -> ControlStamp | None:
        with self._lock:
            return self._current
