"""Low-overhead, per-turn Runtime timing audit.

The audit records observed monotonic timestamps only.  It never estimates a
missing stage and never participates in control or data-plane decisions.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .control import ControlStamp

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TimingEvent:
    name: str
    monotonic_ns: int
    fields: dict[str, Any] = field(default_factory=dict)


@dataclass
class _TurnRecord:
    stamp: ControlStamp
    events: dict[str, TimingEvent] = field(default_factory=dict)
    emitted: bool = False


class RuntimeTimingAudit:
    """Thread-safe first-observation recorder keyed by exact ControlStamp."""

    EXPECTED_EVENTS = (
        "speech_start",
        "speech_end",
        "vad_ready",
        "asr_start",
        "asr_final",
        "turn_started",
        "speech_event_published",
        "agent_start",
        "agent_first_delta",
        "agent_first_delta_published",
        "tts_segment_ready",
        "tts_segment_enqueued",
        "playback_active",
        "tts_synthesis_start",
        "tts_first_pcm",
        "player_enqueue_begin",
        "player_enqueue_complete",
        "playback_dequeue",
        "playback_prewrite_commit",
        "hardware_write_begin",
        "hardware_write_complete",
        "agent_complete",
        "playback_complete",
    )

    _DURATION_PAIRS = {
        "speech_duration_ms": ("speech_start", "speech_end"),
        "vad_tail_ms": ("speech_end", "vad_ready"),
        "asr_queue_ms": ("vad_ready", "asr_start"),
        "asr_inference_ms": ("asr_start", "asr_final"),
        "speech_end_to_asr_final_ms": ("speech_end", "asr_final"),
        "vad_edge_to_invalidation_ms": ("vad_speech_edge", "invalidated"),
        "vad_edge_to_output_preempt_ms": (
            "vad_speech_edge",
            "vad_output_preempted",
        ),
        "vad_edge_to_agent_cancel_ms": (
            "vad_speech_edge",
            "agent_cancel_requested",
        ),
        "asr_final_to_agent_start_ms": ("asr_final", "agent_start"),
        "agent_first_delta_ms": ("agent_start", "agent_first_delta"),
        "first_delta_publish_delay_ms": (
            "agent_first_delta",
            "agent_first_delta_published",
        ),
        "first_delta_to_tts_segment_ms": (
            "agent_first_delta",
            "tts_segment_ready",
        ),
        "tts_queue_ms": ("tts_segment_enqueued", "tts_synthesis_start"),
        "tts_first_pcm_ms": ("tts_synthesis_start", "tts_first_pcm"),
        "pcm_to_player_enqueue_ms": ("tts_first_pcm", "player_enqueue_begin"),
        "player_enqueue_ms": ("player_enqueue_begin", "player_enqueue_complete"),
        "enqueue_to_dequeue_ms": ("player_enqueue_begin", "playback_dequeue"),
        "dequeue_to_prewrite_commit_ms": (
            "playback_dequeue",
            "playback_prewrite_commit",
        ),
        "prewrite_to_hardware_write_ms": (
            "playback_prewrite_commit",
            "hardware_write_begin",
        ),
        "hardware_write_ms": ("hardware_write_begin", "hardware_write_complete"),
        "speech_end_to_first_hardware_write_ms": (
            "speech_end",
            "hardware_write_begin",
        ),
        "asr_final_to_first_hardware_write_ms": (
            "asr_final",
            "hardware_write_begin",
        ),
        "agent_total_ms": ("agent_start", "agent_complete"),
        "turn_playback_ms": ("playback_active", "playback_complete"),
    }

    def __init__(self, *, retained_turns: int = 256) -> None:
        if retained_turns < 1:
            raise ValueError("retained_turns must be positive")
        self._retained_turns = retained_turns
        self._records: dict[ControlStamp, _TurnRecord] = {}
        self._order: deque[ControlStamp] = deque()
        self._lock = threading.Lock()
        self.timelines_emitted = 0

    def mark(
        self,
        stamp: ControlStamp,
        name: str,
        *,
        at_ns: int | None = None,
        **fields: Any,
    ) -> bool:
        """Record a stage once; return False for duplicates or finished turns."""

        if not stamp.scoped or not name:
            return False
        event = TimingEvent(
            name=name,
            monotonic_ns=time.monotonic_ns() if at_ns is None else int(at_ns),
            fields={key: value for key, value in fields.items() if value is not None},
        )
        with self._lock:
            record = self._record_locked(stamp)
            if record.emitted or name in record.events:
                return False
            record.events[name] = event
            return True

    def finish(
        self,
        stamp: ControlStamp,
        outcome: str,
        *,
        at_ns: int | None = None,
        **fields: Any,
    ) -> dict[str, Any] | None:
        """Emit exactly one structured timeline for a terminal turn outcome."""

        if not stamp.scoped:
            return None
        if at_ns is not None:
            self.mark(stamp, "audit_finished", at_ns=at_ns, **fields)
        with self._lock:
            record = self._record_locked(stamp)
            if record.emitted:
                return None
            record.emitted = True
            payload = self._payload_locked(record, outcome, fields)
            self.timelines_emitted += 1
            self._prune_locked()
        logger.info(
            "RUNTIME_TURN_TIMELINE %s",
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        )
        return payload

    def finish_if_complete(self, stamp: ControlStamp) -> dict[str, Any] | None:
        """Finish only after both Agent generation and playback are terminal."""

        if not stamp.scoped:
            return None
        with self._lock:
            record = self._records.get(stamp)
            if (
                record is None
                or record.emitted
                or "agent_complete" not in record.events
                or "playback_complete" not in record.events
            ):
                return None
            record.emitted = True
            outcome = (
                "completed_with_errors"
                if "tts_error" in record.events
                else "completed"
            )
            payload = self._payload_locked(record, outcome, {})
            self.timelines_emitted += 1
            self._prune_locked()
        logger.info(
            "RUNTIME_TURN_TIMELINE %s",
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        )
        return payload

    def snapshot(self, stamp: ControlStamp) -> dict[str, TimingEvent]:
        """Return a test/diagnostic copy without completing the timeline."""

        with self._lock:
            record = self._records.get(stamp)
            return {} if record is None else dict(record.events)

    def has_event(self, stamp: ControlStamp, name: str) -> bool:
        with self._lock:
            record = self._records.get(stamp)
            return record is not None and name in record.events

    def _record_locked(self, stamp: ControlStamp) -> _TurnRecord:
        record = self._records.get(stamp)
        if record is None:
            record = _TurnRecord(stamp)
            self._records[stamp] = record
            self._order.append(stamp)
        return record

    def _payload_locked(
        self,
        record: _TurnRecord,
        outcome: str,
        finish_fields: dict[str, Any],
    ) -> dict[str, Any]:
        ordered = sorted(
            record.events.values(),
            key=lambda event: (event.monotonic_ns, event.name),
        )
        origin_ns = (
            record.events.get("speech_start", ordered[0]).monotonic_ns
            if ordered
            else time.monotonic_ns()
        )
        previous_ns = origin_ns
        timeline = []
        for event in ordered:
            item: dict[str, Any] = {
                "event": event.name,
                "at_ms": round((event.monotonic_ns - origin_ns) / 1_000_000, 3),
                "since_previous_ms": round(
                    (event.monotonic_ns - previous_ns) / 1_000_000,
                    3,
                ),
            }
            item.update(event.fields)
            timeline.append(item)
            previous_ns = event.monotonic_ns

        durations: dict[str, float] = {}
        for name, (start_name, end_name) in self._DURATION_PAIRS.items():
            start = record.events.get(start_name)
            end = record.events.get(end_name)
            if start is not None and end is not None:
                durations[name] = round(
                    (end.monotonic_ns - start.monotonic_ns) / 1_000_000,
                    3,
                )

        present = set(record.events)
        return {
            "schema": "star.runtime.turn_timing.v1",
            "session_id": record.stamp.session_id,
            "turn_id": record.stamp.turn_id,
            "epoch": record.stamp.epoch,
            "outcome": outcome,
            "logged_unix_ns": time.time_ns(),
            "origin_monotonic_ns": origin_ns,
            "timeline": timeline,
            "durations_ms": durations,
            "missing_events": [
                name for name in self.EXPECTED_EVENTS if name not in present
            ],
            "finish": {
                key: value for key, value in finish_fields.items() if value is not None
            },
        }

    def _prune_locked(self) -> None:
        while len(self._records) > self._retained_turns and self._order:
            stamp = self._order[0]
            record = self._records[stamp]
            if not record.emitted:
                return
            self._order.popleft()
            del self._records[stamp]
