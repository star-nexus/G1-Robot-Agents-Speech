"""Transport-neutral ports connecting speech systems to an Agent."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from ..core.control import EpochInvalidated
from ..core.events import PlaybackState, SpeechEvent


@dataclass(frozen=True)
class PublishResult:
    """Backend-neutral result of handing one message to a transport.

    ``accepted`` means the local adapter accepted the message. Remote delivery
    is deliberately reported separately because ROS 2 publish is asynchronous.
    """

    accepted: bool
    matched: bool | None = None
    delivered: bool | None = None
    detail: str = ""

    def __bool__(self) -> bool:
        return self.accepted


class EventSink(Protocol):
    def start(self) -> None: ...

    def publish(self, event: SpeechEvent) -> bool | PublishResult: ...

    def close(self) -> None: ...


class SpeechTransport(Protocol):
    """Transport lifecycle consumed by the integrated speech runtime."""

    sink: EventSink

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...

    def metrics(self) -> dict[str, Any]: ...


class SpeechEventLike(Protocol):
    """Minimum event shape consumed by the voice bridge."""

    event_id: str
    session_id: str
    created_unix_ns: int
    text: str
    is_final: bool


class SpeechInputPort(Protocol):
    """Lifecycle of a transport-specific stream of speech events."""

    @property
    def endpoint(self) -> str: ...

    def set_handler(self, handler: Callable[[SpeechEventLike], None]) -> None: ...

    def set_control_handler(
        self, handler: Callable[[EpochInvalidated], None]
    ) -> None: ...

    def start(self) -> None: ...

    def close(self) -> None: ...


class SpeechOutputPort(Protocol):
    """Transport-neutral incremental text output consumed by a robot mouth."""

    @property
    def endpoint(self) -> str: ...

    def start(self) -> None: ...

    def publish(
        self,
        *,
        request_id: str,
        sequence: int,
        text: str,
        is_final: bool = False,
        interrupt: bool = False,
        language: str = "",
        voice: str = "",
        instructions: str = "",
        session_id: str = "",
        turn_id: str = "",
        epoch: int = 0,
        timeout: float = 0.25,
    ) -> bool | PublishResult: ...

    def close(self) -> None: ...
