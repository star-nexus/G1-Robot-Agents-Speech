"""Memory provider port and allocation-free bounded implementations."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Protocol

from .contracts import ChatMessage


class MemoryProvider(Protocol):
    """Conversation memory boundary; durable memory can implement the same port."""

    def context_messages(self) -> Sequence[ChatMessage]: ...

    def record_turn(self, user_text: str, assistant_text: str) -> None: ...

    def clear(self) -> None: ...


class NullMemory:
    """No-op memory for stateless characters and controlled benchmarks."""

    def context_messages(self) -> tuple[ChatMessage, ...]:
        return ()

    def record_turn(self, user_text: str, assistant_text: str) -> None:
        return None

    def clear(self) -> None:
        return None


class WindowMemory:
    """Keeps only the last N completed user/assistant turns in host memory."""

    def __init__(self, max_turns: int = 4) -> None:
        if max_turns < 1:
            raise ValueError("WindowMemory max_turns must be greater than zero")
        self.max_turns = max_turns
        self._messages: deque[ChatMessage] = deque(maxlen=max_turns * 2)

    def context_messages(self) -> tuple[ChatMessage, ...]:
        return tuple(self._messages)

    def record_turn(self, user_text: str, assistant_text: str) -> None:
        self._messages.append({"role": "user", "content": user_text})
        self._messages.append({"role": "assistant", "content": assistant_text})

    def clear(self) -> None:
        self._messages.clear()
