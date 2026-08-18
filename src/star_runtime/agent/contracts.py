"""Small provider contracts shared by the STAR Agent runtime."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, Protocol, TypeAlias, TypedDict

ChatContentPart: TypeAlias = dict[str, Any]
ChatContent: TypeAlias = str | list[ChatContentPart]


class ChatMessage(TypedDict):
    """OpenAI-compatible message shared by text and multimodal providers."""

    role: str
    content: ChatContent


class ChatModelProvider(Protocol):
    """Streams spoken answer text for an already assembled chat request."""

    def stream(self, messages: list[ChatMessage]) -> Iterator[str]: ...


class AgentLoop(Protocol):
    """Strategy port for a conversational, OODA, or graph-based Agent loop."""

    @property
    def name(self) -> str: ...

    def stream_response(self, user_text: str) -> Iterator[str]: ...

    def commit_turn(self, user_text: str, assistant_text: str) -> None: ...

    def context_messages(self) -> Sequence[ChatMessage]: ...
