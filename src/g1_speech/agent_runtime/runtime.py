"""Agent Runtime kernel; loop strategy and providers remain replaceable."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from .capabilities import CapabilityContext, CapabilityRegistry
from .contracts import AgentLoop, ChatMessage


class AgentRuntime:
    """Owns identity and orchestration without owning model or transport code."""

    def __init__(
        self,
        *,
        agent_id: str,
        role_id: str,
        loop: AgentLoop,
        capabilities: CapabilityRegistry | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.role_id = role_id
        self.loop = loop
        self.capabilities = capabilities or CapabilityRegistry()

    def stream_response(self, user_text: str) -> Iterator[str]:
        # Return the loop iterator directly: no per-token wrapper or allocation.
        return self.loop.stream_response(user_text)

    def commit_turn(self, user_text: str, assistant_text: str) -> None:
        self.loop.commit_turn(user_text, assistant_text)

    def context_messages(self) -> Sequence[ChatMessage]:
        return self.loop.context_messages()

    def invoke_capability(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        session_id: str = "",
    ) -> Any:
        return self.capabilities.invoke(
            name,
            arguments,
            CapabilityContext(
                agent_id=self.agent_id,
                role_id=self.role_id,
                session_id=session_id,
            ),
        )
