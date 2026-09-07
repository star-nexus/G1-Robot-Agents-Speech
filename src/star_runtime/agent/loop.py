"""Low-overhead conversational loop for the STAR Agent runtime."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from .contracts import ChatMessage, ChatModelProvider
from .knowledge import KnowledgeProvider, NullKnowledge
from .memory import MemoryProvider


class ConversationalLoop:
    """One model call per user turn, preserving the existing latency path."""

    name = "conversational"

    def __init__(
        self,
        *,
        system_prompt: str,
        model: ChatModelProvider,
        memory: MemoryProvider,
        knowledge: KnowledgeProvider | None = None,
    ) -> None:
        self._knowledge = knowledge or NullKnowledge()
        core = self._knowledge.core_context().strip()
        self._system_prompt = (
            f"{system_prompt.rstrip()}\n\n# Canonical role knowledge\n\n{core}"
            if core
            else system_prompt
        )
        self._model = model
        self._memory = memory

    def stream_response(self, user_text: str) -> Iterator[str]:
        return self._model.stream(self._messages_for(user_text))

    def commit_turn(self, user_text: str, assistant_text: str) -> None:
        self._memory.record_turn(user_text, assistant_text)

    def cancel_active(self) -> bool:
        cancel = getattr(self._model, "cancel", None)
        return bool(cancel()) if cancel is not None else False

    def context_messages(self) -> Sequence[ChatMessage]:
        return self._memory.context_messages()

    def _messages_for(self, user_text: str) -> list[ChatMessage]:
        messages = [
            {"role": "system", "content": self._system_prompt},
            *self._memory.context_messages(),
        ]
        relevant = self._knowledge.relevant_context(user_text).strip()
        if relevant:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Canonical facts relevant to the current question follow. "
                        "Use them as authoritative and never mention retrieval:\n"
                        + relevant
                    ),
                }
            )
        messages.append({"role": "user", "content": user_text})
        return messages
