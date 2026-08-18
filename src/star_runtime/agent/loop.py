"""Low-overhead conversational loop for the STAR Agent runtime."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from .contracts import ChatMessage, ChatModelProvider
from .knowledge import KnowledgeProvider, NullKnowledge
from .memory import MemoryProvider
from ..perception import VisionInput


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
        vision: VisionInput | None = None,
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
        self._vision = vision

    def stream_response(self, user_text: str) -> Iterator[str]:
        return self._model.stream(self._messages_for(user_text))

    def commit_turn(self, user_text: str, assistant_text: str) -> None:
        self._memory.record_turn(user_text, assistant_text)

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
        image = (
            self._vision.image_for(user_text, self._memory.context_messages())
            if self._vision is not None
            else None
        )
        if image is None:
            content = user_text
        else:
            content = [
                {"type": "image_url", "image_url": {"url": image.data_url()}},
                {
                    "type": "text",
                    "text": (
                        "以上是机器人摄像头刚刚捕获的当前画面。"
                        f"请结合画面回答用户的语音：{user_text}"
                    ),
                },
            ]
        messages.append({"role": "user", "content": content})
        return messages
