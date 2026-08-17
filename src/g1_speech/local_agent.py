"""Compatibility entrypoint for the local Agent Runtime voice application."""

from __future__ import annotations

import signal
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent_runtime import (
    AgentRuntime,
    CapabilityRegistry,
    ConversationalLoop,
    LlamaChatClient,
    LlamaChatSettings,
    KnowledgeProvider,
    KeywordLoreProvider,
    MemoryProvider,
    NullMemory,
    RolePackage,
    WindowMemory,
)
from .agent_runtime.llama import DEFAULT_REASONING_BUDGET_MESSAGE
from .agent_runtime.voice_bridge import VoiceBridgeAdapter, VoiceBridgeSettings
from .config import ServiceConfig
from .dds import DdsSpeechSubscriber, DdsTtsPublisher, initialize_dds


DEFAULT_SYSTEM_PROMPT = (
    "你是运行在机器人本机的交互大脑。回答要自然、简洁、适合直接朗读。"
    "除非用户明确要求详细说明，每次回答不超过30个中文字符。"
    "不要使用Markdown、列表、表情或舞台提示。"
)


def load_role_prompt(path: str | Path) -> str:
    """Load the legacy single-file role format.

    New characters should use a versioned Role Package. Keeping this loader
    prevents existing deployments from breaking during migration.
    """

    role_path = Path(path).expanduser().resolve()
    prompt = role_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"role file is empty: {role_path}")
    return prompt


@dataclass(frozen=True)
class LocalAgentSettings(LlamaChatSettings):
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    history_turns: int = 4
    max_speech_age_seconds: float = 5.0
    tts_voice: str = ""
    tts_language: str = ""
    tts_instructions: str = ""

    def validate(self) -> None:
        super().validate()
        if self.history_turns < 0:
            raise ValueError("history_turns must not be negative")
        if self.max_speech_age_seconds <= 0:
            raise ValueError("max_speech_age_seconds must be greater than zero")


def build_agent_runtime(
    settings: LocalAgentSettings,
    *,
    client: Any | None = None,
    role_package: RolePackage | None = None,
    memory: MemoryProvider | None = None,
    capabilities: CapabilityRegistry | None = None,
    knowledge: KnowledgeProvider | None = None,
) -> AgentRuntime:
    """Construct the light runtime once; nothing here runs per generated token."""

    settings.validate()
    selected_memory = memory if memory is not None else (
        WindowMemory(settings.history_turns)
        if settings.history_turns > 0
        else NullMemory()
    )
    role_id = role_package.role_id if role_package else "local.default"
    permissions = role_package.capabilities if role_package else None
    selected_knowledge = knowledge
    if selected_knowledge is None and role_package is not None:
        role_knowledge = role_package.knowledge
        if role_knowledge.provider == "keyword":
            if role_knowledge.cards_path is None:
                raise ValueError("keyword knowledge cards path is missing")
            selected_knowledge = KeywordLoreProvider.from_files(
                core_path=role_knowledge.core_path,
                cards_path=role_knowledge.cards_path,
                max_cards=role_knowledge.max_cards,
            )
    loop = ConversationalLoop(
        system_prompt=settings.system_prompt,
        model=client or LlamaChatClient(settings),
        memory=selected_memory,
        knowledge=selected_knowledge,
    )
    return AgentRuntime(
        agent_id=f"robot:{role_id}",
        role_id=role_id,
        loop=loop,
        capabilities=(
            capabilities
            if capabilities is not None
            else CapabilityRegistry(permissions)
        ),
    )


class LocalVoiceAgent(VoiceBridgeAdapter):
    """Backward-compatible facade around AgentRuntime + VoiceBridgeAdapter."""

    def __init__(
        self,
        config: ServiceConfig,
        settings: LocalAgentSettings,
        *,
        client: Any | None = None,
        publisher: DdsTtsPublisher | None = None,
        subscriber: DdsSpeechSubscriber | None = None,
        role_package: RolePackage | None = None,
    ) -> None:
        self._agent_settings = settings
        runtime = build_agent_runtime(
            settings,
            client=client,
            role_package=role_package,
        )
        super().__init__(
            config,
            runtime,
            VoiceBridgeSettings(
                max_speech_age_seconds=settings.max_speech_age_seconds,
                tts_voice=settings.tts_voice,
                tts_language=settings.tts_language,
                tts_instructions=settings.tts_instructions,
            ),
            publisher=publisher,
            subscriber=subscriber,
        )

    @property
    def _turns(self) -> list[dict[str, str]]:
        """Legacy test/diagnostic view; memory ownership lives in the runtime."""

        return list(self.runtime.context_messages())

    def start(self) -> None:
        super().start()
        settings = self._agent_settings
        import logging

        logging.getLogger(__name__).info(
            "Local Agent Runtime ready: role=%s loop=%s speech_topic=%s "
            "tts_topic=%s llama=%s tts_voice=%r tts_language=%r "
            "history_turns=%d max_tokens=%d thinking=%s thinking_budget=%d",
            self.runtime.role_id,
            self.runtime.loop.name,
            self.subscriber_topic(),
            self.publisher_topic(),
            settings.url,
            settings.tts_voice or "profile-default",
            settings.tts_language or "profile-default",
            settings.history_turns,
            settings.max_tokens,
            settings.enable_thinking,
            settings.reasoning_budget_tokens,
        )


def run_local_voice_agent(
    config: ServiceConfig,
    settings: LocalAgentSettings,
    *,
    role_package: RolePackage | None = None,
) -> int:
    initialize_dds(config.dds.domain_id, config.dds.network_interface)
    agent = LocalVoiceAgent(config, settings, role_package=role_package)
    stopped = threading.Event()

    def stop(*_args: Any) -> None:
        stopped.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    agent.start()
    try:
        while not stopped.wait(0.2):
            pass
    finally:
        agent.close()
    return 0


__all__ = [
    "DEFAULT_REASONING_BUDGET_MESSAGE",
    "DEFAULT_SYSTEM_PROMPT",
    "LlamaChatClient",
    "LocalAgentSettings",
    "LocalVoiceAgent",
    "build_agent_runtime",
    "load_role_prompt",
    "run_local_voice_agent",
]
