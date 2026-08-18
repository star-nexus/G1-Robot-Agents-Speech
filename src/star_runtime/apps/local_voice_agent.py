"""Local LLM voice Agent composed with a selectable transport backend."""

from __future__ import annotations

import signal
import threading
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from star_runtime.agent import (
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
from star_runtime.agent.contracts import ChatMessage
from star_runtime.perception import VisionInput
from star_runtime.robots import ActiveRobot, RobotAdapterCatalog
from star_runtime.agent.llama import DEFAULT_REASONING_BUDGET_MESSAGE
from star_runtime.agent.voice_bridge import VoiceBridgeAdapter, VoiceBridgeSettings
from star_runtime.speech.config import ServiceConfig
from star_runtime.transports.contracts import SpeechInputPort, SpeechOutputPort


logger = logging.getLogger(__name__)


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
    vision: VisionInput | None = None,
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
        vision=vision,
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
    """Local Agent facade around transport-neutral voice ports."""

    def __init__(
        self,
        config: ServiceConfig,
        settings: LocalAgentSettings,
        *,
        client: Any | None = None,
        publisher: SpeechOutputPort | None = None,
        subscriber: SpeechInputPort | None = None,
        role_package: RolePackage | None = None,
        capabilities: CapabilityRegistry | None = None,
        vision: VisionInput | None = None,
    ) -> None:
        self._agent_settings = settings
        runtime = build_agent_runtime(
            settings,
            client=client,
            role_package=role_package,
            capabilities=capabilities,
            vision=vision,
        )
        if (publisher is None) != (subscriber is None):
            raise ValueError("publisher and subscriber must be provided together")
        if publisher is None or subscriber is None:
            publisher, subscriber = _create_voice_ports(config)
        super().__init__(
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
    def _turns(self) -> list[ChatMessage]:
        """Legacy test/diagnostic view; memory ownership lives in the runtime."""

        return list(self.runtime.context_messages())

    def start(self) -> None:
        super().start()
        settings = self._agent_settings
        logger.info(
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
    robot_adapter_id: str | None = None,
    robot_catalog: RobotAdapterCatalog | None = None,
    vision: VisionInput | None = None,
) -> int:
    capabilities = CapabilityRegistry(
        role_package.capabilities if role_package is not None else None
    )
    active_robot = _activate_robot(
        capabilities,
        adapter_id=robot_adapter_id,
        catalog=robot_catalog,
    )
    try:
        if config.transport.backend == "dds":
            from star_runtime.transports.dds.voice import initialize_dds

            initialize_dds(config.dds.domain_id, config.dds.network_interface)
        agent = LocalVoiceAgent(
            config,
            settings,
            role_package=role_package,
            capabilities=capabilities,
            vision=vision,
        )
    except Exception:
        if active_robot is not None:
            active_robot.close()
        raise
    stopped = threading.Event()

    def stop(*_args: Any) -> None:
        stopped.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        agent.start()
        while not stopped.wait(0.2):
            pass
    finally:
        agent.close()
        if active_robot is not None:
            active_robot.close()
    return 0


def _activate_robot(
    capabilities: CapabilityRegistry,
    *,
    adapter_id: str | None,
    catalog: RobotAdapterCatalog | None,
) -> ActiveRobot | None:
    """Preflight role requirements before any vendor SDK is activated."""

    selected_id = adapter_id
    if selected_id is None and not capabilities.permission.required:
        capabilities.ensure_compatible()
        return None

    selected_catalog = catalog or RobotAdapterCatalog.discover()
    if selected_id is None and capabilities.permission.required:
        compatible = selected_catalog.compatible_adapter_ids(capabilities.permission)
        if not compatible:
            raise RuntimeError(
                "no installed robot adapter satisfies required capabilities: "
                f"{sorted(capabilities.permission.required)!r}"
            )
        if len(compatible) > 1:
            raise RuntimeError(
                "multiple robot adapters satisfy the role; select one explicitly: "
                f"{list(compatible)!r}"
            )
        selected_id = compatible[0]
    return selected_catalog.activate(selected_id, capabilities)


def _create_voice_ports(
    config: ServiceConfig,
) -> tuple[SpeechOutputPort, SpeechInputPort]:
    """Create concrete ears/mouth ports without leaking them into Agent core."""

    backend = config.transport.backend
    if backend == "dds":
        from star_runtime.transports.dds.voice import (
            DdsSpeechSubscriber,
            DdsTtsPublisher,
        )

        publisher = DdsTtsPublisher(
            topic=config.dds.tts_topic,
            source="star-agent-runtime",
        )
        subscriber = DdsSpeechSubscriber(
            lambda _event: None,
            topic=config.dds.speech_topic,
        )
        return publisher, subscriber
    if backend == "ros2":
        from star_runtime.transports.ros2.voice import Ros2VoicePort

        port = Ros2VoicePort(
            config.ros2,
            node_name=f"{config.ros2.node_name}_agent",
        )
        return port, port
    raise ValueError(f"unsupported Agent voice transport: {backend}")


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
