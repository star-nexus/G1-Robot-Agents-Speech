"""Transport- and embodiment-independent STAR Agent kernel."""

from ..capabilities import (
    CapabilityCompatibility,
    CapabilityContext,
    CapabilityPermission,
    CapabilityProvider,
    CapabilityRegistry,
    CapabilitySpec,
)
from .knowledge import KnowledgeProvider, KeywordLoreProvider, NullKnowledge
from .llama import LlamaChatClient, LlamaChatSettings
from .loop import ConversationalLoop
from .memory import MemoryProvider, NullMemory, WindowMemory
from .role import RolePackage, load_role_package
from .runtime import AgentRuntime
from .voice_bridge import VoiceBridgeAdapter, VoiceBridgeSettings

__all__ = [
    "AgentRuntime",
    "CapabilityContext",
    "CapabilityCompatibility",
    "CapabilityPermission",
    "CapabilityProvider",
    "CapabilityRegistry",
    "CapabilitySpec",
    "ConversationalLoop",
    "KnowledgeProvider",
    "KeywordLoreProvider",
    "LlamaChatClient",
    "LlamaChatSettings",
    "MemoryProvider",
    "NullKnowledge",
    "NullMemory",
    "RolePackage",
    "VoiceBridgeAdapter",
    "VoiceBridgeSettings",
    "WindowMemory",
    "load_role_package",
]
