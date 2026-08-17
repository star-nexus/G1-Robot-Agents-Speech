"""Versioned, dependency-free Role Package loader."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .capabilities import CapabilityPermission

ROLE_PACKAGE_SCHEMA_VERSION = 1
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


@dataclass(frozen=True)
class RoleModelDefaults:
    max_tokens: int = 64
    temperature: float = 0.2
    thinking: bool = False
    thinking_budget_tokens: int = -1


@dataclass(frozen=True)
class RoleVoiceDefaults:
    voice: str = ""
    language: str = ""
    instructions: str = ""


@dataclass(frozen=True)
class RoleMemoryConfig:
    provider: str = "window"
    max_turns: int = 4


@dataclass(frozen=True)
class RoleKnowledgeConfig:
    provider: str = "none"
    core_path: Path | None = None
    cards_path: Path | None = None
    max_cards: int = 2


@dataclass(frozen=True)
class RolePackage:
    schema_version: int
    role_id: str
    version: str
    display_name: str
    prompt: str
    root: Path
    loop: str = "conversational"
    model: RoleModelDefaults = field(default_factory=RoleModelDefaults)
    voice: RoleVoiceDefaults = field(default_factory=RoleVoiceDefaults)
    memory: RoleMemoryConfig = field(default_factory=RoleMemoryConfig)
    knowledge: RoleKnowledgeConfig = field(default_factory=RoleKnowledgeConfig)
    capabilities: CapabilityPermission = field(default_factory=CapabilityPermission)


def load_role_package(path: str | Path) -> RolePackage:
    package_path = Path(path).expanduser().resolve()
    manifest_path = package_path / "role.json" if package_path.is_dir() else package_path
    if manifest_path.name != "role.json":
        raise ValueError("Role Package manifest must be named role.json")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Role Package manifest must be a JSON object")

    schema_version = _required_int(raw, "schema_version")
    if schema_version != ROLE_PACKAGE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported Role Package schema_version: {schema_version}"
        )
    role_id = _required_string(raw, "id")
    if not _IDENTIFIER.fullmatch(role_id):
        raise ValueError(f"invalid Role Package id: {role_id!r}")

    root = manifest_path.parent.resolve()
    prompt_name = _required_string(raw, "prompt")
    prompt_path = (root / prompt_name).resolve()
    if root not in prompt_path.parents or not prompt_path.is_file():
        raise ValueError("Role Package prompt must be a file inside the package")
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Role Package prompt is empty: {prompt_path}")

    model_raw = _object(raw, "model")
    voice_raw = _object(raw, "voice")
    memory_raw = _object(raw, "memory")
    capability_raw = _object(raw, "capabilities")
    knowledge_raw = _object(raw, "knowledge")
    thinking = model_raw.get("thinking", False)
    if not isinstance(thinking, bool):
        raise ValueError("Role Package model.thinking must be a boolean")
    model = RoleModelDefaults(
        max_tokens=int(model_raw.get("max_tokens", 64)),
        temperature=float(model_raw.get("temperature", 0.2)),
        thinking=thinking,
        thinking_budget_tokens=int(model_raw.get("thinking_budget_tokens", -1)),
    )
    memory = RoleMemoryConfig(
        provider=str(memory_raw.get("provider", "window")),
        max_turns=int(memory_raw.get("max_turns", 4)),
    )
    if model.max_tokens < 1:
        raise ValueError("Role Package model.max_tokens must be greater than zero")
    if model.thinking_budget_tokens < -1:
        raise ValueError("Role Package thinking budget must be -1 or greater")
    if memory.provider not in {"window", "none"}:
        raise ValueError(f"unsupported memory provider: {memory.provider}")
    if memory.provider == "window" and memory.max_turns < 1:
        raise ValueError("window memory max_turns must be greater than zero")

    knowledge_provider = str(knowledge_raw.get("provider", "none"))
    if knowledge_provider not in {"none", "keyword"}:
        raise ValueError(f"unsupported knowledge provider: {knowledge_provider}")
    knowledge_max_cards = int(knowledge_raw.get("max_cards", 2))
    if knowledge_max_cards < 1:
        raise ValueError("knowledge max_cards must be greater than zero")
    core_path = _optional_package_file(root, knowledge_raw.get("core"), "knowledge.core")
    cards_path = _optional_package_file(
        root,
        knowledge_raw.get("cards"),
        "knowledge.cards",
    )
    if knowledge_provider == "keyword" and cards_path is None:
        raise ValueError("keyword knowledge requires knowledge.cards")

    loop = str(raw.get("loop", "conversational"))
    if loop != "conversational":
        raise ValueError(f"unsupported Agent loop: {loop}")

    return RolePackage(
        schema_version=schema_version,
        role_id=role_id,
        version=_required_string(raw, "version"),
        display_name=_required_string(raw, "display_name"),
        prompt=prompt,
        root=root,
        loop=loop,
        model=model,
        voice=RoleVoiceDefaults(
            voice=str(voice_raw.get("voice", "")),
            language=str(voice_raw.get("language", "")),
            instructions=str(voice_raw.get("instructions", "")),
        ),
        memory=memory,
        knowledge=RoleKnowledgeConfig(
            provider=knowledge_provider,
            core_path=core_path,
            cards_path=cards_path,
            max_cards=knowledge_max_cards,
        ),
        capabilities=CapabilityPermission(
            allow=frozenset(_string_list(capability_raw, "allow")),
            deny=frozenset(_string_list(capability_raw, "deny")),
        ),
    )


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Role Package {key} must be a non-empty string")
    return value.strip()


def _required_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Role Package {key} must be an integer")
    return value


def _object(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"Role Package {key} must be an object")
    return value


def _string_list(data: dict[str, Any], key: str) -> list[str]:
    value = data.get(key, [])
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"Role Package capabilities.{key} must be a string array")
    return value


def _optional_package_file(
    root: Path,
    value: Any,
    field_name: str,
) -> Path | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError(f"Role Package {field_name} must be a string")
    path = (root / value).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError(f"Role Package {field_name} must be inside the package")
    return path
