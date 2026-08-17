"""Capability/tool provider port with deny-by-default role permissions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class CapabilityContext:
    agent_id: str
    role_id: str
    session_id: str = ""


@dataclass(frozen=True)
class CapabilitySpec:
    name: str
    description: str
    input_schema: Mapping[str, Any] = field(default_factory=dict)


class CapabilityProvider(Protocol):
    """One callable robot or software capability."""

    @property
    def spec(self) -> CapabilitySpec: ...

    def invoke(
        self,
        arguments: Mapping[str, Any],
        context: CapabilityContext,
    ) -> Any: ...


@dataclass(frozen=True)
class CapabilityPermission:
    """Role-level allow/deny policy. Empty allow means no callable tools."""

    allow: frozenset[str] = frozenset()
    deny: frozenset[str] = frozenset()

    def allows(self, name: str) -> bool:
        if name in self.deny or "*" in self.deny:
            return False
        return "*" in self.allow or name in self.allow


class CapabilityRegistry:
    """Registers providers once; the conversation hot path does no tool work."""

    def __init__(self, permission: CapabilityPermission | None = None) -> None:
        self.permission = permission or CapabilityPermission()
        self._providers: dict[str, CapabilityProvider] = {}

    def register(self, provider: CapabilityProvider) -> None:
        name = provider.spec.name
        if not name:
            raise ValueError("capability name must not be empty")
        if name in self._providers:
            raise ValueError(f"capability is already registered: {name}")
        self._providers[name] = provider

    def available_specs(self) -> tuple[CapabilitySpec, ...]:
        return tuple(
            provider.spec
            for name, provider in self._providers.items()
            if self.permission.allows(name)
        )

    def invoke(
        self,
        name: str,
        arguments: Mapping[str, Any],
        context: CapabilityContext,
    ) -> Any:
        if not self.permission.allows(name):
            raise PermissionError(f"role is not allowed to invoke capability: {name}")
        try:
            provider = self._providers[name]
        except KeyError as exc:
            raise LookupError(f"capability provider is not registered: {name}") from exc
        return provider.invoke(arguments, context)
