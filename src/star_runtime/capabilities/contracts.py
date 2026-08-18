"""Stable capability contracts shared by Agents and robot adapters."""

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
    contract_version: str = "1.0"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("capability name must not be empty")
        if not self.description.strip():
            raise ValueError("capability description must not be empty")
        if not self.contract_version.strip():
            raise ValueError("capability contract_version must not be empty")


class CapabilityProvider(Protocol):
    """One callable robot or software capability."""

    @property
    def spec(self) -> CapabilitySpec: ...

    def invoke(
        self,
        arguments: Mapping[str, Any],
        context: CapabilityContext,
    ) -> Any: ...
