"""Lightweight robot embodiment contracts that never load a vendor SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from ..capabilities import CapabilityProvider


@dataclass(frozen=True)
class RobotAdapterSpec:
    adapter_id: str
    vendor: str
    model: str
    version: str
    capability_names: frozenset[str]
    morphology: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.adapter_id.strip():
            raise ValueError("robot adapter_id must not be empty")
        if not self.version.strip():
            raise ValueError("robot adapter version must not be empty")
        if any(not name.strip() for name in self.capability_names):
            raise ValueError("robot capability names must not be empty")


class RobotAdapter(Protocol):
    """One active robot body; heavyweight resources begin at ``start``."""

    def capability_providers(self) -> tuple[CapabilityProvider, ...]: ...

    def start(self) -> None: ...

    def close(self) -> None: ...


class RobotAdapterFactory(Protocol):
    """Cheap factory whose ``create`` method may import a vendor SDK."""

    @property
    def spec(self) -> RobotAdapterSpec: ...

    def create(self) -> RobotAdapter: ...
