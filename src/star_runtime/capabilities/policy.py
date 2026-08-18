"""Role authorization and embodiment-compatibility policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CapabilityPermission:
    """Role-level allow/deny policy. Empty allow means no callable tools."""

    allow: frozenset[str] = frozenset()
    deny: frozenset[str] = frozenset()
    required: frozenset[str] = frozenset()

    def allows(self, name: str) -> bool:
        if name in self.deny or "*" in self.deny:
            return False
        return "*" in self.allow or name in self.allow


@dataclass(frozen=True)
class CapabilityCompatibility:
    """Result of matching a role's requirements to one active embodiment."""

    required: frozenset[str]
    available: frozenset[str]
    missing: frozenset[str]
    denied: frozenset[str]

    @property
    def compatible(self) -> bool:
        return not self.missing and not self.denied

    def explain(self) -> str:
        if self.compatible:
            return "compatible"
        parts = []
        if self.missing:
            parts.append(f"missing={sorted(self.missing)!r}")
        if self.denied:
            parts.append(f"denied={sorted(self.denied)!r}")
        return ", ".join(parts)
