"""Lazy robot-adapter catalog for resource-constrained edge deployments."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass

from ..capabilities import (
    CapabilityCompatibility,
    CapabilityPermission,
    CapabilityProvider,
    CapabilityRegistry,
)
from .contracts import RobotAdapter, RobotAdapterFactory, RobotAdapterSpec


@dataclass
class ActiveRobot:
    """Lifecycle handle returned after a compatible embodiment is activated."""

    spec: RobotAdapterSpec
    adapter: RobotAdapter
    registry: CapabilityRegistry
    providers: tuple[CapabilityProvider, ...]
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.registry.unregister_many(self.providers)
        finally:
            self.adapter.close()
            self._closed = True


class RobotAdapterCatalog:
    """Discovers embodiment metadata without loading unused robot SDKs."""

    def __init__(self) -> None:
        self._factories: dict[str, RobotAdapterFactory] = {}

    @classmethod
    def discover(
        cls,
        *,
        group: str = "star_runtime.robot_adapters",
    ) -> "RobotAdapterCatalog":
        """Load lightweight adapter factories registered by ``star-robot-*`` packages."""

        catalog = cls()
        entries = importlib.metadata.entry_points()
        selected = entries.select(group=group)
        for entry in selected:
            factory = entry.load()
            if isinstance(factory, type):
                factory = factory()
            elif callable(factory) and not hasattr(factory, "spec"):
                factory = factory()
            if not hasattr(factory, "spec") or not hasattr(factory, "create"):
                raise TypeError(f"invalid robot adapter entry point: {entry.name}")
            catalog.register(factory)
        return catalog

    def compatible_adapter_ids(
        self,
        permission: CapabilityPermission,
    ) -> tuple[str, ...]:
        return tuple(
            spec.adapter_id
            for spec in self.available_specs()
            if self.compatibility(spec.adapter_id, permission).compatible
        )

    def register(self, factory: RobotAdapterFactory) -> None:
        adapter_id = factory.spec.adapter_id
        if adapter_id in self._factories:
            raise ValueError(f"robot adapter is already registered: {adapter_id}")
        self._factories[adapter_id] = factory

    def available_specs(self) -> tuple[RobotAdapterSpec, ...]:
        return tuple(factory.spec for factory in self._factories.values())

    def compatibility(
        self,
        adapter_id: str,
        permission: CapabilityPermission,
    ) -> CapabilityCompatibility:
        spec = self._factory(adapter_id).spec
        required = permission.required
        denied = frozenset(name for name in required if not permission.allows(name))
        available = frozenset(
            name
            for name in required
            if name in spec.capability_names and name not in denied
        )
        missing = required - available - denied
        return CapabilityCompatibility(
            required=required,
            available=available,
            missing=missing,
            denied=denied,
        )

    def activate(
        self,
        adapter_id: str,
        registry: CapabilityRegistry,
    ) -> ActiveRobot:
        """Preflight first, then load exactly one compatible vendor adapter."""

        factory = self._factory(adapter_id)
        report = self.compatibility(adapter_id, registry.permission)
        if not report.compatible:
            raise RuntimeError(f"robot capability mismatch: {report.explain()}")

        adapter = factory.create()
        providers: tuple[CapabilityProvider, ...] = ()
        registered = False
        try:
            providers = tuple(adapter.capability_providers())
            actual_names = frozenset(provider.spec.name for provider in providers)
            if actual_names != factory.spec.capability_names:
                raise RuntimeError(
                    "robot adapter capability manifest differs from its providers: "
                    f"declared={sorted(factory.spec.capability_names)!r}, "
                    f"actual={sorted(actual_names)!r}"
                )
            adapter.start()
            registry.register_many(providers)
            registered = True
            registry.ensure_compatible()
        except Exception:
            if registered:
                registry.unregister_many(providers)
            adapter.close()
            raise
        return ActiveRobot(
            spec=factory.spec,
            adapter=adapter,
            registry=registry,
            providers=providers,
        )

    def _factory(self, adapter_id: str) -> RobotAdapterFactory:
        try:
            return self._factories[adapter_id]
        except KeyError as exc:
            raise LookupError(f"robot adapter is not registered: {adapter_id}") from exc
