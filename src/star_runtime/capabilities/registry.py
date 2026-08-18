"""Capability provider registration, compatibility checks, and invocation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .contracts import CapabilityContext, CapabilityProvider, CapabilitySpec
from .policy import CapabilityCompatibility, CapabilityPermission


class CapabilityRegistry:
    """Registers providers once; the conversation hot path does no tool work."""

    def __init__(self, permission: CapabilityPermission | None = None) -> None:
        self.permission = permission or CapabilityPermission()
        self._providers: dict[str, CapabilityProvider] = {}

    def register(self, provider: CapabilityProvider) -> None:
        self.register_many((provider,))

    def register_many(self, providers: tuple[CapabilityProvider, ...]) -> None:
        """Register a robot adapter's providers atomically."""

        pending: dict[str, CapabilityProvider] = {}
        for provider in providers:
            name = provider.spec.name
            if not name:
                raise ValueError("capability name must not be empty")
            if name in self._providers or name in pending:
                raise ValueError(f"capability is already registered: {name}")
            pending[name] = provider
        self._providers.update(pending)

    def unregister_many(self, providers: tuple[CapabilityProvider, ...]) -> None:
        """Remove only the exact provider instances owned by one adapter."""

        for provider in providers:
            name = provider.spec.name
            if self._providers.get(name) is provider:
                del self._providers[name]

    def available_specs(self) -> tuple[CapabilitySpec, ...]:
        return tuple(
            provider.spec
            for name, provider in self._providers.items()
            if self.permission.allows(name)
        )

    def check_compatibility(
        self,
        required: frozenset[str] | None = None,
    ) -> CapabilityCompatibility:
        requested = self.permission.required if required is None else required
        denied = frozenset(name for name in requested if not self.permission.allows(name))
        available = frozenset(
            name
            for name in requested
            if name in self._providers and name not in denied
        )
        missing = requested - available - denied
        return CapabilityCompatibility(
            required=requested,
            available=available,
            missing=missing,
            denied=denied,
        )

    def ensure_compatible(
        self,
        required: frozenset[str] | None = None,
    ) -> None:
        report = self.check_compatibility(required)
        if not report.compatible:
            raise RuntimeError(f"robot capability mismatch: {report.explain()}")

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
        _validate_arguments(provider.spec, arguments)
        return provider.invoke(arguments, context)


def _validate_arguments(
    spec: CapabilitySpec,
    arguments: Mapping[str, Any],
) -> None:
    """Validate the dependency-free JSON-Schema subset used by robot tools."""

    schema = spec.input_schema
    if not schema:
        return
    if schema.get("type", "object") != "object":
        raise ValueError(f"capability input schema must describe an object: {spec.name}")
    required = schema.get("required", ())
    missing = [name for name in required if name not in arguments]
    if missing:
        raise ValueError(f"capability {spec.name} missing arguments: {missing!r}")
    properties = schema.get("properties", {})
    if schema.get("additionalProperties") is False:
        unknown = set(arguments) - set(properties)
        if unknown:
            raise ValueError(
                f"capability {spec.name} received unknown arguments: {sorted(unknown)!r}"
            )
    python_types = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": (list, tuple),
        "object": Mapping,
    }
    for field_name, value in arguments.items():
        field_schema = properties.get(field_name, {})
        expected = python_types.get(field_schema.get("type"))
        invalid_boolean_number = isinstance(value, bool) and field_schema.get(
            "type"
        ) in {"integer", "number"}
        if expected is not None and (
            not isinstance(value, expected) or invalid_boolean_number
        ):
            raise TypeError(
                f"capability {spec.name} argument {field_name!r} has invalid type"
            )
        if "enum" in field_schema and value not in field_schema["enum"]:
            raise ValueError(
                f"capability {spec.name} argument {field_name!r} is not an allowed value"
            )
