"""Embodiment adapter contracts and lazy adapter catalog."""

from .catalog import ActiveRobot, RobotAdapterCatalog
from .contracts import (
    RobotAdapter,
    RobotAdapterFactory,
    RobotAdapterSpec,
)

__all__ = [
    "ActiveRobot",
    "RobotAdapter",
    "RobotAdapterCatalog",
    "RobotAdapterFactory",
    "RobotAdapterSpec",
]
