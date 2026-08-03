"""Transport factory with lazy optional-backend imports."""

from __future__ import annotations

from typing import Any

from .config import ServiceConfig
from .contracts import SpeechTransport
from .gate import PlaybackGate


def create_transport(
    config: ServiceConfig,
    gate: PlaybackGate,
    *,
    backend: str | None = None,
    ros_node: Any | None = None,
    ros_lifecycle: bool = False,
) -> SpeechTransport:
    selected = backend or config.transport.backend
    if selected == "dds":
        if ros_node is not None or ros_lifecycle:
            raise ValueError("ROS node options cannot be used with the DDS transport")
        from .dds import DdsTransport

        return DdsTransport(config.dds, gate)
    if selected == "ros2":
        from .ros2 import Ros2Transport

        return Ros2Transport(
            config.ros2,
            gate,
            node=ros_node,
            lifecycle=ros_lifecycle,
        )
    raise ValueError(f"unsupported speech transport: {selected!r}")
