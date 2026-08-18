"""Composition root for the integrated microphone-to-speaker runtime."""

from __future__ import annotations

from typing import Any

from ..speech.config import ServiceConfig
from ..speech.playback import PlaybackGate
from ..speech.service import SpeechService
from ..transports.contracts import SpeechTransport


def create_speech_transport(
    config: ServiceConfig,
    gate: PlaybackGate,
    *,
    backend: str | None = None,
    ros_node: Any | None = None,
    ros_lifecycle: bool = False,
) -> SpeechTransport:
    """Select a remote/distributed adapter at the outer composition boundary."""

    selected = backend or config.transport.backend
    if selected == "dds":
        if ros_node is not None or ros_lifecycle:
            raise ValueError("ROS node options cannot be used with the DDS transport")
        from ..transports.dds.speech import DdsTransport

        return DdsTransport(config.dds, gate.set_active)
    if selected == "ros2":
        from ..transports.ros2.speech import Ros2Transport

        return Ros2Transport(
            config.ros2,
            gate.set_active,
            node=ros_node,
            lifecycle=ros_lifecycle,
        )
    if selected == "inprocess":
        raise ValueError(
            "inprocess is a complete Runtime composition, not a standalone "
            "speech transport; use build_integrated_runtime()"
        )
    raise ValueError(f"unsupported speech transport: {selected!r}")


def build_speech_runtime(
    config: ServiceConfig,
    *,
    transport_backend: str | None = None,
    ros_node: Any | None = None,
    ros_lifecycle: bool = False,
) -> SpeechService:
    """Build one integrated runtime while keeping remote transports optional."""

    gate = PlaybackGate(
        resume_delay_ms=config.playback.resume_delay_ms,
        max_active_seconds=config.playback.max_active_seconds,
    )
    transport = create_speech_transport(
        config,
        gate,
        backend=transport_backend,
        ros_node=ros_node,
        ros_lifecycle=ros_lifecycle,
    )
    return SpeechService(config, transport=transport, playback_gate=gate)
