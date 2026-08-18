"""Single-process STAR Runtime composition for constrained edge devices."""

from __future__ import annotations

import signal
import threading
from dataclasses import dataclass, field

from ..agent.role import RolePackage
from ..capabilities import CapabilityRegistry
from ..robots import ActiveRobot, RobotAdapterCatalog
from ..speech.config import ServiceConfig
from ..speech.playback import PlaybackGate
from ..speech.service import SpeechService
from ..transports.inprocess import InProcessTransport
from .local_voice_agent import (
    LocalAgentSettings,
    LocalVoiceAgent,
    _activate_robot,
)


@dataclass
class IntegratedRuntime:
    """Own one ASR → Agent → TTS process and its optional robot body."""

    speech: SpeechService
    agent: LocalVoiceAgent
    robot: ActiveRobot | None = None
    _started: bool = field(default=False, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def prepare(self) -> None:
        self.speech.prepare()

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("integrated runtime is closed")
        if self._started:
            return
        # Register and start the Agent consumer before microphone capture.
        self.agent.start()
        try:
            self.speech.start()
        except Exception:
            self.agent.close()
            raise
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        # Stop new observations before draining/closing the reasoning bridge.
        try:
            self.speech.stop()
        finally:
            self.agent.close()
            self._started = False

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.stop()
        finally:
            try:
                self.speech.close()
            finally:
                if self.robot is not None:
                    self.robot.close()
                    self.robot = None
                self._closed = True

    def metrics(self) -> dict[str, object]:
        payload = self.speech.metrics()
        payload.update(
            {
                "runtime_mode": "integrated",
                "role_id": self.agent.runtime.role_id,
                "robot_adapter": (
                    None if self.robot is None else self.robot.spec.adapter_id
                ),
            }
        )
        return payload


def build_integrated_runtime(
    config: ServiceConfig,
    settings: LocalAgentSettings,
    *,
    role_package: RolePackage | None = None,
    robot_adapter_id: str | None = None,
    robot_catalog: RobotAdapterCatalog | None = None,
    client: object | None = None,
) -> IntegratedRuntime:
    """Compose the default zero-middleware runtime without loading vendor SDKs early."""

    if not config.tts.enabled:
        raise ValueError("integrated runtime requires tts.enabled=true")
    capabilities = CapabilityRegistry(
        role_package.capabilities if role_package is not None else None
    )
    robot = _activate_robot(
        capabilities,
        adapter_id=robot_adapter_id,
        catalog=robot_catalog,
    )
    try:
        gate = PlaybackGate(
            resume_delay_ms=config.playback.resume_delay_ms,
            max_active_seconds=config.playback.max_active_seconds,
        )
        transport = InProcessTransport(gate.set_active)
        speech = SpeechService(config, transport=transport, playback_gate=gate)
        agent = LocalVoiceAgent(
            config,
            settings,
            client=client,
            publisher=transport.voice,
            subscriber=transport.voice,
            role_package=role_package,
            capabilities=capabilities,
        )
    except Exception:
        if robot is not None:
            robot.close()
        raise
    return IntegratedRuntime(speech=speech, agent=agent, robot=robot)


def run_integrated_runtime(runtime: IntegratedRuntime) -> int:
    stopped = threading.Event()

    def stop(*_args: object) -> None:
        stopped.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        runtime.start()
        while not stopped.wait(0.2):
            pass
    finally:
        runtime.close()
    return 0


__all__ = [
    "IntegratedRuntime",
    "build_integrated_runtime",
    "run_integrated_runtime",
]
