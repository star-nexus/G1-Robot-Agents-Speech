"""ROS-independent lifecycle controller used by the ROS 2 wrapper."""

from __future__ import annotations

from typing import Callable, Protocol

from .app import SpeechService
from .config import ServiceConfig


class LifecycleService(Protocol):
    def prepare(self) -> None: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


class SpeechLifecycleController:
    """Own a speech service across configure/activate/cleanup transitions."""

    def __init__(
        self,
        service_factory: Callable[[ServiceConfig], LifecycleService] = SpeechService,
    ) -> None:
        self._service_factory = service_factory
        self._service: LifecycleService | None = None
        self._active = False

    @property
    def configured(self) -> bool:
        return self._service is not None

    @property
    def active(self) -> bool:
        return self._active

    def configure(self, config: ServiceConfig) -> None:
        if self._service is not None:
            raise RuntimeError("speech service is already configured")
        service = self._service_factory(config)
        try:
            service.prepare()
        except Exception:
            service.close()
            raise
        self._service = service

    def activate(self) -> None:
        if self._service is None:
            raise RuntimeError("speech service is not configured")
        if self._active:
            return
        self._service.start()
        self._active = True

    def deactivate(self) -> None:
        if self._service is None or not self._active:
            return
        self._service.stop()
        self._active = False

    def cleanup(self) -> None:
        if self._service is None:
            return
        self.deactivate()
        self._service.close()
        self._service = None
