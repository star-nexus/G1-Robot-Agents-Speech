from __future__ import annotations

import pytest

from g1_speech.config import ServiceConfig
from g1_speech.lifecycle import SpeechLifecycleController


class Service:
    def __init__(self, calls):
        self.calls = calls

    def prepare(self):
        self.calls.append("prepare")

    def start(self):
        self.calls.append("start")

    def stop(self):
        self.calls.append("stop")

    def close(self):
        self.calls.append("close")


def test_lifecycle_preloads_before_capture_and_reuses_service():
    calls = []
    controller = SpeechLifecycleController(lambda config: Service(calls))

    controller.configure(ServiceConfig())
    assert calls == ["prepare"]
    assert controller.configured
    assert not controller.active

    controller.activate()
    controller.deactivate()
    controller.activate()
    controller.cleanup()

    assert calls == ["prepare", "start", "stop", "start", "stop", "close"]
    assert not controller.configured
    assert not controller.active


def test_lifecycle_rejects_activate_before_configure():
    controller = SpeechLifecycleController(lambda config: Service([]))

    with pytest.raises(RuntimeError, match="not configured"):
        controller.activate()


def test_failed_prepare_closes_partial_service():
    calls = []

    class BrokenService(Service):
        def prepare(self):
            self.calls.append("prepare")
            raise RuntimeError("model failed")

    controller = SpeechLifecycleController(lambda config: BrokenService(calls))

    with pytest.raises(RuntimeError, match="model failed"):
        controller.configure(ServiceConfig())
    assert calls == ["prepare", "close"]
    assert not controller.configured
