from __future__ import annotations

from types import SimpleNamespace

import pytest

from star_runtime.apps.integrated_runtime import IntegratedRuntime


class _Speech:
    def __init__(self, calls):
        self.calls = calls

    def prepare(self):
        self.calls.append("speech.prepare")

    def start(self):
        self.calls.append("speech.start")

    def stop(self):
        self.calls.append("speech.stop")

    def close(self):
        self.calls.append("speech.close")

    def metrics(self):
        return {"transport_backend": "inprocess"}


class _Agent:
    def __init__(self, calls):
        self.calls = calls
        self.runtime = SimpleNamespace(role_id="role.test")

    def start(self):
        self.calls.append("agent.start")

    def close(self):
        self.calls.append("agent.close")


class _Robot:
    def __init__(self, calls):
        self.calls = calls
        self.spec = SimpleNamespace(adapter_id="test.robot")

    def close(self):
        self.calls.append("robot.close")


def test_integrated_runtime_orders_consumers_capture_and_shutdown():
    calls = []
    runtime = IntegratedRuntime(
        speech=_Speech(calls),
        agent=_Agent(calls),
        robot=_Robot(calls),
    )

    runtime.prepare()
    runtime.start()
    assert runtime.metrics()["role_id"] == "role.test"
    runtime.close()
    runtime.close()

    assert calls == [
        "speech.prepare",
        "agent.start",
        "speech.start",
        "speech.stop",
        "agent.close",
        "speech.close",
        "robot.close",
    ]


def test_integrated_runtime_closes_agent_when_speech_start_fails():
    calls = []

    class FailingSpeech(_Speech):
        def start(self):
            super().start()
            raise RuntimeError("capture failed")

    runtime = IntegratedRuntime(speech=FailingSpeech(calls), agent=_Agent(calls))

    with pytest.raises(RuntimeError, match="capture failed"):
        runtime.start()

    assert calls == ["agent.start", "speech.start", "agent.close"]
