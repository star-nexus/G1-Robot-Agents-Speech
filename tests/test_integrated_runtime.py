from __future__ import annotations

from types import SimpleNamespace

import pytest

from star_runtime.apps import integrated_runtime as integrated_module
from star_runtime.apps.integrated_runtime import IntegratedRuntime
from star_runtime.apps.local_voice_agent import LocalAgentSettings
from star_runtime.speech.config import ServiceConfig, TtsConfig


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


def test_integrated_builder_shares_one_timing_audit_across_speech_and_agent(
    monkeypatch,
):
    captured = {}

    class Speech:
        def __init__(self, _config, **kwargs):
            captured["speech"] = kwargs["timing_audit"]

    class Agent:
        def __init__(self, _config, _settings, **kwargs):
            captured["agent"] = kwargs["timing_audit"]
            self.runtime = SimpleNamespace(role_id="role.test")

    monkeypatch.setattr(integrated_module, "SpeechService", Speech)
    monkeypatch.setattr(integrated_module, "LocalVoiceAgent", Agent)

    integrated_module.build_integrated_runtime(
        ServiceConfig(tts=TtsConfig(enabled=True)),
        LocalAgentSettings(),
    )

    assert captured["speech"] is captured["agent"]
