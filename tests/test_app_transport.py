from __future__ import annotations

from g1_speech import app
from g1_speech.config import ServiceConfig
from g1_speech.contracts import PipelineMetricsSnapshot


class Transport:
    def __init__(self):
        self.sink = object()
        self.calls = []

    def start(self):
        self.calls.append("transport.start")

    def stop(self):
        self.calls.append("transport.stop")

    def close(self):
        self.calls.append("transport.close")

    def metrics(self):
        return {"transport_backend": "test", "transport_published": 2}


class Pipeline:
    def __init__(self, **kwargs):
        self.sink = kwargs["sink"]
        self.calls = []

    def prepare(self):
        self.calls.append("pipeline.prepare")

    def start(self):
        self.calls.append("pipeline.start")

    def close(self):
        self.calls.append("pipeline.close")

    def metrics(self):
        return PipelineMetricsSnapshot(*([0] * 13))


def test_service_owns_transport_lifecycle_and_generic_metrics(monkeypatch):
    monkeypatch.setattr(app, "SoundDeviceSource", lambda **kwargs: object())
    monkeypatch.setattr(app, "SileroVadSegmenter", lambda **kwargs: object())
    monkeypatch.setattr(app, "create_asr_engine", lambda config: object())
    monkeypatch.setattr(app, "SpeechPipeline", Pipeline)
    transport = Transport()

    service = app.SpeechService(ServiceConfig(), transport=transport)
    service.prepare()
    service.start()
    assert service.metrics()["transport_backend"] == "test"
    service.stop()
    service.close()

    assert service.pipeline.sink is transport.sink
    assert service.pipeline.calls == [
        "pipeline.prepare",
        "pipeline.start",
        "pipeline.close",
    ]
    assert transport.calls == [
        "transport.start",
        "transport.stop",
        "transport.close",
    ]


def test_close_releases_engine_after_prepare_without_start(monkeypatch):
    monkeypatch.setattr(app, "SoundDeviceSource", lambda **kwargs: object())
    monkeypatch.setattr(app, "SileroVadSegmenter", lambda **kwargs: object())
    monkeypatch.setattr(app, "create_asr_engine", lambda config: object())
    monkeypatch.setattr(app, "SpeechPipeline", Pipeline)
    transport = Transport()
    service = app.SpeechService(ServiceConfig(), transport=transport)

    service.prepare()
    service.close()

    assert service.pipeline.calls == ["pipeline.prepare", "pipeline.close"]
