from __future__ import annotations

from star_runtime.cli import main as cli
from star_runtime.speech.config import ServiceConfig


class FakeCamera:
    def __init__(self) -> None:
        self.calls = []

    def start(self, timeout):
        self.calls.append(("start", timeout))

    def close(self):
        self.calls.append(("close",))


def test_agent_cli_owns_visual_camera_lifecycle(monkeypatch):
    camera = FakeCamera()
    vision = object()
    captured = {}
    monkeypatch.setattr(cli, "load_config", lambda *_args, **_kwargs: ServiceConfig())
    monkeypatch.setattr(cli, "_build_vision_input", lambda _args: (vision, camera))

    def run(_config, _settings, **kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(cli, "run_local_voice_agent", run)

    result = cli.main(
        [
            "agent",
            "--config",
            "unused.json",
            "--vision",
            "always",
            "--camera-start-timeout",
            "3",
        ]
    )

    assert result == 0
    assert captured["vision"] is vision
    assert camera.calls == [("start", 3.0), ("close",)]


def test_agent_cli_closes_camera_when_agent_startup_fails(monkeypatch):
    camera = FakeCamera()
    monkeypatch.setattr(cli, "load_config", lambda *_args, **_kwargs: ServiceConfig())
    monkeypatch.setattr(cli, "_build_vision_input", lambda _args: (object(), camera))

    def fail(*_args, **_kwargs):
        raise RuntimeError("agent failed")

    monkeypatch.setattr(cli, "run_local_voice_agent", fail)

    try:
        cli.main(["agent", "--config", "unused.json", "--vision", "always"])
    except RuntimeError as exc:
        assert str(exc) == "agent failed"
    else:
        raise AssertionError("expected Agent startup failure")

    assert camera.calls == [("start", 5.0), ("close",)]
