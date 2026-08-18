from __future__ import annotations

import json
from pathlib import Path

import pytest

from g1_speech.agent_runtime import load_role_package
from g1_speech import cli
from g1_speech.config import ServiceConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_bundled_olaf_role_package_loads():
    role = load_role_package(PROJECT_ROOT / "roles" / "olaf")

    assert role.role_id == "frozen.olaf"
    assert role.loop == "conversational"
    assert role.memory.provider == "window"
    assert role.memory.max_turns == 4
    assert role.model.thinking is False
    assert "Olaf" in role.prompt
    assert role.capabilities.allow == frozenset()


def test_bundled_tifa_package_owns_model_and_voice_defaults():
    role = load_role_package(PROJECT_ROOT / "roles" / "tifa-lockhart")

    assert role.role_id == "final-fantasy-vii.tifa-lockhart"
    assert role.model.thinking is False
    assert role.model.thinking_budget_tokens == -1
    assert role.memory.max_turns == 3
    assert role.voice.voice == "Ono_Anna"
    assert role.voice.language == "Japanese"
    assert role.knowledge.provider == "keyword"
    assert role.knowledge.max_cards == 2
    assert role.knowledge.core_path.name == "core.md"
    assert role.knowledge.cards_path.name == "cards.jsonl"


def test_role_package_rejects_prompt_path_escape(tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("prompt", encoding="utf-8")
    package = tmp_path / "role"
    package.mkdir()
    (package / "role.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "test.role",
                "version": "1",
                "display_name": "Test",
                "prompt": "../outside.md",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="inside the package"):
        load_role_package(package)


def test_role_package_rejects_unimplemented_graph_loop(tmp_path):
    package = tmp_path / "role"
    package.mkdir()
    (package / "prompt.md").write_text("prompt", encoding="utf-8")
    (package / "role.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "test.role",
                "version": "1",
                "display_name": "Test",
                "prompt": "prompt.md",
                "loop": "graph",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported Agent loop"):
        load_role_package(package)


def test_agent_cli_applies_package_defaults_and_explicit_overrides(monkeypatch):
    captured = {}

    monkeypatch.setattr(cli, "load_config", lambda *_args, **_kwargs: ServiceConfig())

    def run(
        _config,
        settings,
        *,
        role_package=None,
        robot_adapter_id=None,
        vision=None,
    ):
        captured["settings"] = settings
        captured["role"] = role_package
        captured["robot_adapter_id"] = robot_adapter_id
        captured["vision"] = vision
        return 0

    monkeypatch.setattr(cli, "run_local_voice_agent", run)
    result = cli.main(
        [
            "agent",
            "--config",
            "unused.json",
            "--role-package",
            str(PROJECT_ROOT / "roles" / "tifa-lockhart"),
            "--no-thinking",
            "--history-turns",
            "2",
        ]
    )

    assert result == 0
    assert captured["role"].role_id == "final-fantasy-vii.tifa-lockhart"
    settings = captured["settings"]
    assert settings.system_prompt == captured["role"].prompt
    assert settings.tts_voice == "Ono_Anna"
    assert settings.tts_language == "Japanese"
    assert settings.max_tokens == 64
    assert settings.enable_thinking is False
    assert settings.history_turns == 2
    assert captured["robot_adapter_id"] is None
    assert captured["vision"] is None


def test_runtime_cli_uses_integrated_composition(monkeypatch):
    captured = {}
    runtime = object()

    monkeypatch.setattr(cli, "load_config", lambda *_args, **_kwargs: ServiceConfig())

    def build(config, settings, **kwargs):
        captured["config"] = config
        captured["settings"] = settings
        captured.update(kwargs)
        return runtime

    monkeypatch.setattr(cli, "build_integrated_runtime", build)
    monkeypatch.setattr(
        cli,
        "run_integrated_runtime",
        lambda selected: 0 if selected is runtime else 1,
    )

    result = cli.main(
        [
            "runtime",
            "--config",
            "unused.json",
            "--role-package",
            str(PROJECT_ROOT / "roles" / "olaf"),
        ]
    )

    assert result == 0
    assert captured["role_package"].role_id == "frozen.olaf"
    assert captured["settings"].system_prompt == captured["role_package"].prompt
    assert captured["robot_adapter_id"] is None
