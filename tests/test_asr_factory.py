from types import SimpleNamespace

import pytest

from g1_speech import asr
from g1_speech.config import AsrConfig, ServiceConfig
from g1_speech.engine import SenseVoiceEngine
from g1_speech.qwen3_asr import Qwen3AsrEngine


def test_builtin_factories_keep_pipeline_composition_model_agnostic():
    sensevoice = asr.create_asr_engine(ServiceConfig())
    qwen = asr.create_asr_engine(
        ServiceConfig(asr=AsrConfig(backend="qwen3_asr"))
    )

    assert isinstance(sensevoice, SenseVoiceEngine)
    assert isinstance(qwen, Qwen3AsrEngine)


def test_module_factory_allows_external_adapter_without_core_changes(monkeypatch):
    expected = object()

    def factory(config):
        return expected

    monkeypatch.setattr(
        asr.importlib,
        "import_module",
        lambda name: SimpleNamespace(create_engine=factory),
    )

    assert asr.resolve_asr_factory("robot_asr:create_engine") is factory


def test_unknown_backend_has_actionable_error():
    with pytest.raises(ValueError, match="entry point, or module:factory"):
        asr.resolve_asr_factory("missing_backend")
