from __future__ import annotations

import ast
import json
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from star_runtime.agent import AgentRuntime, ConversationalLoop, NullMemory
from star_runtime.agent.role import load_role_package
from star_runtime.agent.voice_bridge import VoiceBridgeAdapter, VoiceBridgeSettings
from star_runtime.capabilities import (
    CapabilityContext,
    CapabilityPermission,
    CapabilityRegistry,
    CapabilitySpec,
)
from star_runtime.robots import RobotAdapterCatalog, RobotAdapterSpec
from star_runtime.apps.local_voice_agent import _activate_robot
from star_runtime.core.control import ControlStamp
from star_runtime.core.timing import RuntimeTimingAudit


ROOT = Path(__file__).resolve().parents[1]


def test_star_runtime_core_does_not_import_speech_or_vendor_transports():
    runtime_root = ROOT / "src" / "star_runtime"
    core_paths = [
        runtime_root / "agent",
        runtime_root / "capabilities",
        runtime_root / "robots",
        runtime_root / "transports" / "contracts.py",
    ]
    forbidden = (
        "g1_speech",
        "star_runtime.speech",
        "cyclonedds",
        "rclpy",
        "unitree",
        "galbot",
    )

    paths = []
    for root in core_paths:
        paths.extend(root.rglob("*.py") if root.is_dir() else [root])
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not any(
            name.startswith(forbidden) for name in imports
        ), f"runtime boundary leak in {path}: {imports}"


def test_transport_layout_keeps_contracts_above_concrete_backends():
    transport_root = ROOT / "src" / "star_runtime" / "transports"

    assert (transport_root / "contracts.py").is_file()
    assert (transport_root / "config.py").is_file()
    assert (transport_root / "inprocess.py").is_file()
    assert not (transport_root / "voice.py").exists()
    assert not (transport_root / "factory.py").exists()
    for backend in ("dds", "ros2"):
        assert (transport_root / backend / "speech.py").is_file()
        assert (transport_root / backend / "voice.py").is_file()
    assert not (transport_root / "dds" / "_runtime.py").exists()
    for module in ("codec.py", "reliability.py", "runtime.py", "types.py"):
        assert (transport_root / "dds" / module).is_file()


def test_speech_configuration_separates_models_from_loading_policy():
    speech_root = ROOT / "src" / "star_runtime" / "speech"

    assert (speech_root / "config.py").is_file()
    assert (speech_root / "settings" / "models.py").is_file()
    assert (speech_root / "settings" / "loader.py").is_file()
    assert len((speech_root / "config.py").read_text(encoding="utf-8").splitlines()) < 80


def test_tts_provider_scheduler_and_controller_are_separate():
    tts_root = ROOT / "src" / "star_runtime" / "speech" / "tts"

    for module in ("contracts.py", "scheduler.py", "vllm_omni.py", "controller.py"):
        assert (tts_root / module).is_file()
    assert len((tts_root / "streaming.py").read_text(encoding="utf-8").splitlines()) < 40


class _Model:
    def stream(self, _messages):
        return iter(("你好",))


class _Input:
    def __init__(self):
        self.handler = None

    def set_handler(self, handler):
        self.handler = handler

    def start(self):
        pass

    def close(self):
        pass


class _Output:
    def __init__(self):
        self.calls = []

    def start(self):
        pass

    def publish(self, **kwargs):
        self.calls.append(kwargs)
        return True

    def close(self):
        pass


class _DuplexPort(_Input, _Output):
    def __init__(self):
        _Input.__init__(self)
        _Output.__init__(self)
        self.starts = 0
        self.closes = 0

    def start(self):
        self.starts += 1

    def close(self):
        self.closes += 1


@dataclass(frozen=True)
class _Event:
    event_id: str = "event-1"
    session_id: str = "session-1"
    created_unix_ns: int = 0
    text: str = "你好"
    is_final: bool = True


def test_voice_bridge_binds_generic_ports_without_dds_configuration():
    runtime = AgentRuntime(
        agent_id="robot:test",
        role_id="role.test",
        loop=ConversationalLoop(
            system_prompt="角色",
            model=_Model(),
            memory=NullMemory(),
        ),
    )
    input_port = _Input()
    output_port = _Output()
    bridge = VoiceBridgeAdapter(
        runtime,
        VoiceBridgeSettings(),
        publisher=output_port,
        subscriber=input_port,
    )

    assert input_port.handler is not None
    input_port.handler(_Event(created_unix_ns=time.time_ns()))
    event = bridge._queue.get_nowait()
    bridge._answer(event)

    assert output_port.calls[-1]["text"] == "你好"
    assert output_port.calls[-1]["is_final"] is True


def test_voice_bridge_audits_hidden_first_delta_publish_and_completion():
    audit = RuntimeTimingAudit()
    runtime = AgentRuntime(
        agent_id="robot:test",
        role_id="role.test",
        loop=ConversationalLoop(
            system_prompt="角色",
            model=_Model(),
            memory=NullMemory(),
        ),
    )
    input_port = _Input()
    output_port = _Output()
    bridge = VoiceBridgeAdapter(
        runtime,
        VoiceBridgeSettings(),
        publisher=output_port,
        subscriber=input_port,
        timing_audit=audit,
    )
    event = _Event(created_unix_ns=time.time_ns())
    input_port.handler(event)
    bridge._answer(bridge._queue.get_nowait())

    events = audit.snapshot(ControlStamp("session-1", "event-1", 1))
    assert events["agent_start"].monotonic_ns <= events[
        "agent_first_delta"
    ].monotonic_ns
    assert events["agent_first_delta"].monotonic_ns <= events[
        "agent_first_delta_published"
    ].monotonic_ns
    assert events["agent_first_delta_published"].monotonic_ns <= events[
        "agent_complete"
    ].monotonic_ns


def test_voice_bridge_starts_a_shared_duplex_port_only_once():
    runtime = AgentRuntime(
        agent_id="robot:test",
        role_id="role.test",
        loop=ConversationalLoop(
            system_prompt="角色",
            model=_Model(),
            memory=NullMemory(),
        ),
    )
    port = _DuplexPort()
    bridge = VoiceBridgeAdapter(
        runtime,
        VoiceBridgeSettings(),
        publisher=port,
        subscriber=port,
    )

    bridge.start()
    bridge.close()

    assert port.starts == 1
    assert port.closes == 1


class _Capability:
    spec = CapabilitySpec(
        name="robot.navigation.goto",
        description="Move the active embodiment to a named waypoint.",
    )

    def invoke(self, arguments, context: CapabilityContext):
        return context.role_id, arguments["waypoint"]


class _Adapter:
    def __init__(self):
        self.started = False
        self.closed = False

    def capability_providers(self):
        return (_Capability(),)

    def start(self):
        self.started = True

    def close(self):
        self.closed = True


class _Factory:
    spec = RobotAdapterSpec(
        adapter_id="test.mobile-base",
        vendor="test",
        model="mobile-base",
        version="1.0.0",
        morphology="wheeled",
        capability_names=frozenset({"robot.navigation.goto"}),
    )

    def __init__(self):
        self.created = 0
        self.instance = None

    def create(self):
        self.created += 1
        self.instance = _Adapter()
        return self.instance


def test_robot_adapter_is_loaded_only_after_capability_preflight():
    catalog = RobotAdapterCatalog()
    factory = _Factory()
    catalog.register(factory)
    incompatible = CapabilityRegistry(
        CapabilityPermission(
            allow=frozenset({"robot.arm.grasp"}),
            required=frozenset({"robot.arm.grasp"}),
        )
    )

    with pytest.raises(RuntimeError, match="missing"):
        catalog.activate(factory.spec.adapter_id, incompatible)
    assert factory.created == 0

    compatible = CapabilityRegistry(
        CapabilityPermission(
            allow=frozenset({"robot.navigation.goto"}),
            required=frozenset({"robot.navigation.goto"}),
        )
    )
    active = catalog.activate(factory.spec.adapter_id, compatible)

    assert factory.created == 1
    assert factory.instance.started is True
    assert compatible.invoke(
        "robot.navigation.goto",
        {"waypoint": "shop"},
        CapabilityContext(agent_id="robot:one", role_id="role.tifa"),
    ) == ("role.tifa", "shop")
    active.close()
    assert factory.instance.closed is True
    with pytest.raises(LookupError, match="not registered"):
        compatible.invoke(
            "robot.navigation.goto",
            {"waypoint": "shop"},
            CapabilityContext(agent_id="robot:one", role_id="role.tifa"),
        )


def test_empty_role_does_not_scan_robot_plugins(monkeypatch):
    def unexpected_discovery(cls, **_kwargs):
        raise AssertionError("adapter discovery should remain lazy")

    monkeypatch.setattr(
        RobotAdapterCatalog,
        "discover",
        classmethod(unexpected_discovery),
    )

    assert _activate_robot(
        CapabilityRegistry(),
        adapter_id=None,
        catalog=None,
    ) is None


def test_role_required_capability_must_also_be_allowed(tmp_path):
    (tmp_path / "prompt.md").write_text("角色", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "id": "test.role",
        "version": "1.0.0",
        "display_name": "Test",
        "prompt": "prompt.md",
        "capabilities": {
            "allow": [],
            "required": ["robot.navigation.goto"],
        },
    }
    (tmp_path / "role.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must also be allowed"):
        load_role_package(tmp_path)


def test_boolean_is_not_accepted_as_a_numeric_capability_argument():
    class NumericCapability:
        spec = CapabilitySpec(
            name="robot.motion.set_speed",
            description="Set motion speed.",
            input_schema={
                "type": "object",
                "properties": {"speed": {"type": "number"}},
                "required": ["speed"],
            },
        )

        def invoke(self, arguments, context):
            return arguments, context

    registry = CapabilityRegistry(
        CapabilityPermission(allow=frozenset({"robot.motion.set_speed"}))
    )
    registry.register(NumericCapability())

    with pytest.raises(TypeError, match="invalid type"):
        registry.invoke(
            "robot.motion.set_speed",
            {"speed": True},
            CapabilityContext(agent_id="robot:one", role_id="role.tifa"),
        )
