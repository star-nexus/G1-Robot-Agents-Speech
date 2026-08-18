from __future__ import annotations

import pytest

from g1_speech.agent_runtime import (
    AgentRuntime,
    CapabilityContext,
    CapabilityPermission,
    CapabilityRegistry,
    CapabilitySpec,
    ConversationalLoop,
    NullMemory,
    WindowMemory,
)


class RecordingModel:
    def __init__(self) -> None:
        self.calls = []

    def stream(self, messages):
        self.calls.append(messages)
        return iter(("答", "案"))


def test_conversational_loop_makes_one_model_call_and_commits_after_success():
    model = RecordingModel()
    memory = WindowMemory(max_turns=1)
    loop = ConversationalLoop(
        system_prompt="角色",
        model=model,
        memory=memory,
    )

    assert list(loop.stream_response("第一问")) == ["答", "案"]
    assert memory.context_messages() == ()
    loop.commit_turn("第一问", "答案")
    assert list(loop.stream_response("第二问")) == ["答", "案"]

    assert len(model.calls) == 2
    assert model.calls[-1] == [
        {"role": "system", "content": "角色"},
        {"role": "user", "content": "第一问"},
        {"role": "assistant", "content": "答案"},
        {"role": "user", "content": "第二问"},
    ]


def test_window_memory_is_bounded_by_completed_turns():
    memory = WindowMemory(max_turns=2)
    memory.record_turn("一", "1")
    memory.record_turn("二", "2")
    memory.record_turn("三", "3")

    assert memory.context_messages() == (
        {"role": "user", "content": "二"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "三"},
        {"role": "assistant", "content": "3"},
    )
    memory.clear()
    assert memory.context_messages() == ()


def test_null_memory_keeps_no_context():
    memory = NullMemory()
    memory.record_turn("问题", "回答")
    assert memory.context_messages() == ()


class EchoCapability:
    spec = CapabilitySpec(
        name="robot.say_internal",
        description="test capability",
    )

    def invoke(self, arguments, context: CapabilityContext):
        return context.role_id, arguments["value"]


def test_capability_registry_is_deny_by_default_and_role_scoped():
    denied = CapabilityRegistry()
    denied.register(EchoCapability())
    context = CapabilityContext(agent_id="robot:one", role_id="role.one")
    with pytest.raises(PermissionError):
        denied.invoke("robot.say_internal", {"value": 1}, context)

    allowed = CapabilityRegistry(
        CapabilityPermission(allow=frozenset({"robot.say_internal"}))
    )
    allowed.register(EchoCapability())
    assert allowed.available_specs() == (EchoCapability.spec,)
    assert allowed.invoke("robot.say_internal", {"value": 7}, context) == (
        "role.one",
        7,
    )


def test_agent_runtime_exposes_loop_and_capability_ports():
    model = RecordingModel()
    loop = ConversationalLoop(
        system_prompt="system",
        model=model,
        memory=NullMemory(),
    )
    runtime = AgentRuntime(
        agent_id="robot:a",
        role_id="role.a",
        loop=loop,
    )

    assert runtime.loop.name == "conversational"
    assert list(runtime.stream_response("hello")) == ["答", "案"]
