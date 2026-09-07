from __future__ import annotations

import threading
import time

from star_runtime.agent.runtime import AgentRuntime
from star_runtime.agent.voice_bridge import VoiceBridgeAdapter, VoiceBridgeSettings
from star_runtime.core.control import RuntimeControlPlane
from star_runtime.core.events import PlaybackState, SpeechEvent, TtsTextChunk
from star_runtime.speech.config import TtsConfig
from star_runtime.speech.playback import PlaybackGate
from star_runtime.speech.tts import TtsAudioChunk, TtsController


def wait_for(predicate, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert predicate()


def speech(stamp, text: str = "question") -> SpeechEvent:
    return SpeechEvent(
        event_id=stamp.turn_id,
        session_id=stamp.session_id,
        sequence=1,
        created_unix_ns=time.time_ns(),
        source="test",
        text=text,
        language="en",
        audio_duration_ms=100,
        inference_ms=1.0,
        turn_id=stamp.turn_id,
        epoch=stamp.epoch,
    )


def text_chunk(stamp, request_id: str, text: str, *, final: bool = True):
    return TtsTextChunk(
        request_id=request_id,
        sequence=0,
        text=text,
        is_final=final,
        session_id=stamp.session_id,
        turn_id=stamp.turn_id,
        epoch=stamp.epoch,
    )


class StreamingLoop:
    name = "test"

    def __init__(self):
        self.blocked = threading.Event()
        self.release = threading.Event()
        self.cancelled = threading.Event()
        self.commits = []

    def stream_response(self, _text):
        yield "old"
        yield " visible"
        self.blocked.set()
        self.release.wait(2.0)
        yield " stale"

    def cancel_active(self):
        self.cancelled.set()
        return True

    def commit_turn(self, user, answer):
        self.commits.append((user, answer))

    def context_messages(self):
        return ()


class VoicePublisher:
    endpoint = "test://mouth"

    def __init__(self):
        self.calls = []

    def start(self):
        pass

    def publish(self, **fields):
        self.calls.append(fields)
        return True

    def close(self):
        pass


class VoiceSubscriber:
    endpoint = "test://ears"

    def set_handler(self, handler):
        self.speech_handler = handler

    def set_control_handler(self, handler):
        self.control_handler = handler

    def start(self):
        pass

    def close(self):
        pass


def test_barge_in_while_agent_streams_cancels_provider_and_drops_late_delta():
    owner = RuntimeControlPlane("session")
    stamp = owner.begin_turn("turn-1")
    loop = StreamingLoop()
    publisher = VoicePublisher()
    subscriber = VoiceSubscriber()
    bridge = VoiceBridgeAdapter(
        AgentRuntime(agent_id="a", role_id="r", loop=loop),
        VoiceBridgeSettings(),
        publisher=publisher,
        subscriber=subscriber,
    )
    bridge.start()
    subscriber.speech_handler(speech(stamp))
    assert loop.blocked.wait(1.0)
    wait_for(lambda: len(publisher.calls) == 1)

    invalidated = owner.invalidate("barge-in")
    assert invalidated is not None
    subscriber.control_handler(invalidated)
    assert loop.cancelled.wait(1.0)
    loop.release.set()
    wait_for(lambda: bridge._queue.empty())
    time.sleep(0.02)
    bridge.close()

    assert [call["text"] for call in publisher.calls if call["text"]] == ["old"]
    assert publisher.calls[-1]["interrupt"] is True
    assert loop.commits == []


class DelayedEngine:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.requests = []
        self.cancelled = threading.Event()

    def stream(self, request, _cancel):
        self.requests.append(request.request_id)
        self.started.set()
        self.release.wait(2.0)
        yield TtsAudioChunk(b"\0\0" * 240, 24000)

    def cancel(self):
        self.cancelled.set()

    def close(self):
        self.release.set()


class RecordingPlayer:
    def __init__(self):
        self.enqueued = []
        self.aborts = 0

    def start(self):
        pass

    def enqueue(self, pcm, rate, *, cancel=None):
        self.enqueued.append((pcm, rate))
        return not (cancel and cancel.is_set())

    def wait_until_idle(self, **_kwargs):
        return True

    def abort(self):
        self.aborts += 1

    def close(self):
        pass

    def metrics(self):
        return {}


def test_barge_in_while_tts_generates_discards_delayed_audio():
    owner = RuntimeControlPlane("session")
    stamp = owner.begin_turn("turn-1")
    engine = DelayedEngine()
    player = RecordingPlayer()
    states = []
    controller = TtsController(
        TtsConfig(),
        engine=engine,
        player=player,
        playback_state=states.append,
        control_plane=owner,
    )
    controller.start()
    controller.accept(text_chunk(stamp, "answer-1", "Hello!"))
    assert engine.started.wait(1.0)

    owner.invalidate("barge-in")
    controller.interrupt(reason="vad")
    engine.release.set()
    time.sleep(0.03)
    controller.close()

    assert engine.cancelled.is_set()
    assert player.enqueued == []
    assert [(state.active, state.turn_id) for state in states] == [
        (True, "turn-1"),
        (False, "turn-1"),
    ]


class PlayingPlayer(RecordingPlayer):
    def __init__(self):
        super().__init__()
        self.playing = threading.Event()
        self.release = threading.Event()

    def enqueue(self, pcm, rate, *, cancel=None):
        self.enqueued.append((pcm, rate))
        self.playing.set()
        return True

    def wait_until_idle(self, *, cancel=None, timeout=None):
        del timeout
        while not self.release.wait(0.005):
            if cancel is not None and cancel.is_set():
                return False
        return True

    def abort(self):
        super().abort()
        self.release.set()


class ImmediateEngine(DelayedEngine):
    def stream(self, request, _cancel):
        self.requests.append(request.request_id)
        yield TtsAudioChunk(b"\0\0" * 240, 24000)


def test_barge_in_while_pcm_plays_aborts_once_and_no_stale_completion_edge():
    owner = RuntimeControlPlane("session")
    stamp = owner.begin_turn("turn-1")
    player = PlayingPlayer()
    states = []
    controller = TtsController(
        TtsConfig(),
        engine=ImmediateEngine(),
        player=player,
        playback_state=states.append,
        control_plane=owner,
    )
    controller.start()
    controller.accept(text_chunk(stamp, "answer-1", "Hello!"))
    assert player.playing.wait(1.0)

    owner.invalidate("barge-in")
    controller.interrupt(reason="vad")
    time.sleep(0.03)
    controller.close()

    assert player.aborts >= 1
    assert [state.active for state in states] == [True, False]


def test_old_agent_delta_after_new_turn_never_reaches_tts_engine():
    owner = RuntimeControlPlane("session")
    old = owner.begin_turn("turn-1")
    owner.begin_turn("turn-2")
    engine = ImmediateEngine()
    controller = TtsController(
        TtsConfig(),
        engine=engine,
        player=RecordingPlayer(),
        playback_state=lambda _state: None,
        control_plane=owner,
    )
    controller.start()
    controller.accept(text_chunk(old, "old-answer", "stale!"))
    time.sleep(0.03)
    controller.close()

    assert engine.requests == []
    assert owner.stale_drops >= 1


def test_old_queued_tts_work_and_completion_are_stale_after_epoch_advance():
    owner = RuntimeControlPlane("session")
    stamp = owner.begin_turn("turn-1")
    engine = DelayedEngine()
    states = []
    controller = TtsController(
        TtsConfig(),
        engine=engine,
        player=RecordingPlayer(),
        playback_state=states.append,
        control_plane=owner,
    )
    controller.start()
    controller.accept(text_chunk(stamp, "first", "One!"))
    assert engine.started.wait(1.0)
    controller.accept(text_chunk(stamp, "second", "Two!"))

    owner.invalidate("explicit-cancel")
    controller.interrupt(reason="explicit")
    engine.release.set()
    time.sleep(0.03)
    controller.close()

    assert engine.requests == ["first"]
    assert [state.active for state in states] == [True, False]


def test_normal_consecutive_turns_each_complete_on_their_own_epoch():
    owner = RuntimeControlPlane("session")
    engine = ImmediateEngine()
    states = []
    controller = TtsController(
        TtsConfig(),
        engine=engine,
        player=RecordingPlayer(),
        playback_state=states.append,
        control_plane=owner,
    )
    controller.start()
    first = owner.begin_turn("turn-1")
    controller.accept(text_chunk(first, "answer-1", "One!"))
    wait_for(lambda: len(states) == 2)
    assert owner.active is False
    assert owner.invalidate("ordinary-next-speech") is None
    second = owner.begin_turn("turn-2")
    controller.accept(text_chunk(second, "answer-2", "Two!"))
    wait_for(lambda: len(states) == 4)
    controller.close()

    assert engine.requests == ["answer-1", "answer-2"]
    assert [(state.turn_id, state.epoch, state.active) for state in states] == [
        ("turn-1", 1, True),
        ("turn-1", 1, False),
        ("turn-2", 1, True),
        ("turn-2", 1, False),
    ]


def test_late_playback_edges_cannot_reactivate_or_clear_another_turn():
    owner = RuntimeControlPlane("session")
    gate = PlaybackGate(resume_delay_ms=0)
    gate.set_validator(owner.is_active)
    first = owner.begin_turn("turn-1")

    def state(stamp, active):
        return PlaybackState(
            request_id=stamp.turn_id,
            active=active,
            created_unix_ns=time.time_ns(),
            source="test",
            session_id=stamp.session_id,
            turn_id=stamp.turn_id,
            epoch=stamp.epoch,
        )

    gate.handle_state(state(first, True))
    assert gate.active
    gate.handle_state(state(first, False))
    owner.complete(first)
    gate.handle_state(state(first, True))
    assert not gate.active

    second = owner.begin_turn("turn-2")
    gate.handle_state(state(second, True))
    gate.handle_state(state(first, False))
    assert gate.active
