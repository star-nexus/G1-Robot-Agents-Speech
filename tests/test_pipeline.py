from __future__ import annotations

import queue
import threading
import time

import numpy as np

from g1_speech.contracts import AudioChunk, RecognitionResult, Utterance
from g1_speech.gate import PlaybackGate
from g1_speech.pipeline import SpeechPipeline
from star_runtime.core.control import RuntimeControlPlane
from star_runtime.core.timing import RuntimeTimingAudit


class FakeSource:
    dropped_chunks = 0
    reconnections = 0
    reconnect_failures = 0

    def __init__(self):
        self.items = queue.Queue()
        self.started = False
        self.discontinuity_count = 0

    @property
    def queue_size(self):
        return self.items.qsize()

    def start(self):
        self.started = True

    def read(self, timeout=0.2):
        try:
            return self.items.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self):
        self.started = False


class EveryChunkIsUtterance:
    def __init__(self):
        self.reset_count = 0

    def accept(self, chunk):
        return [
            Utterance(
                chunk.samples,
                chunk.sample_rate,
                chunk.captured_monotonic_ns,
                chunk.captured_monotonic_ns,
            )
        ]

    def reset(self):
        self.reset_count += 1


class FakeEngine:
    def __init__(self, source=None):
        self.warmups = 0
        self.source = source
        self.source_started_during_warmup = None
        self.warmup_thread_name = None
        self.loads = 0
        self.closes = 0

    def load(self):
        self.loads += 1

    def close(self):
        self.closes += 1

    def warmup(self):
        self.warmups += 1
        self.warmup_thread_name = threading.current_thread().name
        if self.source is not None:
            self.source_started_during_warmup = self.source.started

    def transcribe(self, utterance):
        return RecognitionResult("向前走", "zh", 5.0)


class CollectingSink:
    def __init__(self):
        self.events = []

    def start(self):
        pass

    def publish(self, event):
        self.events.append(event)
        return True

    def close(self):
        pass


def chunk(now_ns=None):
    return AudioChunk(
        np.ones(1600, dtype=np.float32),
        16000,
        time.monotonic_ns() if now_ns is None else now_ns,
    )


def wait_for(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert predicate()


def test_pipeline_publishes_final_event(caplog):
    caplog.set_level("INFO", logger="star_runtime.speech.pipeline")
    source = FakeSource()
    sink = CollectingSink()
    pipeline = SpeechPipeline(
        source=source,
        segmenter=EveryChunkIsUtterance(),
        engine=FakeEngine(),
        sink=sink,
        playback_gate=PlaybackGate(resume_delay_ms=0),
    )
    pipeline.start()
    source.items.put(chunk())
    wait_for(lambda: len(sink.events) == 1)
    pipeline.close()

    event = sink.events[0]
    assert event.text == "向前走"
    assert event.is_final
    assert event.sequence == 1
    assert f"Speech latency event_id={event.event_id}" in caplog.text
    assert "speech_end_to_final=" in caplog.text


def test_pipeline_records_acoustic_vad_and_asr_timestamps_without_estimation():
    source = FakeSource()
    sink = CollectingSink()
    audit = RuntimeTimingAudit()
    pipeline = SpeechPipeline(
        source=source,
        segmenter=EveryChunkIsUtterance(),
        engine=FakeEngine(),
        sink=sink,
        playback_gate=PlaybackGate(resume_delay_ms=0),
        timing_audit=audit,
    )
    pipeline.start()
    captured_ns = time.monotonic_ns()
    source.items.put(chunk(captured_ns))
    wait_for(lambda: len(sink.events) == 1)
    pipeline.close()

    events = audit.snapshot(sink.events[0].control_stamp)
    assert events["speech_start"].monotonic_ns == captured_ns
    assert events["speech_end"].monotonic_ns == captured_ns
    assert events["vad_ready"].monotonic_ns == captured_ns
    assert events["asr_start"].monotonic_ns >= captured_ns
    assert events["asr_final"].monotonic_ns >= events["asr_start"].monotonic_ns
    assert events["speech_event_published"].monotonic_ns >= events[
        "turn_started"
    ].monotonic_ns


def test_asr_final_cleanup_runs_after_new_turn_is_authoritative():
    source = FakeSource()
    sink = CollectingSink()
    control = RuntimeControlPlane("session")
    previous = control.begin_turn("old-turn")
    observed = []

    def cleanup(previous_stamp, new_stamp):
        observed.append((previous_stamp, new_stamp, control.current))

    pipeline = SpeechPipeline(
        source=source,
        segmenter=EveryChunkIsUtterance(),
        engine=FakeEngine(),
        sink=sink,
        playback_gate=PlaybackGate(resume_delay_ms=0),
        control_plane=control,
        on_turn_superseded=cleanup,
    )
    pipeline.start()
    source.items.put(chunk())
    wait_for(lambda: len(sink.events) == 1)
    pipeline.close()

    new_stamp = sink.events[0].control_stamp
    assert observed == [(previous, new_stamp, new_stamp)]
    assert control.current == new_stamp
    assert control.invalidations == 0


def test_pipeline_warms_engine_before_starting_audio_capture():
    source = FakeSource()
    engine = FakeEngine(source)
    pipeline = SpeechPipeline(
        source=source,
        segmenter=EveryChunkIsUtterance(),
        engine=engine,
        sink=CollectingSink(),
        playback_gate=PlaybackGate(resume_delay_ms=0),
    )

    pipeline.start()

    assert engine.warmups == 1
    assert engine.warmup_thread_name == "speech-asr"
    assert engine.source_started_during_warmup is False
    assert source.started is True
    pipeline.close()


def test_playback_audio_is_suppressed_not_recognized():
    source = FakeSource()
    sink = CollectingSink()
    gate = PlaybackGate(resume_delay_ms=0)
    segmenter = EveryChunkIsUtterance()
    pipeline = SpeechPipeline(
        source=source,
        segmenter=segmenter,
        engine=FakeEngine(),
        sink=sink,
        playback_gate=gate,
    )
    pipeline.start()
    gate.set_active(True)
    source.items.put(chunk())
    wait_for(lambda: pipeline.metrics().playback_chunks_suppressed == 1)
    gate.set_active(False)
    source.items.put(chunk())
    wait_for(lambda: len(sink.events) == 1)
    pipeline.close()

    assert len(sink.events) == 1
    assert segmenter.reset_count >= 1


def test_full_duplex_keeps_capture_open_and_emits_one_barge_in_edge():
    class EdgeSegmenter(EveryChunkIsUtterance):
        def __init__(self):
            super().__init__()
            self.speech_active = True

        def accept(self, _chunk):
            return []

    source = FakeSource()
    gate = PlaybackGate(resume_delay_ms=0)
    segmenter = EdgeSegmenter()
    edges = []
    pipeline = SpeechPipeline(
        source=source,
        segmenter=segmenter,
        engine=FakeEngine(),
        sink=CollectingSink(),
        playback_gate=gate,
        suppress_during_playback=False,
        on_speech_start=lambda detected_ns: edges.append(detected_ns),
    )
    pipeline.start()
    gate.set_active(True)
    source.items.put(chunk())
    source.items.put(chunk())
    wait_for(lambda: pipeline.metrics().audio_chunks_received == 2)

    segmenter.speech_active = False
    source.items.put(chunk())
    wait_for(lambda: pipeline.metrics().audio_chunks_received == 3)
    segmenter.speech_active = True
    source.items.put(chunk())
    wait_for(lambda: pipeline.metrics().audio_chunks_received == 4)
    pipeline.close()

    assert len(edges) == 2
    assert all(isinstance(detected_ns, int) for detected_ns in edges)
    assert edges[1] >= edges[0]
    assert pipeline.metrics().playback_chunks_suppressed == 0


def test_audio_reconnect_resets_vad_state():
    source = FakeSource()
    segmenter = EveryChunkIsUtterance()
    pipeline = SpeechPipeline(
        source=source,
        segmenter=segmenter,
        engine=FakeEngine(),
        sink=CollectingSink(),
        playback_gate=PlaybackGate(resume_delay_ms=0),
    )
    pipeline.start()
    source.discontinuity_count += 1
    source.items.put(chunk())
    wait_for(lambda: segmenter.reset_count == 1)
    pipeline.close()


def test_close_resets_vad_before_pipeline_reactivation():
    segmenter = EveryChunkIsUtterance()
    pipeline = SpeechPipeline(
        source=FakeSource(),
        segmenter=segmenter,
        engine=FakeEngine(),
        sink=CollectingSink(),
        playback_gate=PlaybackGate(resume_delay_ms=0),
    )

    pipeline.start()
    pipeline.close()
    pipeline.start()
    pipeline.close()

    assert segmenter.reset_count == 2


def test_stop_keeps_model_loaded_but_close_releases_it():
    engine = FakeEngine()
    pipeline = SpeechPipeline(
        source=FakeSource(),
        segmenter=EveryChunkIsUtterance(),
        engine=engine,
        sink=CollectingSink(),
        playback_gate=PlaybackGate(resume_delay_ms=0),
    )

    pipeline.prepare()
    pipeline.prepare()
    pipeline.start()
    pipeline.stop()
    pipeline.start()
    pipeline.stop()

    assert engine.loads == 1
    assert engine.closes == 0
    pipeline.close()
    assert engine.closes == 1
