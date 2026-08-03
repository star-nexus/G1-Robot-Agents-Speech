from __future__ import annotations

import queue
import time

import numpy as np

from g1_speech.contracts import AudioChunk, RecognitionResult, Utterance
from g1_speech.gate import PlaybackGate
from g1_speech.pipeline import SpeechPipeline


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
    def load(self):
        pass

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


def test_pipeline_publishes_final_event():
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
