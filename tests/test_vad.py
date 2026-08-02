from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from g1_speech.config import VadConfig
from g1_speech.contracts import AudioChunk
from g1_speech.vad import SileroVadSegmenter, _SampleRingBuffer


def test_segmenter_uses_vad_config_without_local_defaults(tmp_path):
    model = tmp_path / "silero_vad.onnx"
    model.write_bytes(b"test")

    class FakeSherpa:
        detector_config = None
        detector_buffer_seconds = None
        detector = None

        class VadModelConfig:
            def __init__(self):
                self.silero_vad = SimpleNamespace(window_size=512)
                self.sample_rate = 0

        class Detector:
            def __init__(self):
                self.windows = []

            def accept_waveform(self, window):
                self.windows.append(window.copy())

            def empty(self):
                return True

        @classmethod
        def VoiceActivityDetector(cls, config, *, buffer_size_in_seconds):
            cls.detector_config = config
            cls.detector_buffer_seconds = buffer_size_in_seconds
            cls.detector = cls.Detector()
            return cls.detector

    settings = VadConfig(
        model=str(model),
        threshold=0.42,
        speech_pre_roll_seconds=0.6,
        min_silence_seconds=0.4,
        min_speech_seconds=0.2,
        max_speech_seconds=8.0,
        buffer_seconds=20.0,
    )

    segmenter = SileroVadSegmenter(settings=settings, sherpa_module=FakeSherpa)

    runtime = FakeSherpa.detector_config
    assert runtime.silero_vad.model == str(model)
    assert runtime.silero_vad.threshold == settings.threshold
    assert runtime.silero_vad.min_silence_duration == settings.min_silence_seconds
    assert runtime.silero_vad.min_speech_duration == settings.min_speech_seconds
    assert runtime.silero_vad.max_speech_duration == settings.max_speech_seconds
    assert FakeSherpa.detector_buffer_seconds == settings.buffer_seconds
    assert segmenter._pre_roll_samples == round(  # noqa: SLF001
        settings.speech_pre_roll_seconds * 16000
    )

    segmenter.accept(AudioChunk(np.arange(300, dtype=np.float32), 16000, 0))
    segmenter.accept(AudioChunk(np.arange(300, 1100, dtype=np.float32), 16000, 0))

    accepted = np.concatenate(FakeSherpa.detector.windows)
    np.testing.assert_array_equal(accepted, np.arange(1024, dtype=np.float32))
    assert segmenter._pending_size == 76  # noqa: SLF001


def test_sample_ring_buffer_preserves_order_after_wraparound():
    ring = _SampleRingBuffer(5)
    ring.append(np.array([0, 1, 2], dtype=np.float32))
    np.testing.assert_array_equal(ring.read(0, 3), [0, 1, 2])

    ring.append(np.array([3, 4, 5, 6], dtype=np.float32))

    assert ring.start_sample == 2
    assert ring.end_sample == 7
    assert ring.read(0, 2) is None
    np.testing.assert_array_equal(ring.read(2, 7), [2, 3, 4, 5, 6])
