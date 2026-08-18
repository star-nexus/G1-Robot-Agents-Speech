from __future__ import annotations

import numpy as np

from g1_speech.audio_processing import (
    PassthroughAudioProcessor,
    WebRtcApmProcessor,
    create_audio_processor,
)
from g1_speech.config import AudioProcessingConfig


class FakeApm:
    def __init__(self):
        self.stream_format = None
        self.reverse_format = None
        self.delay = None
        self.ns_level = None
        self.capture = []
        self.render = []

    def set_stream_format(self, *args):
        self.stream_format = args

    def set_reverse_stream_format(self, *args):
        self.reverse_format = args

    def set_stream_delay(self, delay):
        self.delay = delay

    def set_ns_level(self, level):
        self.ns_level = level

    def process_stream(self, frame):
        self.capture.append(frame)
        return frame

    def process_reverse_stream(self, frame):
        self.render.append(frame)
        return frame


def test_processing_modes_are_pluggable_without_native_dependency():
    off = create_audio_processor(AudioProcessingConfig(mode="off"))
    hardware = create_audio_processor(AudioProcessingConfig(mode="hardware"))

    assert isinstance(off, PassthroughAudioProcessor)
    assert off.metrics()["aec_active"] is False
    assert hardware.metrics()["aec_active"] is True


def test_webrtc_apm_receives_10ms_capture_and_resampled_render_frames():
    native = FakeApm()
    processor = WebRtcApmProcessor(
        AudioProcessingConfig(mode="webrtc", stream_delay_ms=70),
        apm=native,
    )

    capture = np.linspace(-0.25, 0.25, 320, dtype=np.float32)
    output = processor.process_capture(capture, 16000)
    render = np.zeros(240, dtype=np.float32)  # 10 ms at Qwen3-TTS's 24 kHz
    processor.push_render(render, 24000)

    assert output.shape == capture.shape
    assert len(native.capture) == 2
    assert all(len(frame) == 320 for frame in native.capture)
    assert len(native.render) == 1
    assert len(native.render[0]) == 320
    assert native.delay == 70
    assert processor.metrics()["webrtc_capture_frames"] == 2
    assert processor.metrics()["webrtc_render_frames"] == 1
