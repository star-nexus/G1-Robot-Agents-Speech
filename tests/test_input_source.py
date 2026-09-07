from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from g1_speech.audio import (
    AudioInputUnavailable,
    SoundDeviceSource,
    read_alsa_card_numbers,
    resolve_alsa_input_device,
)
from g1_speech.config import AudioConfig


class FakeStream:
    def __init__(self, kwargs, *, fail_devices=()):
        self.kwargs = kwargs
        self.latency = 0.012
        self.active = False
        self._fail_devices = set(fail_devices)

    def start(self):
        if self.kwargs["device"] in self._fail_devices:
            raise RuntimeError("device busy")
        self.active = True

    def abort(self):
        self.active = False

    def close(self):
        self.active = False


class FakeSoundDevice:
    def __init__(self, devices, *, fail_devices=()):
        self.devices = devices
        self.fail_devices = fail_devices
        self.streams = []

    def query_devices(self, device=None, _kind=None):
        if device is None:
            return self.devices
        if isinstance(device, int):
            return self.devices[device]
        for candidate in self.devices:
            if str(device).lower() in candidate["name"].lower():
                return candidate
        raise ValueError(f"unknown device: {device}")

    def InputStream(self, **kwargs):
        stream = FakeStream(kwargs, fail_devices=self.fail_devices)
        self.streams.append(stream)
        return stream


def make_card(proc_root: Path, number: int, card_id: str) -> None:
    directory = proc_root / f"card{number}"
    directory.mkdir(parents=True)
    (directory / "id").write_text(card_id + "\n", encoding="utf-8")


def devices():
    return [
        {"name": "Camera (hw:0,0)", "max_input_channels": 2},
        {"name": "Wireless Microphone (hw:7,0)", "max_input_channels": 2},
        {"name": "pulse", "max_input_channels": 32},
    ]


def test_stable_alsa_card_id_resolves_current_hotplug_number(tmp_path):
    make_card(tmp_path, 0, "camera")
    make_card(tmp_path, 7, "Microphone")
    sd = FakeSoundDevice(devices())

    assert read_alsa_card_numbers(tmp_path) == {"camera": 0, "Microphone": 7}
    assert resolve_alsa_input_device(
        sd, card_id="Microphone", pcm_device=0, proc_root=tmp_path
    ) == (1, "Wireless Microphone (hw:7,0)")


def test_source_prefers_alsa_hardware_with_small_low_latency_blocks(tmp_path):
    make_card(tmp_path, 7, "Microphone")
    sd = FakeSoundDevice(devices())
    source = SoundDeviceSource(
        settings=AudioConfig(alsa_card="Microphone"),
        sounddevice_module=sd,
        proc_asound_root=tmp_path,
    )

    source.start()
    try:
        assert source.active_backend == "alsa"
        assert source.active_device == "Wireless Microphone (hw:7,0)"
        assert source.block_samples == 320
        assert source.actual_latency_seconds == 0.012
        assert sd.streams[0].kwargs["device"] == 1
        assert sd.streams[0].kwargs["latency"] == "low"
        assert sd.streams[0].kwargs["samplerate"] == 48000
        assert sd.streams[0].kwargs["blocksize"] == 960
        assert sd.streams[0].kwargs["channels"] == 2
        assert sd.streams[0].kwargs["dtype"] == "int16"

        native = np.full((960, 2), 16384, dtype=np.int16)
        sd.streams[0].kwargs["callback"](native, 960, None, None)
        chunk = source.read()
        assert chunk is not None
        assert chunk.sample_rate == 16000
        assert chunk.samples.shape == (320,)
        np.testing.assert_allclose(chunk.samples[-100:], 0.5, atol=1e-5)
    finally:
        source.close()


def test_source_acoustic_trace_uses_one_capture_timestamp_for_raw_and_post(tmp_path):
    class Trace:
        def __init__(self):
            self.raw = []
            self.post = []

        def record_raw_capture(self, samples, sample_rate, captured_ns):
            self.raw.append((samples.copy(), sample_rate, captured_ns))

        def record_post_aec_capture(self, samples, sample_rate, captured_ns):
            self.post.append((samples.copy(), sample_rate, captured_ns))

    make_card(tmp_path, 7, "Microphone")
    sd = FakeSoundDevice(devices())
    trace = Trace()
    source = SoundDeviceSource(
        settings=AudioConfig(alsa_card="Microphone"),
        sounddevice_module=sd,
        proc_asound_root=tmp_path,
    )
    source.set_acoustic_trace_sink(trace)
    source.start()
    try:
        native = np.full((960, 2), 8192, dtype=np.int16)
        sd.streams[0].kwargs["callback"](native, 960, None, None)
        chunk = source.read()
        assert chunk is not None
    finally:
        source.close()

    assert len(trace.raw) == len(trace.post) == 1
    assert trace.raw[0][1:] == trace.post[0][1:]
    assert trace.raw[0][2] == chunk.captured_monotonic_ns
    np.testing.assert_array_equal(trace.raw[0][0], trace.post[0][0])


def test_source_falls_back_to_pulse_when_hardware_is_busy(tmp_path, caplog):
    make_card(tmp_path, 7, "Microphone")
    sd = FakeSoundDevice(devices(), fail_devices=(1,))
    source = SoundDeviceSource(
        settings=AudioConfig(alsa_card="Microphone", fallback_backend="pulse"),
        sounddevice_module=sd,
        proc_asound_root=tmp_path,
    )

    source.start()
    try:
        assert source.active_backend == "pulse"
        assert source.active_device == "pulse"
        assert [stream.kwargs["device"] for stream in sd.streams] == [1, "pulse"]
        assert "falling back to pulse" in caplog.text
    finally:
        source.close()


def test_source_fails_closed_without_explicit_pulse_fallback(tmp_path):
    make_card(tmp_path, 7, "Microphone")
    sd = FakeSoundDevice(devices(), fail_devices=(1,))
    source = SoundDeviceSource(
        settings=AudioConfig(alsa_card="Microphone"),
        sounddevice_module=sd,
        proc_asound_root=tmp_path,
    )

    with pytest.raises(AudioInputUnavailable, match="device busy"):
        source.start()


def test_native_48khz_capture_is_low_pass_filtered_before_decimation(tmp_path):
    make_card(tmp_path, 7, "Microphone")

    def convert_tone(frequency: float) -> np.ndarray:
        sd = FakeSoundDevice(devices())
        source = SoundDeviceSource(
            settings=AudioConfig(alsa_card="Microphone"),
            sounddevice_module=sd,
            proc_asound_root=tmp_path,
        )
        source.start()
        try:
            time_axis = np.arange(4800, dtype=np.float64) / 48000
            mono = (0.8 * np.sin(2 * np.pi * frequency * time_axis) * 32767).astype(
                np.int16
            )
            stereo = np.column_stack((mono, mono))
            output = []
            for offset in range(0, stereo.shape[0], 960):
                sd.streams[0].kwargs["callback"](
                    stereo[offset : offset + 960], 960, None, None
                )
                chunk = source.read()
                assert chunk is not None
                output.append(chunk.samples)
            return np.concatenate(output)[100:]
        finally:
            source.close()

    passband_rms = np.sqrt(np.mean(np.square(convert_tone(1000))))
    stopband_rms = np.sqrt(np.mean(np.square(convert_tone(12000))))

    assert passband_rms > 0.5
    assert stopband_rms < passband_rms * 0.05
