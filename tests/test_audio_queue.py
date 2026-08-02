import time

import numpy as np

from g1_speech.audio import SoundDeviceSource
from g1_speech.config import AudioConfig


class FakeStream:
    def __init__(self, **kwargs):
        self.callback = kwargs["callback"]
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def abort(self):
        self.started = False

    def close(self):
        pass

    @property
    def active(self):
        return self.started


class FakeSoundDevice:
    def __init__(self):
        self.stream = None
        self.streams = []
        self.fail_opens = 0

    def InputStream(self, **kwargs):
        if self.fail_opens:
            self.fail_opens -= 1
            raise RuntimeError("device unavailable")
        self.stream = FakeStream(**kwargs)
        self.streams.append(self.stream)
        return self.stream


def test_audio_callback_queue_stays_bounded_under_overload():
    sd = FakeSoundDevice()
    source = SoundDeviceSource(
        settings=AudioConfig(queue_seconds=0.2),
        sounddevice_module=sd,
    )
    source.start()
    for _ in range(100):
        sd.stream.callback(np.zeros((1600, 1), dtype=np.float32), 1600, None, None)

    assert source.queue_capacity == 2
    assert source.queue_size == 2
    assert source.dropped_chunks > 0
    source.close()


def test_inactive_audio_stream_is_reconnected():
    sd = FakeSoundDevice()
    source = SoundDeviceSource(
        settings=AudioConfig(
            heartbeat_timeout_seconds=0.01,
            reconnect_initial_seconds=0.001,
            reconnect_max_seconds=0.01,
        ),
        sounddevice_module=sd,
    )
    source.start()
    first_stream = sd.stream
    first_stream.started = False

    source.read(timeout=0)

    assert len(sd.streams) == 2
    assert sd.stream is not first_stream
    assert sd.stream.active
    assert source.reconnections == 1
    assert source.discontinuity_count == 1
    source.close()


def test_missing_audio_heartbeat_is_reconnected():
    sd = FakeSoundDevice()
    source = SoundDeviceSource(
        settings=AudioConfig(
            heartbeat_timeout_seconds=0.01,
            reconnect_initial_seconds=0.001,
            reconnect_max_seconds=0.01,
        ),
        sounddevice_module=sd,
    )
    source.start()
    source._last_callback_monotonic = time.monotonic() - 1  # noqa: SLF001

    source.read(timeout=0)

    assert len(sd.streams) == 2
    assert source.reconnections == 1
    source.close()


def test_reconnect_failures_back_off_and_eventually_recover():
    sd = FakeSoundDevice()
    source = SoundDeviceSource(
        settings=AudioConfig(
            reconnect_initial_seconds=1.0,
            reconnect_max_seconds=4.0,
        ),
        sounddevice_module=sd,
    )
    source.start()
    sd.stream.started = False
    sd.fail_opens = 1

    source.read(timeout=0)
    assert source.reconnect_failures == 1
    assert source.reconnections == 0

    source.read(timeout=0)
    assert source.reconnect_failures == 1

    source._next_reconnect_monotonic = 0  # noqa: SLF001
    source.read(timeout=0)
    assert source.reconnections == 1
    assert source.discontinuity_count == 1
    source.close()
