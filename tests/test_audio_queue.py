import numpy as np

from g1_speech.audio import SoundDeviceSource


class FakeStream:
    def __init__(self, **kwargs):
        self.callback = kwargs["callback"]
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        pass


class FakeSoundDevice:
    def __init__(self):
        self.stream = None

    def InputStream(self, **kwargs):
        self.stream = FakeStream(**kwargs)
        return self.stream


def test_audio_callback_queue_stays_bounded_under_overload():
    sd = FakeSoundDevice()
    source = SoundDeviceSource(
        sample_rate=16000,
        block_ms=100,
        queue_seconds=0.2,
        sounddevice_module=sd,
    )
    source.start()
    for _ in range(100):
        sd.stream.callback(np.zeros((1600, 1), dtype=np.float32), 1600, None, None)

    assert source.queue_capacity == 2
    assert source.queue_size == 2
    assert source.dropped_chunks > 0
    source.close()
