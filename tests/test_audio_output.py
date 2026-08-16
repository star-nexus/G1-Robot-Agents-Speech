from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from g1_speech.audio_output import AlsaOutputPlayer, resolve_alsa_output_device
from g1_speech.audio_processing import PassthroughAudioProcessor
from g1_speech.config import AudioOutputConfig


def make_card(proc_root: Path, number: int, card_id: str) -> None:
    directory = proc_root / f"card{number}"
    directory.mkdir(parents=True)
    (directory / "id").write_text(card_id + "\n", encoding="utf-8")


class FakeRawOutputStream:
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.active = False
        self.latency = 0.01
        self.writes = []
        self.aborts = 0

    def start(self):
        self.active = True

    def write(self, block):
        self.writes.append(block)

    def abort(self):
        self.aborts += 1
        self.active = False

    def close(self):
        self.active = False


class FakeSoundDevice:
    def __init__(self):
        self.devices = [
            {"name": "Built-in (hw:0,0)", "max_output_channels": 2},
            {"name": "Robot Speaker (hw:4,0)", "max_output_channels": 2},
        ]
        self.stream = None

    def query_devices(self):
        return self.devices

    def RawOutputStream(self, **kwargs):
        self.stream = FakeRawOutputStream(kwargs)
        return self.stream


class BlockingRawOutputStream(FakeRawOutputStream):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.write_started = threading.Event()
        self.release_write = threading.Event()

    def write(self, block):
        self.write_started.set()
        self.release_write.wait(2.0)
        super().write(block)

    def abort(self):
        self.release_write.set()
        super().abort()


class BlockingSoundDevice(FakeSoundDevice):
    def RawOutputStream(self, **kwargs):
        self.stream = BlockingRawOutputStream(kwargs)
        return self.stream


class RenderCollector(PassthroughAudioProcessor):
    def __init__(self):
        super().__init__("hardware")
        self.render = []

    def push_render(self, samples, sample_rate):
        self.render.append((np.asarray(samples), sample_rate))


def test_output_resolves_stable_card_and_tees_exact_blocks_to_aec(tmp_path):
    make_card(tmp_path, 0, "tegra")
    make_card(tmp_path, 4, "RobotSpeaker")
    sd = FakeSoundDevice()

    assert resolve_alsa_output_device(
        sd,
        card_id="RobotSpeaker",
        pcm_device=0,
        proc_root=tmp_path,
    ) == (1, "Robot Speaker (hw:4,0)")

    render = RenderCollector()
    player = AlsaOutputPlayer(
        AudioOutputConfig(alsa_card="RobotSpeaker"),
        render_sink=render,
        sounddevice_module=sd,
        proc_asound_root=tmp_path,
    )
    player.start()
    assert player.enqueue(np.zeros(240, dtype="<i2").tobytes(), 24000)
    assert player.wait_until_idle(timeout=1.0)

    assert sd.stream.kwargs["samplerate"] == 48000
    assert len(sd.stream.writes) == 1
    assert len(sd.stream.writes[0]) == 480 * 2 * 2
    assert len(render.render) == 1
    assert render.render[0][0].shape == (480,)
    assert render.render[0][1] == 48000

    player.abort()
    assert sd.stream.aborts == 1
    player.close()


def test_output_joins_adjacent_chunks_into_one_continuous_hardware_block(tmp_path):
    make_card(tmp_path, 4, "RobotSpeaker")
    sd = FakeSoundDevice()
    render = RenderCollector()
    player = AlsaOutputPlayer(
        AudioOutputConfig(
            alsa_card="RobotSpeaker",
            sample_rate=48000,
            block_ms=10,
            buffer_seconds=0.1,
        ),
        render_sink=render,
        sounddevice_module=sd,
        proc_asound_root=tmp_path,
    )
    player.start()

    # Two independently enqueued 5 ms fragments become one uninterrupted
    # 10 ms hardware block. The player has no sentence/request boundary API.
    fragment = np.zeros(240, dtype="<i2").tobytes()
    assert player.enqueue(fragment, 48000)
    assert player.enqueue(fragment, 48000)
    assert player.wait_until_idle(timeout=1.0)

    assert len(sd.stream.writes) == 1
    assert len(render.render) == 1
    assert player.metrics()["audio_output_peak_buffered_ms"] == 10.0
    player.close()


def test_abort_clears_ring_and_releases_generation_waiting_on_backpressure(tmp_path):
    make_card(tmp_path, 4, "RobotSpeaker")
    sd = BlockingSoundDevice()
    player = AlsaOutputPlayer(
        AudioOutputConfig(
            alsa_card="RobotSpeaker",
            sample_rate=48000,
            block_ms=10,
            buffer_seconds=0.1,
        ),
        render_sink=RenderCollector(),
        sounddevice_module=sd,
        proc_asound_root=tmp_path,
    )
    player.start()
    result = []
    producer = threading.Thread(
        target=lambda: result.append(
            player.enqueue(np.zeros(9600, dtype="<i2").tobytes(), 48000)
        )
    )
    producer.start()

    assert sd.stream.write_started.wait(1.0)
    assert producer.is_alive()
    player.abort()
    producer.join(timeout=1.0)

    assert result == [False]
    assert player.metrics()["audio_output_buffered_ms"] <= 10.0
    assert player.metrics()["audio_output_backpressure_waits"] >= 1
    player.close()
