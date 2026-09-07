from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from g1_speech.audio_output import AlsaOutputPlayer, resolve_alsa_output_device
from g1_speech.audio_processing import PassthroughAudioProcessor
from g1_speech.config import AudioOutputConfig
from star_runtime.core.control import ControlStamp
from star_runtime.core.timing import RuntimeTimingAudit


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


class UnderflowRawOutputStream(FakeRawOutputStream):
    def write(self, block):
        super().write(block)
        return True


class UnderflowSoundDevice(FakeSoundDevice):
    def RawOutputStream(self, **kwargs):
        self.stream = UnderflowRawOutputStream(kwargs)
        return self.stream


class RestartBlockingRawOutputStream(FakeRawOutputStream):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.start_calls = 0
        self.restart_started = threading.Event()
        self.release_restart = threading.Event()

    def start(self):
        self.start_calls += 1
        if self.start_calls > 1:
            self.restart_started.set()
            self.release_restart.wait(2.0)
        self.active = True


class RestartBlockingSoundDevice(FakeSoundDevice):
    def RawOutputStream(self, **kwargs):
        self.stream = RestartBlockingRawOutputStream(kwargs)
        return self.stream


class CorruptingCommitPlayer(AlsaOutputPlayer):
    def _submit_hardware_write(
        self,
        stream,
        output,
        reference,
        generation,
        commit_sequence,
    ):
        return super()._submit_hardware_write(
            stream,
            output,
            reference,
            generation - 1,
            commit_sequence,
        )


class RenderCollector(PassthroughAudioProcessor):
    def __init__(self):
        super().__init__("hardware")
        self.render = []

    def push_render(self, samples, sample_rate):
        self.render.append((np.asarray(samples), sample_rate))


class ContinuousRenderCollector(RenderCollector):
    def __init__(self):
        super().__init__()
        self.mode = "webrtc"


class PacedRawOutputStream(FakeRawOutputStream):
    """Small device-clock stand-in that prevents a fake-stream busy loop."""

    def write(self, block):
        super().write(block)
        time.sleep(0.005)


class PacedSoundDevice(FakeSoundDevice):
    def RawOutputStream(self, **kwargs):
        self.stream = PacedRawOutputStream(kwargs)
        return self.stream


def wait_for(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return False


class InterruptingRenderCollector(RenderCollector):
    def __init__(self):
        super().__init__()
        self.player = None
        self.interrupted = False

    def push_render(self, samples, sample_rate):
        if self.player is not None and not self.interrupted:
            self.interrupted = True
            self.player.abort()
        super().push_render(samples, sample_rate)


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
        AudioOutputConfig(
            alsa_card="RobotSpeaker",
            interrupt_strategy="hard_abort",
        ),
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


def test_output_audit_tags_pcm_through_dequeue_and_exact_hardware_commit(tmp_path):
    make_card(tmp_path, 4, "RobotSpeaker")
    audit = RuntimeTimingAudit()
    stamp = ControlStamp("session", "turn-audio", 1)
    player = AlsaOutputPlayer(
        AudioOutputConfig(alsa_card="RobotSpeaker"),
        render_sink=RenderCollector(),
        sounddevice_module=FakeSoundDevice(),
        proc_asound_root=tmp_path,
        timing_audit=audit,
    )
    player.start()
    player.set_control_stamp(stamp)
    assert player.enqueue(np.zeros(480, dtype="<i2").tobytes(), 48000)
    assert player.wait_until_idle(timeout=1.0)
    player.close()

    events = audit.snapshot(stamp)
    assert events["playback_dequeue"].monotonic_ns <= events[
        "playback_prewrite_commit"
    ].monotonic_ns
    assert events["playback_prewrite_commit"].monotonic_ns <= events[
        "hardware_write_begin"
    ].monotonic_ns
    assert events["hardware_write_begin"].monotonic_ns <= events[
        "hardware_write_complete"
    ].monotonic_ns


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


def test_webrtc_render_clock_submits_silence_and_exact_speaker_pcm_continuously(
    tmp_path,
):
    make_card(tmp_path, 4, "RobotSpeaker")
    sd = PacedSoundDevice()
    render = ContinuousRenderCollector()
    player = AlsaOutputPlayer(
        AudioOutputConfig(
            alsa_card="RobotSpeaker",
            sample_rate=48000,
            block_ms=10,
            interrupt_strategy="persistent",
        ),
        render_sink=render,
        sounddevice_module=sd,
        proc_asound_root=tmp_path,
    )
    player.start()
    assert wait_for(lambda: len(sd.stream.writes) >= 2)

    payload = np.full(480, 1234, dtype="<i2")
    assert player.enqueue(payload.tobytes(), 48000)
    assert player.wait_until_idle(timeout=1.0)
    assert wait_for(lambda: len(sd.stream.writes) >= 4)
    player.close()

    speaker_blocks = [
        np.frombuffer(block, dtype="<i2").reshape(-1, 2)
        for block in sd.stream.writes
    ]
    payload_indexes = [
        index
        for index, block in enumerate(speaker_blocks)
        if np.any(block)
    ]
    assert len(payload_indexes) == 1
    payload_index = payload_indexes[0]
    np.testing.assert_array_equal(speaker_blocks[payload_index][:, 0], payload)
    np.testing.assert_array_equal(
        speaker_blocks[payload_index][:, 0],
        speaker_blocks[payload_index][:, 1],
    )
    assert len(render.render) == len(speaker_blocks)
    for (reference, rate), speaker in zip(render.render, speaker_blocks):
        assert rate == 48000
        np.testing.assert_array_equal(
            reference,
            speaker[:, 0].astype(np.float32) / 32768.0,
        )

    metrics = player.metrics()
    assert metrics["audio_output_continuous_render_clock"] is True
    assert metrics["audio_output_blocks"] == 1
    assert metrics["audio_output_render_clock_silence_blocks"] >= 3
    assert metrics["audio_output_stale_blocks_dropped_before_write"] == 0
    assert metrics["audio_output_residual_writes_committed"] == 0


def test_webrtc_clock_silence_during_invalidation_is_not_residual_audio(tmp_path):
    make_card(tmp_path, 4, "RobotSpeaker")
    sd = PacedSoundDevice()
    player = AlsaOutputPlayer(
        AudioOutputConfig(
            alsa_card="RobotSpeaker",
            block_ms=10,
            interrupt_strategy="persistent",
        ),
        render_sink=ContinuousRenderCollector(),
        sounddevice_module=sd,
        proc_asound_root=tmp_path,
    )
    player.start()
    assert wait_for(
        lambda: player.metrics()["audio_output_hardware_write_kind"]
        == "aec_silence"
    )
    player.abort()
    assert wait_for(
        lambda: player.metrics()["audio_output_render_clock_silence_blocks"] >= 1
    )

    metrics = player.metrics()
    assert metrics["audio_output_residual_writes_committed"] == 0
    assert metrics["audio_output_residual_write_completions"] == 0
    assert metrics["audio_output_stale_blocks_dropped_before_write"] == 0
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


def test_hard_abort_restarts_inactive_stream_and_reports_recovery(tmp_path):
    make_card(tmp_path, 4, "RobotSpeaker")
    sd = FakeSoundDevice()
    player = AlsaOutputPlayer(
        AudioOutputConfig(
            alsa_card="RobotSpeaker",
            interrupt_strategy="hard_abort",
        ),
        render_sink=RenderCollector(),
        sounddevice_module=sd,
        proc_asound_root=tmp_path,
    )
    player.start()
    player.abort()
    player.note_pcm_ready()
    assert player.enqueue(np.full(480, 2000, dtype="<i2").tobytes(), 48000)
    assert player.wait_until_idle(timeout=1.0)

    metrics = player.metrics()
    assert sd.stream.aborts == 1
    assert metrics["audio_output_stream_restart_attempts"] == 1
    assert metrics["audio_output_stream_restart_successes"] == 1
    assert metrics["audio_output_recoveries"] == 1
    assert metrics["audio_output_recovery_pending"] is False
    assert metrics["audio_output_forbidden_stale_write_attempts"] == 0
    player.close()


def test_persistent_interrupt_flushes_old_queue_and_recovers_after_residual_write(
    tmp_path,
):
    make_card(tmp_path, 4, "RobotSpeaker")
    sd = BlockingSoundDevice()
    render = RenderCollector()
    player = AlsaOutputPlayer(
        AudioOutputConfig(
            alsa_card="RobotSpeaker",
            interrupt_strategy="persistent",
            buffer_seconds=0.1,
        ),
        render_sink=render,
        sounddevice_module=sd,
        proc_asound_root=tmp_path,
    )
    player.start()

    old = np.full(480 * 3, 1000, dtype="<i2").tobytes()
    assert player.enqueue(old, 48000)
    assert sd.stream.write_started.wait(1.0)
    player.abort()
    player.note_pcm_ready()
    replacement = np.full(480, 2000, dtype="<i2").tobytes()
    assert player.enqueue(replacement, 48000)
    sd.stream.release_write.set()
    assert player.wait_until_idle(timeout=1.0)

    metrics = player.metrics()
    assert sd.stream.aborts == 0
    assert len(sd.stream.writes) == 2
    first = np.frombuffer(sd.stream.writes[0], dtype="<i2").reshape(-1, 2)
    second = np.frombuffer(sd.stream.writes[1], dtype="<i2").reshape(-1, 2)
    assert np.all(first == 1000)
    assert np.all(second == 2000)
    assert len(render.render) == 2
    assert metrics["audio_output_residual_writes_committed"] == 1
    assert metrics["audio_output_residual_write_completions"] == 1
    assert metrics["audio_output_recoveries"] == 1
    assert metrics["audio_output_stream_restart_attempts"] == 0
    assert metrics["audio_output_forbidden_stale_write_attempts"] == 0
    player.close()


def test_portaudio_underflow_result_is_counted(tmp_path):
    make_card(tmp_path, 4, "RobotSpeaker")
    player = AlsaOutputPlayer(
        AudioOutputConfig(alsa_card="RobotSpeaker"),
        render_sink=RenderCollector(),
        sounddevice_module=UnderflowSoundDevice(),
        proc_asound_root=tmp_path,
    )
    player.start()
    assert player.enqueue(np.zeros(480, dtype="<i2").tobytes(), 48000)
    assert player.wait_until_idle(timeout=1.0)

    assert player.metrics()["audio_output_write_underflows"] == 1
    player.close()


def test_invalidation_before_final_gate_drops_block_without_hardware_or_aec(tmp_path):
    make_card(tmp_path, 4, "RobotSpeaker")
    sd = RestartBlockingSoundDevice()
    render = RenderCollector()
    player = AlsaOutputPlayer(
        AudioOutputConfig(alsa_card="RobotSpeaker"),
        render_sink=render,
        sounddevice_module=sd,
        proc_asound_root=tmp_path,
    )
    player.start()
    sd.stream.active = False
    assert player.enqueue(np.zeros(480, dtype="<i2").tobytes(), 48000)
    assert sd.stream.restart_started.wait(1.0)

    player.abort()
    sd.stream.release_restart.set()
    assert player.wait_until_idle(timeout=1.0)

    metrics = player.metrics()
    assert sd.stream.writes == []
    assert render.render == []
    assert metrics["audio_output_stale_blocks_dropped_before_write"] == 1
    assert metrics["audio_output_residual_writes_committed"] == 0
    assert metrics["audio_output_forbidden_stale_write_attempts"] == 0
    player.close()


def test_invalidation_immediately_after_final_gate_is_committed_residual(tmp_path):
    make_card(tmp_path, 4, "RobotSpeaker")
    sd = FakeSoundDevice()
    render = InterruptingRenderCollector()
    player = AlsaOutputPlayer(
        AudioOutputConfig(alsa_card="RobotSpeaker"),
        render_sink=render,
        sounddevice_module=sd,
        proc_asound_root=tmp_path,
    )
    render.player = player
    player.start()
    assert player.enqueue(np.full(480, 1234, dtype="<i2").tobytes(), 48000)
    assert player.wait_until_idle(timeout=1.0)

    metrics = player.metrics()
    assert len(sd.stream.writes) == 1
    assert len(render.render) == 1
    assert metrics["audio_output_stale_blocks_dropped_before_write"] == 0
    assert metrics["audio_output_residual_writes_committed"] == 1
    assert metrics["audio_output_residual_write_completions"] == 1
    assert metrics["audio_output_forbidden_stale_write_attempts"] == 0
    player.close()


def test_stale_commit_at_hardware_boundary_is_counted_and_rejected(tmp_path):
    make_card(tmp_path, 4, "RobotSpeaker")
    sd = FakeSoundDevice()
    render = RenderCollector()
    player = CorruptingCommitPlayer(
        AudioOutputConfig(alsa_card="RobotSpeaker"),
        render_sink=render,
        sounddevice_module=sd,
        proc_asound_root=tmp_path,
    )
    player.start()
    assert player.enqueue(np.zeros(480, dtype="<i2").tobytes(), 48000)
    assert player.wait_until_idle(timeout=1.0)

    metrics = player.metrics()
    assert sd.stream.writes == []
    assert render.render == []
    assert metrics["audio_output_blocks"] == 0
    assert metrics["audio_output_forbidden_stale_write_attempts"] == 1
    player.close()


def test_every_exported_playback_metric_is_documented(tmp_path):
    player = AlsaOutputPlayer(
        AudioOutputConfig(alsa_card="RobotSpeaker"),
        render_sink=RenderCollector(),
        sounddevice_module=FakeSoundDevice(),
        proc_asound_root=tmp_path,
    )
    metrics = player.metrics()
    documentation = (
        Path(__file__).parents[1] / "docs" / "playback_metrics.md"
    ).read_text(encoding="utf-8")

    assert "audio_output_post_invalidation_stale_write_submissions" not in metrics
    for name in metrics:
        assert f"`{name}`" in documentation
