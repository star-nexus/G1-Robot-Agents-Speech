#!/usr/bin/env python3
"""Live provider/ALSA control-plane check for JetPack Orin deployments.

This injects finalized user turns at the ASR -> Agent boundary so it can be run
unattended.  It complements, but does not replace, the final human VAD barge-in
check described in docs/runtime_architecture.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import replace

from star_runtime.agent import load_role_package
from star_runtime.apps.integrated_runtime import build_integrated_runtime
from star_runtime.apps.local_voice_agent import LocalAgentSettings
from star_runtime.core.events import SpeechEvent
from star_runtime.speech.config import load_config


def wait_for(predicate, description: str, timeout: float = 90.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise TimeoutError(f"timed out waiting for {description}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--role-package")
    parser.add_argument("--voice", default="Ono_Anna")
    parser.add_argument("--force-cpu-asr", action="store_true")
    parser.add_argument(
        "--audio-block-ms",
        type=int,
        default=20,
        help="production microphone block timing (default: 20 ms)",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    config = load_config(args.config, runtime_environment=os.environ)
    config = replace(
        config,
        audio=replace(config.audio, block_ms=args.audio_block_ms),
    )
    if args.force_cpu_asr:
        config = replace(
            config,
            sensevoice=replace(config.sensevoice, device="cpu"),
        )
    role = load_role_package(args.role_package) if args.role_package else None
    settings = LocalAgentSettings(
        system_prompt=role.prompt if role is not None else "回答适合直接朗读。",
        max_tokens=128,
        history_turns=3,
        tts_voice=args.voice,
        tts_language="Chinese",
    )
    runtime = build_integrated_runtime(config, settings, role_package=role)

    def inject(turn_id: str, text: str) -> None:
        stamp = runtime.speech.control.begin_turn(turn_id)
        event = SpeechEvent(
            event_id=turn_id,
            session_id=stamp.session_id,
            sequence=int(turn_id.rsplit("-", 1)[-1]),
            created_unix_ns=time.time_ns(),
            source="live-control-acceptance",
            text=text,
            language="zh",
            audio_duration_ms=1000,
            inference_ms=0.0,
            engine="injected-final-asr",
            turn_id=stamp.turn_id,
            epoch=stamp.epoch,
        )
        if not runtime.speech.transport.sink.publish(event):
            raise RuntimeError(f"in-process transport rejected {turn_id}")

    try:
        runtime.start()
        logging.info(
            "LIVE_MARKER robot-only false-barge-in check started; "
            "keep the near-end microphone area silent"
        )
        robot_only_invalidations = int(runtime.metrics()["control_invalidations"])
        robot_only_interruptions = int(
            runtime.metrics()["audio_output_interruptions"]
        )
        inject(
            "live-turn-0",
            "请只用一句不换行的完整中文句子，介绍一位可靠的朋友为什么值得信任。",
        )
        wait_for(
            lambda: bool(runtime.metrics()["tts_playback_active"]),
            "robot-only speech playback start",
        )
        wait_for(
            lambda: not bool(runtime.metrics()["tts_playback_active"]),
            "robot-only speech playback completion",
        )
        time.sleep(0.5)
        invalidation_delta = (
            int(runtime.metrics()["control_invalidations"])
            - robot_only_invalidations
        )
        interruption_delta = (
            int(runtime.metrics()["audio_output_interruptions"])
            - robot_only_interruptions
        )
        if invalidation_delta or interruption_delta:
            failure_metrics = runtime.metrics()
            raise RuntimeError(
                "robot-only speech caused a false acoustic barge-in: "
                f"invalidations={invalidation_delta} "
                f"playback_interruptions={interruption_delta} "
                f"capture_frames={failure_metrics.get('webrtc_capture_frames')} "
                f"render_frames={failure_metrics.get('webrtc_render_frames')} "
                f"capture_errors={failure_metrics.get('webrtc_capture_errors')} "
                f"render_errors={failure_metrics.get('webrtc_render_errors')}"
            )
        logging.info(
            "LIVE_MARKER robot-only speech produced zero false VAD "
            "barge-ins and zero playback interrupts"
        )

        baseline = int(runtime.metrics()["audio_output_blocks"])
        inject(
            "live-turn-1",
            "请详细数到五十，每个数字都说出来，不要省略。",
        )
        wait_for(
            lambda: int(runtime.metrics()["audio_output_blocks"]) > baseline,
            "first turn PCM playback",
        )
        logging.info("LIVE_MARKER old turn PCM externally visible")
        if not runtime.speech.cancel_active_turn("live-barge-in"):
            raise RuntimeError("first live turn was not active")
        logging.info("LIVE_MARKER old epoch invalidated and PCM timeline flushed")

        inject("live-turn-2", "打断成功了吗？请只回答成功。")
        first_blocks = int(runtime.metrics()["audio_output_blocks"])
        wait_for(
            lambda: int(runtime.metrics()["audio_output_blocks"]) > first_blocks,
            "replacement turn PCM playback",
        )
        wait_for(
            lambda: not bool(runtime.metrics()["tts_playback_active"]),
            "replacement turn completion",
        )
        logging.info("LIVE_MARKER replacement turn completed")

        inject("live-turn-3", "这是正常连续第三轮，请回答第三轮完成。")
        second_blocks = int(runtime.metrics()["audio_output_blocks"])
        wait_for(
            lambda: int(runtime.metrics()["audio_output_blocks"]) > second_blocks,
            "third turn PCM playback",
        )
        wait_for(
            lambda: not bool(runtime.metrics()["tts_playback_active"]),
            "third turn completion",
        )
        logging.info("LIVE_MARKER normal consecutive turn completed")
        print(json.dumps(runtime.metrics(), ensure_ascii=False, sort_keys=True))
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
