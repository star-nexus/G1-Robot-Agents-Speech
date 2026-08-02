"""Subscribe to speech events from a robot or Agent process."""

from __future__ import annotations

import argparse
import time
import uuid

from g1_speech.dds import (
    DdsPlaybackPublisher,
    DdsSpeechSubscriber,
    initialize_unitree_dds,
)


def on_speech(event) -> None:
    # Keep this callback short: enqueue into the Agent's perception queue.
    print(f"Agent perception <- [{event.event_id}] {event.text}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("network_interface")
    parser.add_argument("--domain", type=int, default=0)
    args = parser.parse_args()

    # In the real Agent, initialize DDS only once before SDK2 clients/subscribers.
    initialize_unitree_dds(args.domain, args.network_interface)
    speech = DdsSpeechSubscriber(on_speech)
    playback = DdsPlaybackPublisher()
    speech.start()
    playback.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        playback.close()
        speech.close()


def play_tts_without_self_trigger(audio_client, text: str, playback_seconds: float) -> None:
    """Example wrapper; keep the gate active until the asynchronous playback ends."""
    playback = DdsPlaybackPublisher()
    playback.start()
    request_id = str(uuid.uuid4())
    playback.set_active_reliably(True, request_id=request_id)
    try:
        audio_client.TtsMaker(text, 0)
        time.sleep(playback_seconds)
    finally:
        playback.set_active_reliably(False, request_id=request_id)
        playback.close()


if __name__ == "__main__":
    main()
