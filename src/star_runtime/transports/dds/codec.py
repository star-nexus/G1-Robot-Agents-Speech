"""Conversions between canonical HRI events and Cyclone DDS IDL messages."""

from ...core.events import SpeechEvent
from .types import SpeechEventMessage

def event_to_message(event: SpeechEvent) -> SpeechEventMessage:
    return SpeechEventMessage(
        event_id=event.event_id,
        session_id=event.session_id,
        sequence=event.sequence,
        created_unix_ns=event.created_unix_ns,
        source=event.source,
        text=event.text,
        language=event.language,
        audio_duration_ms=event.audio_duration_ms,
        inference_ms=event.inference_ms,
        engine=event.engine,
        is_final=event.is_final,
        turn_id=event.turn_id,
        epoch=event.epoch,
    )


def message_to_event(message: SpeechEventMessage) -> SpeechEvent:
    return SpeechEvent(
        event_id=message.event_id,
        session_id=message.session_id,
        sequence=int(message.sequence),
        created_unix_ns=int(message.created_unix_ns),
        source=message.source,
        text=message.text,
        language=message.language,
        audio_duration_ms=int(message.audio_duration_ms),
        inference_ms=float(message.inference_ms),
        engine=message.engine,
        is_final=bool(message.is_final),
        turn_id=getattr(message, "turn_id", ""),
        epoch=int(getattr(message, "epoch", 0)),
    )
