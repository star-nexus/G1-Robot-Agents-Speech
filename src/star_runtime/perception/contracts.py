"""Small observation contracts kept independent from Agent and transport code."""

from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ImageFrame:
    """One encoded camera frame; encoding is deferred until a model needs it."""

    data: bytes
    captured_monotonic: float
    mime_type: str = "image/jpeg"

    def data_url(self) -> str:
        encoded = base64.b64encode(self.data).decode("ascii")
        return f"data:{self.mime_type};base64,{encoded}"


class ImageSource(Protocol):
    """Latest-frame camera port; local and remote cameras share this boundary."""

    def snapshot(self, max_age_seconds: float) -> ImageFrame: ...


class VisionInput(Protocol):
    """Selects an optional image for the current Agent turn."""

    def image_for(
        self,
        user_text: str,
        recent_messages: Sequence[object],
    ) -> ImageFrame | None: ...
