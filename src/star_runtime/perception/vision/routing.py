"""Decide when a conversational turn needs a fresh camera frame."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from ..contracts import ImageFrame, ImageSource


VISION = "vision"
TEXT = "text"
UNCERTAIN = "uncertain"
VisionClassifier = Callable[[str, Sequence[object], bool], tuple[str, str]]

_EXPLICIT_VISUAL = re.compile(
    r"(?:你|您)?(?:看|看看|看下|看一下|观察|识别|辨认|描述|检查|瞧|瞧瞧)"
    r"|(?:画面|镜头|摄像头|眼前|面前|周围|现场|这里).{0,10}"
    r"(?:什么|谁|哪|有|是|在|颜色|状态|样子)"
    r"|(?:什么|谁|哪|有|是|在|颜色|状态|样子).{0,10}"
    r"(?:画面|镜头|摄像头|眼前|面前|周围|现场|这里)",
    re.IGNORECASE,
)
_RESCUE_VISUAL = re.compile(
    r"(?:不对|错了|不是|再看|重新看|仔细看|看清楚|你没看|看错)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class VisionRouteDecision:
    requested_route: str
    effective_route: str
    source: str
    reason: str
    explicit_trigger: str | None = None
    classifier_error: str | None = None

    @property
    def attach_image(self) -> bool:
        return self.effective_route == VISION


class VisionRouter:
    """Fast rules first, optional semantic classifier second, fail-safe to vision."""

    def __init__(
        self,
        *,
        mode: str = "auto",
        classifier: VisionClassifier | None = None,
    ) -> None:
        if mode not in {"auto", "always", "off"}:
            raise ValueError("vision mode must be auto, always, or off")
        if mode == "auto" and classifier is None:
            raise ValueError("auto vision mode requires a classifier")
        self._mode = mode
        self._classifier = classifier
        self._last_used_vision = False

    def decide(
        self,
        user_text: str,
        recent_messages: Sequence[object],
    ) -> VisionRouteDecision:
        if self._mode == "always":
            return VisionRouteDecision(VISION, VISION, "mode", "vision mode is always")
        if self._mode == "off":
            return VisionRouteDecision(TEXT, TEXT, "mode", "vision mode is off")
        rescue = _RESCUE_VISUAL.search(user_text)
        if rescue is not None:
            return VisionRouteDecision(
                VISION,
                VISION,
                "rescue_trigger",
                "visual correction or retry language",
                rescue.group(0),
            )
        trigger = _EXPLICIT_VISUAL.search(user_text)
        if trigger is not None:
            return VisionRouteDecision(
                VISION,
                VISION,
                "explicit_trigger",
                "explicit visual language",
                trigger.group(0),
            )
        assert self._classifier is not None
        try:
            requested, reason = self._classifier(
                user_text, recent_messages, self._last_used_vision
            )
            if requested not in {VISION, TEXT, UNCERTAIN}:
                raise ValueError(f"invalid classifier route: {requested!r}")
            effective = VISION if requested == UNCERTAIN else requested
            return VisionRouteDecision(
                requested,
                effective,
                "semantic_classifier",
                reason,
            )
        except Exception as exc:
            return VisionRouteDecision(
                UNCERTAIN,
                VISION,
                "classifier_fallback",
                "classifier failed; using a fresh frame",
                classifier_error=str(exc),
            )

    def observe(self, decision: VisionRouteDecision) -> None:
        self._last_used_vision = decision.attach_image

    @property
    def last_used_vision(self) -> bool:
        return self._last_used_vision

    def reset(self) -> None:
        self._last_used_vision = False


class RoutedVisionInput:
    """Compose routing and capture without coupling either one to the Agent loop."""

    def __init__(
        self,
        router: VisionRouter,
        source: ImageSource,
        *,
        max_frame_age_seconds: float = 2.0,
        recorder: VisionRouteRecorder | None = None,
    ) -> None:
        if max_frame_age_seconds <= 0:
            raise ValueError("max_frame_age_seconds must be positive")
        self._router = router
        self._source = source
        self._max_frame_age_seconds = max_frame_age_seconds
        self._recorder = recorder

    def image_for(
        self,
        user_text: str,
        recent_messages: Sequence[object],
    ) -> ImageFrame | None:
        decision = self._router.decide(user_text, recent_messages)
        frame = (
            self._source.snapshot(self._max_frame_age_seconds)
            if decision.attach_image
            else None
        )
        self._router.observe(decision)
        if self._recorder is not None:
            self._recorder.append(decision)
        return frame


class VisionRouteRecorder:
    """Optional private JSONL diagnostics; disabled unless explicitly supplied."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def append(self, decision: VisionRouteDecision) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        record = {"schema_version": 1, **asdict(decision)}
        fd = os.open(
            self._path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        try:
            os.fchmod(fd, 0o600)
            os.write(fd, (json.dumps(record, ensure_ascii=False) + "\n").encode())
        finally:
            os.close(fd)
