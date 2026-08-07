"""Low-latency, context-aware routing for optional camera input."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Generic, Protocol, Sequence, TypeVar


VISION = "VISION"
TEXT = "TEXT"
UNCERTAIN = "UNCERTAIN"
VALID_ROUTES = {VISION, TEXT, UNCERTAIN}

# Keep this list deliberately narrow. Ambiguous language belongs in the semantic
# classifier, while these expressions are reliable enough to bypass it.
EXPLICIT_VISUAL_TRIGGER = re.compile(
    r"(?:摄像头|镜头|画面|眼前|面前|你看见|你看到|看一下|看一眼|"
    r"看看(?:这个|那个|这里|那里|前面|后面|左边|右边|画面))"
)
RESCUE_VISUAL_TRIGGER = re.compile(
    r"(?:再看|重新看|看清楚|仔细看|你看错|你没看|画面不对|不是这个)"
)


@dataclass(frozen=True)
class VisionRouteDecision:
    requested_route: str
    effective_route: str
    source: str
    reason: str
    latency_ms: float
    explicit_trigger: str | None = None
    classifier_error: str | None = None

    @property
    def attach_image(self) -> bool:
        return self.effective_route == VISION


Classifier = Callable[
    [str, Sequence[dict[str, Any]], bool], tuple[str, str]
]

ResponseT = TypeVar("ResponseT")


@dataclass(frozen=True)
class PreparedVisionTurn(Generic[ResponseT]):
    """A route decision, optionally with an already generated text answer."""

    decision: VisionRouteDecision
    direct_response: ResponseT | None = None


class VisionStrategy(Protocol[ResponseT]):
    """Interchangeable policy used before the optional camera snapshot."""

    name: str

    @property
    def last_used_vision(self) -> bool: ...

    def prepare(
        self,
        user_text: str,
        recent_messages: Sequence[dict[str, Any]],
        text_request: list[dict[str, Any]],
        on_content: Callable[[str], None],
    ) -> PreparedVisionTurn[ResponseT]: ...

    def observe(self, decision: VisionRouteDecision) -> None: ...

    def reset(self) -> None: ...


@dataclass(frozen=True)
class VisualTrigger:
    source: str
    text: str


def match_visual_trigger(user_text: str) -> VisualTrigger | None:
    """Return only high-confidence visual and correction triggers."""

    rescue = RESCUE_VISUAL_TRIGGER.search(user_text)
    if rescue is not None:
        return VisualTrigger("rescue_trigger", rescue.group(0))
    explicit = EXPLICIT_VISUAL_TRIGGER.search(user_text)
    if explicit is not None:
        return VisualTrigger("explicit_trigger", explicit.group(0))
    return None


class VisionRouter:
    """Route camera input with strong triggers and a semantic fallback."""

    def __init__(self, *, mode: str, classifier: Classifier | None) -> None:
        if mode not in {"auto", "always", "off"}:
            raise ValueError(f"unsupported vision mode: {mode!r}")
        if mode == "auto" and classifier is None:
            raise ValueError("auto vision mode requires a classifier")
        self.mode = mode
        self._classifier = classifier
        self._last_used_vision = False

    @property
    def last_used_vision(self) -> bool:
        return self._last_used_vision

    def reset(self) -> None:
        self._last_used_vision = False

    def decide(
        self,
        user_text: str,
        recent_messages: Sequence[dict[str, Any]],
    ) -> VisionRouteDecision:
        started = time.monotonic()
        if self.mode == "always":
            return self._decision(
                requested=VISION,
                effective=VISION,
                source="mode",
                reason="vision mode is always",
                started=started,
            )
        if self.mode == "off":
            return self._decision(
                requested=TEXT,
                effective=TEXT,
                source="mode",
                reason="vision mode is off",
                started=started,
            )

        trigger = match_visual_trigger(user_text)
        if trigger is not None:
            return self._decision(
                requested=VISION,
                effective=VISION,
                source=trigger.source,
                reason=f"matched {trigger.text!r}",
                started=started,
                explicit_trigger=trigger.text,
            )

        assert self._classifier is not None
        try:
            requested, reason = self._classifier(
                user_text,
                recent_messages,
                self._last_used_vision,
            )
            requested = requested.strip().upper()
            if requested not in VALID_ROUTES:
                raise ValueError(f"invalid classifier route: {requested!r}")
        except Exception as exc:  # noqa: BLE001
            return self._decision(
                requested=UNCERTAIN,
                effective=VISION,
                source="classifier_fallback",
                reason="classifier failed; fail-safe camera attachment",
                started=started,
                classifier_error=f"{type(exc).__name__}: {exc}",
            )

        # A false negative is much more damaging than an unnecessary frame, so
        # uncertain semantic decisions intentionally fall back to vision.
        effective = TEXT if requested == TEXT else VISION
        return self._decision(
            requested=requested,
            effective=effective,
            source="semantic_classifier",
            reason=reason,
            started=started,
        )

    def observe(self, decision: VisionRouteDecision) -> None:
        self._last_used_vision = decision.attach_image

    @staticmethod
    def _decision(
        *,
        requested: str,
        effective: str,
        source: str,
        reason: str,
        started: float,
        explicit_trigger: str | None = None,
        classifier_error: str | None = None,
    ) -> VisionRouteDecision:
        return VisionRouteDecision(
            requested_route=requested,
            effective_route=effective,
            source=source,
            reason=reason,
            latency_ms=(time.monotonic() - started) * 1000,
            explicit_trigger=explicit_trigger,
            classifier_error=classifier_error,
        )


class RoutedVisionStrategy(Generic[ResponseT]):
    """Adapt a route-first policy to the common turn strategy interface."""

    def __init__(self, *, name: str, router: VisionRouter) -> None:
        self.name = name
        self._router = router

    @property
    def last_used_vision(self) -> bool:
        return self._router.last_used_vision

    def prepare(
        self,
        user_text: str,
        recent_messages: Sequence[dict[str, Any]],
        text_request: list[dict[str, Any]],
        on_content: Callable[[str], None],
    ) -> PreparedVisionTurn[ResponseT]:
        del text_request, on_content
        return PreparedVisionTurn(self._router.decide(user_text, recent_messages))

    def observe(self, decision: VisionRouteDecision) -> None:
        self._router.observe(decision)

    def reset(self) -> None:
        self._router.reset()


class SeeVisionStrategy(Generic[ResponseT]):
    """Use an independent classifier session to select ``[SEE]`` or text."""

    name = "see"

    def __init__(
        self,
        classifier: Classifier,
        reset_classifier: Callable[[], None] | None = None,
    ) -> None:
        self._classifier = classifier
        self._reset_classifier = reset_classifier
        self._last_used_vision = False

    @property
    def last_used_vision(self) -> bool:
        return self._last_used_vision

    def prepare(
        self,
        user_text: str,
        recent_messages: Sequence[dict[str, Any]],
        text_request: list[dict[str, Any]],
        on_content: Callable[[str], None],
    ) -> PreparedVisionTurn[ResponseT]:
        del text_request, on_content
        started = time.monotonic()
        try:
            requested, reason = self._classifier(
                user_text,
                recent_messages,
                self._last_used_vision,
            )
            requested = requested.strip().upper()
            if requested not in {VISION, TEXT}:
                raise ValueError(f"invalid [SEE] classifier route: {requested!r}")
        except Exception as exc:  # noqa: BLE001
            return PreparedVisionTurn(
                VisionRouteDecision(
                    requested_route=UNCERTAIN,
                    effective_route=VISION,
                    source="see_classifier_fallback",
                    reason="[SEE] classifier failed; fail-safe camera attachment",
                    latency_ms=(time.monotonic() - started) * 1000,
                    classifier_error=f"{type(exc).__name__}: {exc}",
                )
            )
        return PreparedVisionTurn(
            VisionRouteDecision(
                requested_route=requested,
                effective_route=requested,
                source="see_classifier",
                reason=reason,
                latency_ms=(time.monotonic() - started) * 1000,
            )
        )

    def observe(self, decision: VisionRouteDecision) -> None:
        self._last_used_vision = decision.attach_image

    def reset(self) -> None:
        self._last_used_vision = False
        if self._reset_classifier is not None:
            self._reset_classifier()


def default_router_log_path() -> Path:
    state_home = os.environ.get("XDG_STATE_HOME")
    base = Path(state_home).expanduser() if state_home else Path.home() / ".local/state"
    return base / "g1-speech" / "vision-router.jsonl"


class VisionRouteRecorder:
    """Append privacy-sensitive router training records to a local JSONL file."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, event: dict[str, Any]) -> None:
        payload = {
            "schema_version": 1,
            "recorded_unix_ns": time.time_ns(),
            **event,
        }
        encoded = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        with self._lock:
            descriptor = os.open(
                self.path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o600,
            )
            try:
                os.write(descriptor, encoded)
            finally:
                os.close(descriptor)
            os.chmod(self.path, 0o600)


def decision_record(decision: VisionRouteDecision) -> dict[str, Any]:
    return asdict(decision)
