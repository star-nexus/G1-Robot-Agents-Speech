"""llama.cpp adapter for semantic visual-turn routing."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from urllib import error, request

from .routing import TEXT, UNCERTAIN, VISION


_SYSTEM_PROMPT = (
    "你只做分类，不回答用户。判断机器人回答 CURRENT 是否必须观察摄像头的实时画面。\n"
    "V=必须看当前画面，例如物体、人物、位置、动作、颜色或当前环境。\n"
    "T=仅凭文字、常识或对话即可回答，例如身份、知识、闲聊或计算。\n"
    "U=结合上下文仍无法可靠判断。不要因上一轮用了视觉就自动选 V。\n"
    "只输出一个大写字母 V、T 或 U。"
)


@dataclass(frozen=True, slots=True)
class LlamaVisionClassifierSettings:
    url: str = "http://127.0.0.1:8080/v1/chat/completions"
    model: str = "qwen3-vl-4b-q4"
    timeout_seconds: float = 30.0

    def validate(self) -> None:
        if not self.url.startswith(("http://", "https://")):
            raise ValueError("classifier URL must use http:// or https://")
        if self.timeout_seconds <= 0:
            raise ValueError("classifier timeout must be positive")


class LlamaVisionClassifier:
    """Constrained V/T/U request that can run locally or on another machine."""

    def __init__(
        self,
        settings: LlamaVisionClassifierSettings,
        *,
        opener: Any = request.urlopen,
    ) -> None:
        settings.validate()
        self._settings = settings
        self._opener = opener

    def __call__(
        self,
        user_text: str,
        recent_messages: Sequence[object],
        last_used_vision: bool,
    ) -> tuple[str, str]:
        history = []
        for raw_message in recent_messages:
            if not isinstance(raw_message, dict):
                continue
            role = raw_message.get("role")
            content = raw_message.get("content")
            if role in {"user", "assistant"} and isinstance(content, str):
                history.append({"role": role, "content": content})
        payload = {
            "model": self._settings.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "last_used_vision": last_used_vision,
                            "recent_messages": history,
                            "CURRENT": user_text,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "stream": False,
            "max_tokens": 2,
            "temperature": 0,
            "grammar": 'root ::= "V" | "T" | "U"',
        }
        http_request = request.Request(
            self._settings.url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            response = self._opener(
                http_request,
                timeout=self._settings.timeout_seconds,
            )
            with response:
                result = json.load(response)
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"llama.cpp classifier returned HTTP {exc.code}: {detail[:500]}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"cannot reach llama.cpp classifier at {self._settings.url}: {exc}"
            ) from exc
        try:
            label = str(result["choices"][0]["message"]["content"]).strip()
            route = {"V": VISION, "T": TEXT, "U": UNCERTAIN}[label]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"invalid vision classifier response: {result!r}") from exc
        return route, f"llama.cpp visual route label {label}"
