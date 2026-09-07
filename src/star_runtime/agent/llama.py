"""OpenAI-compatible llama.cpp provider for the local STAR Runtime."""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from typing import Any
from urllib import error, request

from .contracts import ChatMessage


DEFAULT_REASONING_BUDGET_MESSAGE = (
    "\n现在停止分析，严格按照系统指令只输出最终回答。"
)


@dataclass(frozen=True)
class LlamaChatSettings:
    url: str = "http://127.0.0.1:8080/v1/chat/completions"
    model: str = "local-qwen3-4b"
    max_tokens: int = 64
    temperature: float = 0.2
    enable_thinking: bool = False
    reasoning_budget_tokens: int = -1
    reasoning_budget_message: str = DEFAULT_REASONING_BUDGET_MESSAGE
    request_timeout_seconds: float = 120.0

    def validate(self) -> None:
        if not self.url.startswith(("http://", "https://")):
            raise ValueError("Agent URL must use http:// or https://")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be greater than zero")
        if self.reasoning_budget_tokens < -1:
            raise ValueError("reasoning_budget_tokens must be -1 or greater")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be greater than zero")


class LlamaChatClient:
    """Small streaming model provider with no third-party dependency."""

    def __init__(
        self,
        settings: LlamaChatSettings,
        *,
        opener: Any = request.urlopen,
    ) -> None:
        settings.validate()
        self._settings = settings
        self._opener = opener
        self._active_lock = threading.Lock()
        self._active_response: Any | None = None

    def cancel(self) -> bool:
        """Close the active HTTP stream when urllib/provider supports it."""

        with self._active_lock:
            response = self._active_response
        if response is None:
            return False
        response.close()
        return True

    def stream(self, messages: list[ChatMessage]):
        payload = {
            "model": self._settings.model,
            "messages": messages,
            "stream": True,
            "temperature": self._settings.temperature,
            "max_tokens": self._settings.max_tokens,
            "chat_template_kwargs": {
                "enable_thinking": self._settings.enable_thinking,
            },
        }
        if self._settings.enable_thinking:
            payload["reasoning_budget_tokens"] = (
                self._settings.reasoning_budget_tokens
            )
            payload["reasoning_budget_message"] = (
                self._settings.reasoning_budget_message
            )
            payload["reasoning_format"] = "deepseek"
        http_request = request.Request(
            self._settings.url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            response = self._opener(
                http_request,
                timeout=self._settings.request_timeout_seconds,
            )
            with self._active_lock:
                self._active_response = response
            buffered_content: list[str] = []
            finish_reason: str | None = None
            with response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line or line.startswith(":") or not line.startswith("data:"):
                        continue
                    encoded = line[5:].strip()
                    if encoded == "[DONE]":
                        break
                    event = json.loads(encoded)
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    if choice.get("finish_reason") is not None:
                        finish_reason = str(choice["finish_reason"])
                    content = (choice.get("delta") or {}).get("content")
                    if isinstance(content, str) and content:
                        if self._settings.enable_thinking:
                            # Buffer reasoned output until completion so an
                            # unfinished thought can never reach DDS/TTS.
                            buffered_content.append(content)
                        else:
                            yield content
            if self._settings.enable_thinking:
                if finish_reason == "length":
                    raise RuntimeError(
                        "llama.cpp exhausted max_tokens before completing the "
                        "budgeted-thinking answer"
                    )
                spoken = _strip_reasoning_markup("".join(buffered_content))
                if not spoken:
                    raise RuntimeError(
                        "llama.cpp returned no final answer after reasoning"
                    )
                yield spoken
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"llama.cpp returned HTTP {exc.code}: {detail[:500]}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"cannot reach llama.cpp at {self._settings.url}: {exc}"
            ) from exc
        finally:
            with self._active_lock:
                self._active_response = None


def _strip_reasoning_markup(content: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    if "<think>" in cleaned:
        raise RuntimeError("llama.cpp returned an unterminated reasoning block")
    return cleaned.replace("</think>", "").strip()
