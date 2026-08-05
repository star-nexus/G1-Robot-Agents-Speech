#!/usr/bin/env python3
"""Subscribe to g1-speech events and answer with a local Ollama model."""

from __future__ import annotations

import argparse
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from g1_speech.config import load_config
from g1_speech.contracts import SpeechEvent
from g1_speech.dds import DdsSpeechSubscriber, initialize_dds
from g1_speech.ros2 import Ros2SpeechSubscriber


logger = logging.getLogger("qwen_voice_chat")

DEFAULT_SYSTEM_PROMPT = (
    "你是运行在机器人本地的语音助手。请优先使用自然、简洁的中文回答，通常控制在一到三句话。"
    "用户输入来自语音识别，可能含有同音字或少量错字；请结合对话上下文理解。"
    "如果关键信息不确定，请简短确认，不要编造。不要展示思考过程。"
)


@dataclass(frozen=True)
class ChatResult:
    content: str
    first_token_seconds: float | None
    total_seconds: float
    prompt_tokens: int
    generated_tokens: int
    tokens_per_second: float | None


class OllamaChatClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout: float,
        keep_alive: str,
        num_ctx: int,
        num_predict: int,
        temperature: float,
        think: bool,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._keep_alive = keep_alive
        self._options = {
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "temperature": temperature,
        }
        self._think = think

    def ensure_ready(self) -> None:
        payload = self._request_json("GET", "/api/tags")
        names = {
            item.get("name", "")
            for item in payload.get("models", [])
            if isinstance(item, dict)
        }
        aliases = {self._model, f"{self._model}:latest"}
        if names.isdisjoint(aliases):
            available = ", ".join(sorted(names)) or "none"
            raise RuntimeError(
                f"Ollama model {self._model!r} is not installed; available: {available}"
            )

    def chat(
        self,
        messages: list[dict[str, str]],
        on_content: Callable[[str], None],
    ) -> ChatResult:
        body = json.dumps(
            {
                "model": self._model,
                "messages": messages,
                "stream": True,
                "think": self._think,
                "keep_alive": self._keep_alive,
                "options": self._options,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = Request(
            f"{self._base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        first_token: float | None = None
        parts: list[str] = []
        final: dict = {}
        try:
            with urlopen(request, timeout=self._timeout) as response:
                for raw_line in response:
                    if not raw_line.strip():
                        continue
                    chunk = json.loads(raw_line)
                    if chunk.get("error"):
                        raise RuntimeError(f"Ollama error: {chunk['error']}")
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        if first_token is None:
                            first_token = time.monotonic() - started
                        parts.append(content)
                        on_content(content)
                    if chunk.get("done"):
                        final = chunk
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Cannot reach Ollama at {self._base_url}: {exc.reason}") from exc

        total = time.monotonic() - started
        content = "".join(parts).strip()
        if not content:
            raise RuntimeError("Ollama returned an empty answer")
        generated = int(final.get("eval_count", 0))
        eval_duration = int(final.get("eval_duration", 0))
        tokens_per_second = (
            generated / (eval_duration / 1_000_000_000)
            if generated > 0 and eval_duration > 0
            else None
        )
        return ChatResult(
            content=content,
            first_token_seconds=first_token,
            total_seconds=total,
            prompt_tokens=int(final.get("prompt_eval_count", 0)),
            generated_tokens=generated,
            tokens_per_second=tokens_per_second,
        )

    def _request_json(self, method: str, path: str) -> dict:
        request = Request(f"{self._base_url}{path}", method=method)
        try:
            with urlopen(request, timeout=self._timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Cannot reach Ollama at {self._base_url}: {exc.reason}") from exc


class Conversation:
    def __init__(self, system_prompt: str, history_turns: int) -> None:
        self._system = {"role": "system", "content": system_prompt}
        self._history_turns = history_turns
        self._messages: list[dict[str, str]] = [self._system]

    def request(self, text: str) -> list[dict[str, str]]:
        return [*self._messages, {"role": "user", "content": text}]

    def commit(self, user_text: str, assistant_text: str) -> None:
        self._messages.extend(
            [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]
        )
        conversation = self._messages[1:]
        self._messages = [self._system, *conversation[-2 * self._history_turns :]]

    def clear(self) -> None:
        self._messages = [self._system]

    @property
    def messages(self) -> list[dict[str, str]]:
        return list(self._messages)


def normalize_phrase(text: str) -> str:
    return text.strip().rstrip("。！？，、,.!? ").strip()


def format_stats(result: ChatResult) -> str:
    first = (
        f"{result.first_token_seconds:.2f}s"
        if result.first_token_seconds is not None
        else "n/a"
    )
    speed = (
        f"{result.tokens_per_second:.1f} tok/s"
        if result.tokens_per_second is not None
        else "n/a"
    )
    return (
        f"first={first}, total={result.total_seconds:.2f}s, "
        f"tokens={result.prompt_tokens}+{result.generated_tokens}, speed={speed}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json", type=Path)
    parser.add_argument("--transport", choices=("dds", "ros2"))
    parser.add_argument("--model", default="qwen3-8b-q5")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--history-turns", type=int, default=6)
    parser.add_argument("--queue-size", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--keep-alive", default="30m")
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--num-predict", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--reset-phrase", default="清空对话")
    parser.add_argument(
        "--accept-backlog",
        action="store_true",
        help="Accept speech events created before this Agent started",
    )
    parser.add_argument(
        "--prompt",
        help="Send one text prompt without subscribing; useful for checking Ollama",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.history_turns < 1:
        raise ValueError("--history-turns must be greater than zero")
    if args.queue_size < 1:
        raise ValueError("--queue-size must be greater than zero")
    if args.timeout <= 0 or args.num_ctx < 256 or args.num_predict < 1:
        raise ValueError("timeout/context/generation limits must be positive")
    if not 0 <= args.temperature <= 2:
        raise ValueError("--temperature must be between 0 and 2")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    client = OllamaChatClient(
        base_url=args.ollama_url,
        model=args.model,
        timeout=args.timeout,
        keep_alive=args.keep_alive,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        temperature=args.temperature,
        think=args.think,
    )
    client.ensure_ready()
    conversation = Conversation(args.system, args.history_turns)

    def answer(text: str) -> None:
        print(f"\nYou > {text}")
        print("Qwen3 > ", end="", flush=True)
        result = client.chat(
            conversation.request(text),
            lambda token: print(token, end="", flush=True),
        )
        print(f"\n[{format_stats(result)}]", flush=True)
        conversation.commit(text, result.content)

    if args.prompt:
        answer(args.prompt)
        return 0

    config = load_config(args.config)
    transport = args.transport or config.transport.backend
    pending: queue.Queue[SpeechEvent] = queue.Queue(maxsize=args.queue_size)
    stopped = threading.Event()
    started_unix_ns = time.time_ns()

    def on_speech(event: SpeechEvent) -> None:
        if not args.accept_backlog and event.created_unix_ns < started_unix_ns:
            logger.info("Ignoring speech event created before Agent startup: %s", event.event_id)
            return
        try:
            pending.put_nowait(event)
        except queue.Full:
            try:
                dropped = pending.get_nowait()
                logger.warning("Dropping queued speech: %r", dropped.text)
            except queue.Empty:
                pass
            try:
                pending.put_nowait(event)
            except queue.Full:
                logger.warning("Dropping incoming speech: %r", event.text)

    def worker() -> None:
        while not stopped.is_set():
            try:
                event = pending.get(timeout=0.2)
            except queue.Empty:
                continue
            text = event.text.strip()
            if normalize_phrase(text) == normalize_phrase(args.reset_phrase):
                conversation.clear()
                print("\n[Conversation cleared]", flush=True)
                continue
            try:
                answer(text)
            except Exception:  # noqa: BLE001
                logger.exception("Qwen3 request failed")

    if transport == "dds":
        initialize_dds(config.dds.domain_id, config.dds.network_interface)
        subscriber = DdsSpeechSubscriber(on_speech, topic=config.dds.speech_topic)
        topic = config.dds.speech_topic
    elif transport == "ros2":
        subscriber = Ros2SpeechSubscriber(on_speech, config.ros2)
        topic = config.ros2.speech_topic
    else:
        raise ValueError(f"unsupported transport: {transport!r}")
    thread = threading.Thread(target=worker, name="qwen-chat-worker", daemon=True)
    thread.start()
    subscriber.start()
    print(
        f"Ready: {transport.upper()} {topic} -> Ollama {args.model}. "
        f"Say {args.reset_phrase!r} to reset; Ctrl-C to exit.",
        flush=True,
    )
    try:
        while not stopped.wait(0.5):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        stopped.set()
        subscriber.close()
        thread.join(timeout=2.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
