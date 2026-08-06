#!/usr/bin/env python3
"""Subscribe to g1-speech events and answer with a local Qwen model."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
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


@dataclass(frozen=True)
class CameraSnapshot:
    jpeg: bytes
    captured_monotonic: float

    @property
    def data_url(self) -> str:
        encoded = base64.b64encode(self.jpeg).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"


class MjpegStreamParser:
    """Extract complete JPEG images from a byte stream."""

    _SOI = b"\xff\xd8"
    _EOI = b"\xff\xd9"

    def __init__(self, max_buffer_bytes: int = 16 * 1024 * 1024) -> None:
        self._buffer = bytearray()
        self._max_buffer_bytes = max_buffer_bytes

    def feed(self, chunk: bytes) -> list[bytes]:
        self._buffer.extend(chunk)
        frames: list[bytes] = []
        while True:
            start = self._buffer.find(self._SOI)
            if start < 0:
                if len(self._buffer) > 1:
                    self._buffer[:] = self._buffer[-1:]
                break
            end = self._buffer.find(self._EOI, start + len(self._SOI))
            if end < 0:
                if start:
                    del self._buffer[:start]
                break
            end += len(self._EOI)
            frames.append(bytes(self._buffer[start:end]))
            del self._buffer[:end]
        if len(self._buffer) > self._max_buffer_bytes:
            logger.warning("Discarding oversized incomplete MJPEG frame")
            self._buffer.clear()
        return frames


class LatestFrameCamera:
    """Continuously retain the newest MJPEG frame from a V4L2 camera."""

    def __init__(
        self,
        *,
        device: str,
        width: int,
        height: int,
        fps: int,
        ffmpeg: str,
        reconnect_seconds: float = 1.0,
    ) -> None:
        self._device = device
        self._width = width
        self._height = height
        self._fps = fps
        self._ffmpeg = ffmpeg
        self._reconnect_seconds = reconnect_seconds
        self._stopped = threading.Event()
        self._frame_ready = threading.Event()
        self._lock = threading.Lock()
        self._snapshot: CameraSnapshot | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._thread: threading.Thread | None = None

    def start(self, timeout: float) -> None:
        if shutil.which(self._ffmpeg) is None:
            raise RuntimeError(f"FFmpeg executable not found: {self._ffmpeg!r}")
        self._thread = threading.Thread(
            target=self._run,
            name="camera-latest-frame",
            daemon=True,
        )
        self._thread.start()
        if not self._frame_ready.wait(timeout):
            self.close()
            raise RuntimeError(
                f"No frame received from camera {self._device!r} within {timeout:.1f}s"
            )
        logger.info(
            "Camera started: device=%r resolution=%dx%d fps=%d",
            self._device,
            self._width,
            self._height,
            self._fps,
        )

    def snapshot(self, max_age_seconds: float) -> CameraSnapshot:
        with self._lock:
            snapshot = self._snapshot
        if snapshot is None:
            raise RuntimeError(f"Camera {self._device!r} has no frame")
        age = time.monotonic() - snapshot.captured_monotonic
        if age > max_age_seconds:
            raise RuntimeError(
                f"Latest camera frame is stale: age={age:.2f}s "
                f"limit={max_age_seconds:.2f}s"
            )
        return snapshot

    def close(self) -> None:
        self._stopped.set()
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _command(self) -> list[str]:
        return [
            self._ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "v4l2",
            "-input_format",
            "mjpeg",
            "-framerate",
            str(self._fps),
            "-video_size",
            f"{self._width}x{self._height}",
            "-i",
            self._device,
            "-c:v",
            "copy",
            "-f",
            "image2pipe",
            "pipe:1",
        ]

    def _run(self) -> None:
        while not self._stopped.is_set():
            parser = MjpegStreamParser()
            process: subprocess.Popen[bytes] | None = None
            try:
                process = subprocess.Popen(
                    self._command(),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=0,
                )
                with self._lock:
                    self._process = process
                assert process.stdout is not None
                while not self._stopped.is_set():
                    chunk = process.stdout.read(64 * 1024)
                    if not chunk:
                        break
                    for jpeg in parser.feed(chunk):
                        with self._lock:
                            self._snapshot = CameraSnapshot(jpeg, time.monotonic())
                        self._frame_ready.set()
            except OSError as exc:
                if not self._stopped.is_set():
                    logger.warning("Camera process failed: %s", exc)
            finally:
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=1.0)
                with self._lock:
                    if self._process is process:
                        self._process = None
            if not self._stopped.wait(self._reconnect_seconds):
                logger.warning("Camera stream stopped; reconnecting to %r", self._device)


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
        messages: list[dict[str, Any]],
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


class LlamaCppChatClient:
    """Streaming client for llama.cpp's OpenAI-compatible HTTP API."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout: float,
        num_predict: int,
        temperature: float,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._num_predict = num_predict
        self._temperature = temperature

    def ensure_ready(self) -> None:
        payload = self._request_json("GET", "/v1/models")
        names = {
            item.get("id", item.get("name", ""))
            for item in payload.get("data", payload.get("models", []))
            if isinstance(item, dict)
        }
        if self._model not in names:
            available = ", ".join(sorted(names)) or "none"
            raise RuntimeError(
                f"llama.cpp model {self._model!r} is not served; available: {available}"
            )

    def chat(
        self,
        messages: list[dict[str, Any]],
        on_content: Callable[[str], None],
    ) -> ChatResult:
        body = json.dumps(
            {
                "model": self._model,
                "messages": messages,
                "stream": True,
                "stream_options": {"include_usage": True},
                "max_tokens": self._num_predict,
                "temperature": self._temperature,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = Request(
            f"{self._base_url}/v1/chat/completions",
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
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if chunk.get("error"):
                        raise RuntimeError(f"llama.cpp error: {chunk['error']}")
                    choices = chunk.get("choices", [])
                    content = choices[0].get("delta", {}).get("content") if choices else None
                    if content:
                        if first_token is None:
                            first_token = time.monotonic() - started
                        parts.append(content)
                        on_content(content)
                    if chunk.get("usage") or chunk.get("timings"):
                        final = chunk
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"llama.cpp HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(
                f"Cannot reach llama.cpp at {self._base_url}: {exc.reason}"
            ) from exc

        total = time.monotonic() - started
        content = "".join(parts).strip()
        if not content:
            raise RuntimeError("llama.cpp returned an empty answer")
        usage = final.get("usage", {})
        timings = final.get("timings", {})
        speed = timings.get("predicted_per_second")
        return ChatResult(
            content=content,
            first_token_seconds=first_token,
            total_seconds=total,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            generated_tokens=int(usage.get("completion_tokens", 0)),
            tokens_per_second=float(speed) if speed is not None else None,
        )

    def _request_json(self, method: str, path: str) -> dict:
        request = Request(f"{self._base_url}{path}", method=method)
        try:
            with urlopen(request, timeout=self._timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"llama.cpp HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(
                f"Cannot reach llama.cpp at {self._base_url}: {exc.reason}"
            ) from exc


class Conversation:
    def __init__(self, system_prompt: str, history_turns: int) -> None:
        self._system = {"role": "system", "content": system_prompt}
        self._history_turns = history_turns
        self._messages: list[dict[str, Any]] = [self._system]

    def request(
        self,
        text: str,
        image_data_url: str | None = None,
    ) -> list[dict[str, Any]]:
        content: str | list[dict[str, Any]] = text
        if image_data_url is not None:
            content = [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {
                    "type": "text",
                    "text": (
                        "以上是机器人摄像头刚刚捕获的当前画面。"
                        f"请结合画面回答用户的语音：{text}"
                    ),
                },
            ]
        return [*self._messages, {"role": "user", "content": content}]

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
    def messages(self) -> list[dict[str, Any]]:
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
    parser.add_argument("--provider", choices=("llama", "ollama"), default="llama")
    parser.add_argument(
        "--api-url",
        help="Inference API root; defaults to :11434 for Ollama or :8080 for llama.cpp",
    )
    parser.add_argument("--ollama-url", dest="api_url", help=argparse.SUPPRESS)
    parser.add_argument("--model", default="qwen3-8b-q5")
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
        "--camera",
        metavar="DEVICE",
        help="Attach the latest frame from a V4L2 camera to every request",
    )
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=5)
    parser.add_argument("--camera-max-age", type=float, default=2.0)
    parser.add_argument("--camera-startup-timeout", type=float, default=8.0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument(
        "--accept-backlog",
        action="store_true",
        help="Accept speech events created before this Agent started",
    )
    parser.add_argument(
        "--prompt",
        help="Send one text prompt without subscribing; useful for checking the LLM API",
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
    if args.camera and args.provider != "llama":
        raise ValueError("--camera currently requires --provider llama")
    if min(args.camera_width, args.camera_height, args.camera_fps) < 1:
        raise ValueError("camera dimensions and FPS must be greater than zero")
    if args.camera_max_age <= 0 or args.camera_startup_timeout <= 0:
        raise ValueError("camera timing limits must be greater than zero")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    if args.provider == "ollama":
        client = OllamaChatClient(
            base_url=args.api_url or "http://127.0.0.1:11434",
            model=args.model,
            timeout=args.timeout,
            keep_alive=args.keep_alive,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
            temperature=args.temperature,
            think=args.think,
        )
    else:
        if args.think:
            raise ValueError(
                "--think for llama.cpp is configured when starting llama serve"
            )
        client = LlamaCppChatClient(
            base_url=args.api_url or "http://127.0.0.1:8080",
            model=args.model,
            timeout=args.timeout,
            num_predict=args.num_predict,
            temperature=args.temperature,
        )
    client.ensure_ready()
    conversation = Conversation(args.system, args.history_turns)
    camera: LatestFrameCamera | None = None

    def start_camera() -> LatestFrameCamera | None:
        if not args.camera:
            return None
        source = LatestFrameCamera(
            device=args.camera,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            ffmpeg=args.ffmpeg,
        )
        source.start(args.camera_startup_timeout)
        return source

    def answer(text: str) -> None:
        image_data_url: str | None = None
        if camera is not None:
            snapshot = camera.snapshot(args.camera_max_age)
            age_ms = (time.monotonic() - snapshot.captured_monotonic) * 1000
            logger.info(
                "Attaching camera frame: bytes=%d age=%.0fms",
                len(snapshot.jpeg),
                age_ms,
            )
            image_data_url = snapshot.data_url
        print(f"\nYou > {text}")
        print("Qwen3 > ", end="", flush=True)
        result = client.chat(
            conversation.request(text, image_data_url),
            lambda token: print(token, end="", flush=True),
        )
        print(f"\n[{format_stats(result)}]", flush=True)
        conversation.commit(text, result.content)

    if args.prompt:
        camera = start_camera()
        try:
            answer(args.prompt)
            return 0
        finally:
            if camera is not None:
                camera.close()

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
    camera = start_camera()
    thread = threading.Thread(target=worker, name="qwen-chat-worker", daemon=True)
    try:
        thread.start()
        subscriber.start()
        print(
            f"Ready: {transport.upper()} {topic} -> {args.provider} {args.model}. "
            f"Camera: {args.camera or 'off'}. "
            f"Say {args.reset_phrase!r} to reset; Ctrl-C to exit.",
            flush=True,
        )
        while not stopped.wait(0.5):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        stopped.set()
        subscriber.close()
        if thread.is_alive():
            thread.join(timeout=2.0)
        if camera is not None:
            camera.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
