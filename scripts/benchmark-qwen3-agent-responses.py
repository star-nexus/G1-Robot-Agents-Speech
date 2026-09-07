#!/usr/bin/env python3
"""Benchmark a streaming llama.cpp Agent through the Responses API.

The benchmark is intentionally outside STAR Runtime.  It measures only the
HTTP/SSE model path, so ASR, Agent-to-TTS sentence buffering, TTS, playback and
AEC cannot affect the reported timings.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any
from urllib import error, request


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _json_request(
    url: str,
    *,
    payload: dict[str, Any] | None,
    api_key: str | None,
    method: str = "POST",
) -> request.Request:
    encoded = None
    if payload is not None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return request.Request(
        url,
        data=encoded,
        headers=_headers(api_key),
        method=method,
    )


def _read_json_response(response: Any) -> dict[str, Any]:
    raw = response.read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    decoded = json.loads(raw)
    if not isinstance(decoded, dict):
        raise RuntimeError("server returned a non-object JSON response")
    return decoded


def discover_model(
    responses_url: str,
    *,
    api_key: str | None,
    timeout: float,
    opener: Callable[..., Any] = request.urlopen,
) -> str:
    suffix = "/v1/responses"
    normalized_url = responses_url.rstrip("/")
    models_url = (
        normalized_url[: -len(suffix)] + "/v1/models"
        if normalized_url.endswith(suffix)
        else normalized_url + "/v1/models"
    )
    http_request = _json_request(
        models_url,
        payload=None,
        api_key=api_key,
        method="GET",
    )
    with opener(http_request, timeout=timeout) as response:
        body = _read_json_response(response)
    models = body.get("data") or []
    if not models or not isinstance(models[0], dict) or not models[0].get("id"):
        raise RuntimeError(f"no model was reported by {models_url}")
    return str(models[0]["id"])


def count_input_tokens(
    responses_url: str,
    *,
    payload: dict[str, Any],
    api_key: str | None,
    timeout: float,
    opener: Callable[..., Any] = request.urlopen,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, Any]:
    count_url = responses_url.rstrip("/") + "/input_tokens"
    count_payload = {
        key: payload[key]
        for key in ("model", "instructions", "input", "chat_template_kwargs")
        if key in payload
    }
    started_ns = clock_ns()
    http_request = _json_request(
        count_url,
        payload=count_payload,
        api_key=api_key,
    )
    with opener(http_request, timeout=timeout) as response:
        body = _read_json_response(response)
    return {
        "input_tokens": int(body["input_tokens"]),
        "request_ms": round((clock_ns() - started_ns) / 1_000_000.0, 3),
    }


def iter_sse_events(response: Any) -> Iterator[tuple[str, dict[str, Any]]]:
    """Parse SSE without assuming one JSON event per TCP read."""

    event_name = "message"
    data_lines: list[str] = []

    def dispatch() -> tuple[str, dict[str, Any]] | None:
        nonlocal event_name, data_lines
        if not data_lines:
            event_name = "message"
            return None
        encoded = "\n".join(data_lines)
        selected_name = event_name
        event_name = "message"
        data_lines = []
        if encoded == "[DONE]":
            return None
        decoded = json.loads(encoded)
        if not isinstance(decoded, dict):
            raise RuntimeError("SSE data must decode to a JSON object")
        return selected_name, decoded

    for raw_line in response:
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        line = line.rstrip("\r\n")
        if not line:
            item = dispatch()
            if item is not None:
                yield item
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_name = value
        elif field == "data":
            data_lines.append(value)
    item = dispatch()
    if item is not None:
        yield item


def _float(timings: dict[str, Any], key: str) -> float | None:
    value = timings.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _int(data: dict[str, Any], key: str) -> int | None:
    value = data.get(key)
    return int(value) if isinstance(value, (int, float)) else None


def _token_bins(
    samples: list[dict[str, Any]],
    *,
    first_text_delta_ms: float,
) -> list[dict[str, int]]:
    if not samples:
        return []
    tokens: defaultdict[int, int] = defaultdict(int)
    previous = 0
    for sample in samples:
        cumulative = int(sample["cumulative_tokens"])
        increment = max(0, cumulative - previous)
        previous = max(previous, cumulative)
        second = max(0, int((float(sample["at_ms"]) - first_text_delta_ms) // 1000))
        tokens[second] += increment
    last_second = max(tokens, default=-1)
    cumulative = 0
    result: list[dict[str, int]] = []
    for second in range(last_second + 1):
        new_tokens = tokens[second]
        cumulative += new_tokens
        result.append(
            {
                "second_from_first_text_delta": second,
                "new_tokens": new_tokens,
                "cumulative_tokens": cumulative,
            }
        )
    return result


def run_streaming_request(
    responses_url: str,
    *,
    payload: dict[str, Any],
    api_key: str | None,
    timeout: float,
    opener: Callable[..., Any] = request.urlopen,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, Any]:
    """Run one request and retain both client and server timing authorities."""

    started_ns = clock_ns()
    http_request = _json_request(
        responses_url,
        payload=payload,
        api_key=api_key,
    )
    event_counts: Counter[str] = Counter()
    first_event_ms: float | None = None
    first_text_delta_ms: float | None = None
    completed_ms: float | None = None
    first_delta_token_index: int | None = None
    output_parts: list[str] = []
    token_samples: list[dict[str, Any]] = []
    completed_event: dict[str, Any] | None = None

    try:
        response = opener(http_request, timeout=timeout)
        with response:
            for sse_name, event in iter_sse_events(response):
                observed_ms = (clock_ns() - started_ns) / 1_000_000.0
                event_type = str(event.get("type") or sse_name)
                event_counts[event_type] += 1
                if first_event_ms is None:
                    first_event_ms = observed_ms
                if event_type in {"error", "response.failed"}:
                    raise RuntimeError(f"Responses stream failed: {event}")
                if event_type == "response.output_text.delta":
                    delta = event.get("delta")
                    if isinstance(delta, str) and delta:
                        output_parts.append(delta)
                        if first_text_delta_ms is None:
                            first_text_delta_ms = observed_ms
                    timings = event.get("timings") or {}
                    predicted_n = _int(timings, "predicted_n")
                    if predicted_n is not None:
                        if first_delta_token_index is None:
                            first_delta_token_index = predicted_n
                        token_samples.append(
                            {
                                "at_ms": round(observed_ms, 3),
                                "cumulative_tokens": predicted_n,
                                "server_predicted_ms": _float(timings, "predicted_ms"),
                                "server_predicted_tokens_per_second": _float(
                                    timings,
                                    "predicted_per_second",
                                ),
                            }
                        )
                elif event_type == "response.completed":
                    completed_event = event
                    completed_ms = observed_ms
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Responses API returned HTTP {exc.code}: {detail[:1000]}"
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(f"cannot reach Responses API at {responses_url}: {exc}") from exc

    if completed_event is None or completed_ms is None:
        raise RuntimeError("Responses stream ended without response.completed")
    if first_text_delta_ms is None:
        raise RuntimeError("Responses stream completed without a text delta")

    response_body = completed_event.get("response") or {}
    usage = response_body.get("usage") or {}
    input_details = usage.get("input_tokens_details") or {}
    timings = completed_event.get("timings") or {}
    output_tokens = int(usage.get("output_tokens") or 0)
    final_predicted_n = _int(timings, "predicted_n")
    if final_predicted_n is None:
        final_predicted_n = output_tokens
    if not token_samples or token_samples[-1]["cumulative_tokens"] != final_predicted_n:
        token_samples.append(
            {
                "at_ms": round(completed_ms, 3),
                "cumulative_tokens": final_predicted_n,
                "server_predicted_ms": _float(timings, "predicted_ms"),
                "server_predicted_tokens_per_second": _float(
                    timings,
                    "predicted_per_second",
                ),
            }
        )

    first_token_index = first_delta_token_index or min(final_predicted_n, 1)
    post_first_tokens = max(0, final_predicted_n - first_token_index)
    post_first_seconds = max(0.0, (completed_ms - first_text_delta_ms) / 1000.0)
    prompt_ms = _float(timings, "prompt_ms")
    predicted_ms = _float(timings, "predicted_ms")
    provider_work_ms = (
        prompt_ms + predicted_ms
        if prompt_ms is not None and predicted_ms is not None
        else None
    )
    return {
        "response_id": response_body.get("id"),
        "model": response_body.get("model"),
        "status": response_body.get("status"),
        "input_tokens": int(usage.get("input_tokens") or 0),
        "cached_input_tokens": int(input_details.get("cached_tokens") or 0),
        "output_tokens": output_tokens,
        "total_tokens": int(usage.get("total_tokens") or 0),
        "output_characters": len("".join(output_parts)),
        "output_text": "".join(output_parts),
        "client": {
            "first_sse_event_ms": round(first_event_ms or 0.0, 3),
            "time_to_first_text_delta_ms": round(first_text_delta_ms, 3),
            "response_completed_ms": round(completed_ms, 3),
            "post_first_token_tokens_per_second": (
                round(post_first_tokens / post_first_seconds, 3)
                if post_first_seconds > 0.0
                else None
            ),
            "unattributed_overhead_ms": (
                round(max(0.0, completed_ms - provider_work_ms), 3)
                if provider_work_ms is not None
                else None
            ),
        },
        "server": {
            "cache_tokens": _int(timings, "cache_n"),
            "prompt_tokens_evaluated": _int(timings, "prompt_n"),
            "prompt_eval_ms": prompt_ms,
            "prompt_tokens_per_second": _float(timings, "prompt_per_second"),
            "decoded_tokens": final_predicted_n,
            "decode_ms": predicted_ms,
            "decode_tokens_per_second": _float(timings, "predicted_per_second"),
        },
        "token_arrival_by_second": _token_bins(
            token_samples,
            first_text_delta_ms=first_text_delta_ms,
        ),
        "token_progress": token_samples,
        "sse_event_counts": dict(sorted(event_counts.items())),
    }


def _load_history(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("history file must contain a JSON array")
    result: list[dict[str, str]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"history item {index} must be an object")
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            raise ValueError(
                f"history item {index} requires user/assistant role and string content"
            )
        result.append({"role": role, "content": content})
    return result


def _load_role_context(
    path: Path | None,
    *,
    prompt: str,
) -> tuple[str | None, str | None, dict[str, Any]]:
    if path is None:
        return None, None, {}
    from star_runtime.agent.knowledge import KeywordLoreProvider
    from star_runtime.agent.role import load_role_package

    package = load_role_package(path)
    instructions = package.prompt
    relevant = ""
    if package.knowledge.provider == "keyword":
        assert package.knowledge.cards_path is not None
        provider = KeywordLoreProvider.from_files(
            core_path=package.knowledge.core_path,
            cards_path=package.knowledge.cards_path,
            max_cards=package.knowledge.max_cards,
        )
        core = provider.core_context().strip()
        if core:
            instructions = (
                f"{instructions.rstrip()}\n\n# Canonical role knowledge\n\n{core}"
            )
        relevant = provider.relevant_context(prompt).strip()
    defaults = {
        "role_id": package.role_id,
        "max_output_tokens": package.model.max_tokens,
        "temperature": package.model.temperature,
        "enable_thinking": package.model.thinking,
    }
    return instructions, relevant or None, defaults


def build_payload(
    args: argparse.Namespace,
    *,
    model: str,
    prompt: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    role_instructions, relevant, role_defaults = _load_role_context(
        args.role_package,
        prompt=prompt,
    )
    instructions = args.instructions
    if args.instructions_file is not None:
        instructions = args.instructions_file.read_text(encoding="utf-8").strip()
    if instructions is None:
        instructions = role_instructions

    history = _load_history(args.history_file)
    input_items: list[dict[str, str]] = list(history)
    if relevant:
        input_items.append(
            {
                "role": "system",
                "content": (
                    "Canonical facts relevant to the current question follow. "
                    "Use them as authoritative and never mention retrieval:\n" + relevant
                ),
            }
        )
    input_items.append({"role": "user", "content": prompt})
    input_value: str | list[dict[str, str]] = (
        prompt if len(input_items) == 1 else input_items
    )

    max_output_tokens = (
        args.max_output_tokens
        if args.max_output_tokens is not None
        else int(role_defaults.get("max_output_tokens", 64))
    )
    temperature = (
        args.temperature
        if args.temperature is not None
        else float(role_defaults.get("temperature", 0.2))
    )
    enable_thinking = (
        args.enable_thinking
        if args.enable_thinking is not None
        else bool(role_defaults.get("enable_thinking", False))
    )
    payload: dict[str, Any] = {
        "model": model,
        "input": input_value,
        "stream": True,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        # llama.cpp extensions.  Final usage remains standard Responses API.
        "timings_per_token": True,
        "return_progress": True,
    }
    if instructions:
        payload["instructions"] = instructions
    if args.seed is not None:
        payload["seed"] = args.seed
    metadata = {
        "prompt": prompt,
        "role_package": str(args.role_package) if args.role_package else None,
        "role_id": role_defaults.get("role_id"),
        "history_turns": len(history) // 2,
        "instructions_characters": len(instructions or ""),
        "relevant_knowledge_characters": len(relevant or ""),
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "enable_thinking": enable_thinking,
        "seed": args.seed,
    }
    return payload, metadata


def _metric_summary(runs: list[dict[str, Any]], path: tuple[str, ...]) -> dict[str, float]:
    values: list[float] = []
    for run in runs:
        value: Any = run
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, (int, float)):
            values.append(float(value))
    if not values:
        return {}
    return {
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "p95": round(percentile(values, 0.95), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "time_to_first_text_delta_ms": _metric_summary(
            runs,
            ("client", "time_to_first_text_delta_ms"),
        ),
        "response_completed_ms": _metric_summary(
            runs,
            ("client", "response_completed_ms"),
        ),
        "prompt_eval_ms": _metric_summary(runs, ("server", "prompt_eval_ms")),
        "decode_tokens_per_second": _metric_summary(
            runs,
            ("server", "decode_tokens_per_second"),
        ),
        "input_tokens": _metric_summary(runs, ("input_tokens",)),
        "output_tokens": _metric_summary(runs, ("output_tokens",)),
    }


def _show_run(label: str, result: dict[str, Any]) -> None:
    client = result["client"]
    server = result["server"]
    print(
        f"{label}: input={result['input_tokens']} cached={result['cached_input_tokens']} "
        f"output={result['output_tokens']} "
        f"TTFT={client['time_to_first_text_delta_ms']:.1f}ms "
        f"total={client['response_completed_ms']:.1f}ms "
        f"prompt={server['prompt_eval_ms'] or 0.0:.1f}ms "
        f"decode={server['decode_tokens_per_second'] or 0.0:.2f} tok/s",
        file=sys.stderr,
        flush=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark llama.cpp Agent latency via POST /v1/responses.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1/responses")
    parser.add_argument("--model", help="auto-discovered from /v1/models when omitted")
    parser.add_argument("--api-key")
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help=(
            "prompt to benchmark; repeat for realistic consecutive new questions "
            "instead of an exact-prompt KV-cache test"
        ),
    )
    parser.add_argument("--role-package", type=Path)
    parser.add_argument("--instructions")
    parser.add_argument("--instructions-file", type=Path)
    parser.add_argument(
        "--history-file",
        type=Path,
        help="JSON array of prior user/assistant messages",
    )
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path, help="write the complete JSON report")
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="also print the complete JSON report to stdout",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.runs < 1 or args.warmup < 0:
        raise SystemExit("--runs must be >= 1 and --warmup must be >= 0")
    if args.max_output_tokens is not None and args.max_output_tokens < 1:
        raise SystemExit("--max-output-tokens must be >= 1")
    if args.instructions is not None and args.instructions_file is not None:
        raise SystemExit("use only one of --instructions and --instructions-file")

    try:
        model = args.model or discover_model(
            args.url,
            api_key=args.api_key,
            timeout=args.timeout,
        )
        prompts = args.prompt or ["你的伙伴都有谁？"]
        cases: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        for prompt in prompts:
            payload, request_metadata = build_payload(
                args,
                model=model,
                prompt=prompt,
            )
            preflight = count_input_tokens(
                args.url,
                payload=payload,
                api_key=args.api_key,
                timeout=args.timeout,
            )
            cases.append((payload, request_metadata, preflight))
        print(
            f"Responses API: {args.url}\n"
            f"model: {model}",
            file=sys.stderr,
        )
        for index, (_, metadata, preflight) in enumerate(cases, start=1):
            print(
                f"case {index}: input={preflight['input_tokens']} prompt={metadata['prompt']!r}",
                file=sys.stderr,
            )
        warmups: list[dict[str, Any]] = []
        for index in range(args.warmup):
            payload, metadata, _ = cases[index % len(cases)]
            result = run_streaming_request(
                args.url,
                payload=payload,
                api_key=args.api_key,
                timeout=args.timeout,
            )
            result["request"] = metadata
            warmups.append(result)
            _show_run(f"warmup {index + 1}/{args.warmup}", result)
        runs: list[dict[str, Any]] = []
        for index in range(args.runs):
            payload, metadata, _ = cases[index % len(cases)]
            result = run_streaming_request(
                args.url,
                payload=payload,
                api_key=args.api_key,
                timeout=args.timeout,
            )
            result["request"] = metadata
            runs.append(result)
            _show_run(f"run {index + 1}/{args.runs}", result)
    except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr)
        return 1

    summary = summarize(runs)
    report = {
        "schema": "star.agent.responses_benchmark.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "responses_url": args.url,
        "model_requested": model,
        "request_cases": [metadata for _, metadata, _ in cases],
        "input_token_preflight": [
            {"prompt": metadata["prompt"], **preflight}
            for _, metadata, preflight in cases
        ],
        "warmup_runs": warmups,
        "runs": runs,
        "summary": summary,
        "metric_semantics": {
            "time_to_first_text_delta_ms": (
                "client monotonic time from HTTP request start to the first non-empty "
                "response.output_text.delta"
            ),
            "response_completed_ms": (
                "client monotonic time from HTTP request start to response.completed"
            ),
            "input_output_tokens": "standard response.completed usage counters",
            "cached_input_tokens": (
                "standard usage cached-token count; TTFT comparisons are only "
                "like-for-like when this cache condition is comparable"
            ),
            "prompt_decode_timing": (
                "llama.cpp timings from response.completed; decode rate includes all "
                "predicted tokens reported by the provider"
            ),
            "token_arrival_by_second": (
                "exact cumulative llama.cpp predicted_n deltas bucketed by client "
                "arrival time from the first text delta"
            ),
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"report: {args.output.resolve()}", file=sys.stderr)
    if args.print_json:
        print(rendered)
    else:
        ttft = summary["time_to_first_text_delta_ms"]
        total = summary["response_completed_ms"]
        decode = summary["decode_tokens_per_second"]
        print(
            "summary: "
            f"TTFT median={ttft['median']:.1f}ms p95={ttft['p95']:.1f}ms; "
            f"total median={total['median']:.1f}ms; "
            f"decode median={decode['median']:.2f} tok/s",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
