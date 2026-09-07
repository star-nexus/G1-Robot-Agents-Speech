from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark-qwen3-agent-responses.py"
SPEC = importlib.util.spec_from_file_location("agent_responses_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class FakeResponse:
    def __init__(self, lines: list[bytes] | None = None, body: bytes = b"") -> None:
        self._lines = lines or []
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def __iter__(self):
        return iter(self._lines)

    def read(self) -> bytes:
        return self._body


def _event(event_type: str, body: dict) -> list[bytes]:
    payload = {"type": event_type, **body}
    return [
        f"event: {event_type}\n".encode(),
        f"data: {json.dumps(payload)}\n".encode(),
        b"\n",
    ]


def test_stream_metrics_use_usage_and_provider_token_counts_at_race_boundaries():
    lines = [
        *_event("response.created", {"response": {"id": "resp-1"}}),
        *_event(
            "response.output_text.delta",
            {
                "delta": "first",
                "timings": {"predicted_n": 1, "predicted_ms": 0.1},
            },
        ),
        *_event(
            "response.output_text.delta",
            {
                "delta": " second",
                "timings": {"predicted_n": 5, "predicted_ms": 1000.0},
            },
        ),
        *_event(
            "response.completed",
            {
                "response": {
                    "id": "resp-1",
                    "status": "completed",
                    "model": "model.gguf",
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 6,
                        "total_tokens": 26,
                        "input_tokens_details": {"cached_tokens": 8},
                    },
                },
                "timings": {
                    "cache_n": 8,
                    "prompt_n": 12,
                    "prompt_ms": 300.0,
                    "prompt_per_second": 40.0,
                    "predicted_n": 6,
                    "predicted_ms": 1300.0,
                    "predicted_per_second": 4.615,
                },
            },
        ),
    ]
    response = FakeResponse(lines)
    times = iter([0, 100_000_000, 700_000_000, 1_800_000_000, 2_000_000_000])

    result = benchmark.run_streaming_request(
        "http://localhost:8080/v1/responses",
        payload={"model": "test", "input": "hello", "stream": True},
        api_key=None,
        timeout=1.0,
        opener=lambda *_args, **_kwargs: response,
        clock_ns=lambda: next(times),
    )

    assert result["input_tokens"] == 20
    assert result["cached_input_tokens"] == 8
    assert result["output_tokens"] == 6
    assert result["client"]["time_to_first_text_delta_ms"] == 700.0
    assert result["client"]["response_completed_ms"] == 2000.0
    assert result["client"]["post_first_token_tokens_per_second"] == 3.846
    assert result["server"]["decode_tokens_per_second"] == 4.615
    assert result["token_arrival_by_second"] == [
        {
            "second_from_first_text_delta": 0,
            "new_tokens": 1,
            "cumulative_tokens": 1,
        },
        {
            "second_from_first_text_delta": 1,
            "new_tokens": 5,
            "cumulative_tokens": 6,
        },
    ]
    assert result["output_text"] == "first second"


def test_input_token_preflight_uses_responses_input_tokens_endpoint():
    captured = {}
    response = FakeResponse(body=b'{"input_tokens":37,"object":"response.input_tokens"}')

    def opener(http_request, **_kwargs):
        captured["url"] = http_request.full_url
        captured["payload"] = json.loads(http_request.data)
        return response

    times = iter([1_000_000_000, 1_012_000_000])
    result = benchmark.count_input_tokens(
        "http://localhost:8080/v1/responses",
        payload={
            "model": "test",
            "instructions": "role",
            "input": "question",
            "stream": True,
            "timings_per_token": True,
        },
        api_key=None,
        timeout=1.0,
        opener=opener,
        clock_ns=lambda: next(times),
    )

    assert result == {"input_tokens": 37, "request_ms": 12.0}
    assert captured["url"].endswith("/v1/responses/input_tokens")
    assert captured["payload"] == {
        "model": "test",
        "instructions": "role",
        "input": "question",
    }


def test_summary_keeps_ttft_and_decode_throughput_as_distinct_metrics():
    runs = [
        {
            "input_tokens": 100,
            "output_tokens": 20,
            "client": {
                "time_to_first_text_delta_ms": 500.0,
                "response_completed_ms": 2500.0,
            },
            "server": {"prompt_eval_ms": 300.0, "decode_tokens_per_second": 10.0},
        },
        {
            "input_tokens": 120,
            "output_tokens": 24,
            "client": {
                "time_to_first_text_delta_ms": 700.0,
                "response_completed_ms": 3100.0,
            },
            "server": {"prompt_eval_ms": 400.0, "decode_tokens_per_second": 8.0},
        },
    ]

    summary = benchmark.summarize(runs)

    assert summary["time_to_first_text_delta_ms"]["median"] == 600.0
    assert summary["decode_tokens_per_second"]["median"] == 9.0
    assert summary["input_tokens"]["median"] == 110.0
