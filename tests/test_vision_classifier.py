from __future__ import annotations

import io
import json

from star_runtime.perception.vision import (
    VISION,
    LlamaVisionClassifier,
    LlamaVisionClassifierSettings,
)


class JsonResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


def test_llama_classifier_uses_constrained_non_streaming_request():
    captured = {}

    def opener(http_request, timeout):
        captured["payload"] = json.loads(http_request.data)
        captured["timeout"] = timeout
        return JsonResponse(b'{"choices":[{"message":{"content":"V"}}]}')

    classifier = LlamaVisionClassifier(
        LlamaVisionClassifierSettings(timeout_seconds=7),
        opener=opener,
    )

    route, reason = classifier(
        "门关了吗？",
        [
            {"role": "user", "content": "看看门"},
            {"role": "assistant", "content": "关着"},
        ],
        True,
    )

    assert route == VISION
    assert reason.endswith("V")
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["grammar"] == 'root ::= "V" | "T" | "U"'
    assert captured["timeout"] == 7
