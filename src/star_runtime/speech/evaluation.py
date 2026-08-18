"""Dependency-free ASR accuracy metrics for on-device acceptance tests."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence


def normalize_characters(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return [character for character in normalized if not character.isspace()]


def normalize_content_characters(text: str) -> list[str]:
    """Normalize CER content while excluding Unicode punctuation.

    The strict character metric intentionally retains punctuation. ASR precision
    comparisons also need a content-only view so optional punctuation does not
    look like an acoustic or lexical regression.
    """
    return [
        character
        for character in normalize_characters(text)
        if not unicodedata.category(character).startswith("P")
    ]


def normalize_words(text: str) -> list[str]:
    return unicodedata.normalize("NFKC", text).casefold().split()


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_item in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hypothesis_index] + 1,
                    previous[hypothesis_index - 1]
                    + (reference_item != hypothesis_item),
                )
            )
        previous = current
    return previous[-1]


def error_rate(reference: Sequence[str], hypothesis: Sequence[str]) -> float | None:
    if not reference:
        return None
    return edit_distance(reference, hypothesis) / len(reference)
