"""Dependency-free role knowledge providers for low-latency robot dialogue."""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class KnowledgeProvider(Protocol):
    """Supplies fixed and query-specific role knowledge without model calls."""

    def core_context(self) -> str: ...

    def relevant_context(self, query: str) -> str: ...


class NullKnowledge:
    def core_context(self) -> str:
        return ""

    def relevant_context(self, query: str) -> str:
        return ""


@dataclass(frozen=True)
class LoreCard:
    card_id: str
    aliases: tuple[str, ...]
    facts: str
    normalized_aliases: tuple[str, ...]


class KeywordLoreProvider:
    """Matches pre-normalized multilingual aliases entirely in host memory."""

    def __init__(
        self,
        *,
        core: str = "",
        cards: tuple[LoreCard, ...] = (),
        max_cards: int = 2,
    ) -> None:
        if max_cards < 1:
            raise ValueError("knowledge max_cards must be greater than zero")
        self._core = core.strip()
        self._cards = cards
        self._max_cards = max_cards

    @classmethod
    def from_files(
        cls,
        *,
        core_path: Path | None,
        cards_path: Path,
        max_cards: int = 2,
    ) -> KeywordLoreProvider:
        core = (
            core_path.read_text(encoding="utf-8").strip()
            if core_path is not None
            else ""
        )
        cards: list[LoreCard] = []
        seen_ids: set[str] = set()
        for line_number, line in enumerate(
            cards_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid lore card JSON at {cards_path}:{line_number}"
                ) from exc
            if not isinstance(raw, dict):
                raise ValueError(
                    f"lore card must be an object at {cards_path}:{line_number}"
                )
            card_id = _required_text(raw, "id", cards_path, line_number)
            if card_id in seen_ids:
                raise ValueError(f"duplicate lore card id: {card_id}")
            seen_ids.add(card_id)
            aliases_raw = raw.get("aliases")
            if (
                not isinstance(aliases_raw, list)
                or not aliases_raw
                or any(not isinstance(item, str) or not item.strip() for item in aliases_raw)
            ):
                raise ValueError(
                    f"lore card aliases must be a non-empty string array at "
                    f"{cards_path}:{line_number}"
                )
            aliases = tuple(item.strip() for item in aliases_raw)
            normalized_aliases = tuple(
                normalized
                for alias in aliases
                if (normalized := _normalize(alias))
            )
            cards.append(
                LoreCard(
                    card_id=card_id,
                    aliases=aliases,
                    facts=_required_text(raw, "facts", cards_path, line_number),
                    normalized_aliases=normalized_aliases,
                )
            )
        return cls(core=core, cards=tuple(cards), max_cards=max_cards)

    def core_context(self) -> str:
        return self._core

    def relevant_context(self, query: str) -> str:
        normalized_query = _normalize(query)
        if not normalized_query:
            return ""
        matches: list[tuple[int, int, LoreCard]] = []
        for index, card in enumerate(self._cards):
            score = max(
                (
                    len(alias)
                    for alias in card.normalized_aliases
                    if alias in normalized_query
                ),
                default=0,
            )
            if score:
                matches.append((score, -index, card))
        if not matches:
            return ""
        matches.sort(reverse=True, key=lambda item: (item[0], item[1]))
        selected = [item[2] for item in matches[: self._max_cards]]
        return "\n".join(f"- {card.facts}" for card in selected)


def _normalize(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in folded if character.isalnum())


def _required_text(
    raw: dict,
    key: str,
    path: Path,
    line_number: int,
) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"lore card {key} must be a non-empty string at {path}:{line_number}"
        )
    return value.strip()
