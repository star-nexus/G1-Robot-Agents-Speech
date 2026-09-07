from __future__ import annotations

import json
from pathlib import Path

import pytest

from g1_speech.agent_runtime import (
    ConversationalLoop,
    KeywordLoreProvider,
    NullMemory,
    load_role_package,
)
from g1_speech.local_agent import LocalAgentSettings, build_agent_runtime


class RecordingModel:
    def __init__(self) -> None:
        self.calls = []

    def stream(self, messages):
        self.calls.append(messages)
        return iter(("回答",))


def _write_cards(path, cards):
    path.write_text(
        "\n".join(json.dumps(card, ensure_ascii=False) for card in cards),
        encoding="utf-8",
    )


def test_keyword_lore_matches_multilingual_aliases_without_a_model_call(tmp_path):
    core = tmp_path / "core.md"
    core.write_text("固定角色知识", encoding="utf-8")
    cards = tmp_path / "cards.jsonl"
    _write_cards(
        cards,
        [
            {
                "id": "cloud",
                "aliases": ["Cloud", "克劳德", "クラウド・ストライフ"],
                "facts": "クラウドは幼なじみ。",
            },
            {
                "id": "barret",
                "aliases": ["Barret", "巴雷特", "バレット"],
                "facts": "バレットは仲間。",
            },
        ],
    )
    provider = KeywordLoreProvider.from_files(
        core_path=core,
        cards_path=cards,
        max_cards=1,
    )

    assert provider.core_context() == "固定角色知识"
    assert provider.relevant_context("你认识Ｃｌｏｕｄ吗？") == (
        "- クラウドは幼なじみ。"
    )
    assert provider.relevant_context("天气怎么样") == ""


def test_conversational_loop_injects_core_and_only_relevant_lore(tmp_path):
    cards = tmp_path / "cards.jsonl"
    _write_cards(
        cards,
        [
            {
                "id": "cloud",
                "aliases": ["克劳德"],
                "facts": "クラウドは幼なじみ。",
            }
        ],
    )
    provider = KeywordLoreProvider.from_files(
        core_path=None,
        cards_path=cards,
    )
    model = RecordingModel()
    loop = ConversationalLoop(
        system_prompt="角色规则",
        model=model,
        memory=NullMemory(),
        knowledge=provider,
    )

    assert list(loop.stream_response("克劳德是谁？")) == ["回答"]
    assert len(model.calls) == 1
    assert model.calls[0][0] == {"role": "system", "content": "角色规则"}
    assert model.calls[0][1]["role"] == "system"
    assert "クラウドは幼なじみ" in model.calls[0][1]["content"]
    assert model.calls[0][-1] == {"role": "user", "content": "克劳德是谁？"}


def test_lore_cards_are_validated_at_startup(tmp_path):
    cards = tmp_path / "cards.jsonl"
    cards.write_text('{"id":"broken","aliases":[],"facts":"x"}', encoding="utf-8")

    with pytest.raises(ValueError, match="aliases"):
        KeywordLoreProvider.from_files(core_path=None, cards_path=cards)


def test_runtime_factory_loads_role_lore_once_and_keeps_one_model_request():
    root = Path(__file__).resolve().parents[1]
    role = load_role_package(root / "roles" / "tifa-lockhart")
    model = RecordingModel()
    runtime = build_agent_runtime(
        LocalAgentSettings(system_prompt=role.prompt, history_turns=3),
        client=model,
        role_package=role,
    )

    assert list(runtime.stream_response("你认识Cloud吗？")) == ["回答"]
    assert len(model.calls) == 1
    assert "Canonical role knowledge" in model.calls[0][0]["content"]
    assert any(
        "クラウドはニブルヘイムで育った幼なじみ" in message["content"]
        for message in model.calls[0]
    )


@pytest.mark.parametrize(
    "query",
    [
        "你现在穿的什么衣服？",
        "第七天堂的衣服长什么样子，你描述一下。",
        "今はどんな服を着ているの？",
        "What are you wearing?",
    ],
)
def test_bundled_tifa_outfit_lore_matches_common_questions(query):
    root = Path(__file__).resolve().parents[1]
    role = load_role_package(root / "roles" / "tifa-lockhart")
    assert role.knowledge.cards_path is not None
    provider = KeywordLoreProvider.from_files(
        core_path=role.knowledge.core_path,
        cards_path=role.knowledge.cards_path,
        max_cards=role.knowledge.max_cards,
    )

    relevant = provider.relevant_context(query)

    assert "白いクロップド丈のタンクトップ" in relevant
    assert "黒いミニスカート" in relevant
    assert "赤い編み上げブーツ" in relevant
    assert "セブンスヘブン専用の作業着ではない" in relevant


@pytest.mark.parametrize(
    "query",
    [
        "蒂法的三围是多少？",
        "スリーサイズを教えて。",
        "What are your body measurements?",
    ],
)
def test_bundled_tifa_measurements_include_source_period_caveat(query):
    root = Path(__file__).resolve().parents[1]
    role = load_role_package(root / "roles" / "tifa-lockhart")
    assert role.knowledge.cards_path is not None
    provider = KeywordLoreProvider.from_files(
        core_path=role.knowledge.core_path,
        cards_path=role.knowledge.cards_path,
        max_cards=role.knowledge.max_cards,
    )

    relevant = provider.relevant_context(query)

    assert "B92・W60・H88 cm" in relevant
    assert "初期デザイン資料" in relevant
    assert "現行公式プロフィールとして再公表された数値ではない" in relevant


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("你为什么喜欢夏天，不会融化吗？", "随身飘雪"),
        ("你是谁创造出来的？", "艾莎的冰雪魔法"),
        ("你觉得真爱是什么？", "把他们的需要放在自己前面"),
        ("你的胡萝卜鼻子会掉吗？", "三段雪团"),
    ],
)
def test_bundled_olaf_lore_matches_first_film_questions(query, expected):
    root = Path(__file__).resolve().parents[1]
    role = load_role_package(root / "roles" / "olaf")
    assert role.knowledge.cards_path is not None
    provider = KeywordLoreProvider.from_files(
        core_path=role.knowledge.core_path,
        cards_path=role.knowledge.cards_path,
        max_cards=role.knowledge.max_cards,
    )

    assert expected in provider.relevant_context(query)
