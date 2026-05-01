from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PersonalityLexiconEntry:
    archetype_heading: str
    archetype_group_cn: str
    archetype_group_en: str
    archetype_core: str
    personality_name: str
    personality_name_en: str
    alignment: str
    summary: str
    traits: str
    intro: str

    @property
    def archetype_group(self) -> str:
        return self.archetype_group_cn

    @property
    def summary_line(self) -> str:
        return (
            f"- {self.personality_name} ({self.personality_name_en} / "
            f"{self.alignment.title()})：{self.summary}"
        )

    @property
    def traits_line(self) -> str:
        return f"  - 特征：{self.traits}"


@dataclass(frozen=True)
class PromptBundle:
    archetype_prompt: str
    dimension_reference: str
    analysis_instruction: str
    allowed_personality_names: tuple[str, ...] = ()
    analysis_lexicon_text: str = ""
    lexicon_entries: tuple[PersonalityLexiconEntry, ...] = ()
    persona_intro_map: dict[str, str] = field(default_factory=dict)


ANALYSIS_INSTRUCTION = """
你是一位洞察力极强的人格分析师，请基于群成员的聊天信息，分析其性格特点、表达风格、情绪稳定性、社交倾向和可能的人际模式。要求客观、具体、不贴标签，不过度解读。对{sender_name}进行深度分析，。并最多使用 {evidence_item_limit} 条证据。

重要要求：
1. 必须引用原文：分析时必须摘录 1-2 句用户的具体发言或交互细节，并放入 JSON 的 "evidence_items" 数组。
2. 每个证据项都必须包含 quote 和 analysis 两个字段，其中 analysis 必须用 2-3 句解释这段原文为什么能支撑判断。
3. 结论必须克制：只能依据给定发言和对话细节推断，不要编造未出现的经历、身份或现实背景。
4. 输出必须是合法 JSON：不要输出 Markdown、解释文字或代码块。

请严格输出以下 JSON 结构：
{
  "personality_id": "string",
  "personality_name": "string",
  "personality_name_en": "string",
  "archetype_group": "string",
  "alignment": "positive|neutral|negative",
  "matching_rate": 0,
  "intensity_tag": "string",
  "description": "结合原文证据的人格分析",
  "persona_quote": "符合人格形象的一句语录",
  "evidence_items": [
    {
      "quote": "原文短句",
      "analysis": "2-3句分析"
    }
  ],
  "dimensions": {
    "驱动极性": 0,
    "逻辑载荷": 0,
    "社交熵值": 0,
    "变革烈度": 0,
    "共情阈值": 0
  }
}

约束：
- "evidence_items" 至少包含 1 项，最多使用 {evidence_item_limit} 项。
- 每个 evidence item 的 quote 必须来自给定消息。
- 每个 analysis 必须是 2-3 句中文分析。
- "matching_rate" 为 0-100 的整数。
- 五个维度都必须给出 0-100 的整数分数。""".strip()

GROUP_PATTERN = re.compile(
    r"^##\s+(?P<index>\d+)\.\s+(?P<group_cn>[^(\n]+?)\s+\((?P<group_en>[^)]+)\)\s*-\s*核心[:：]\s*(?P<core>.+?)\s*$",
    re.MULTILINE,
)
ENTRY_PATTERN = re.compile(
    r"^- \*\*(?P<cn>[^(\n]+?)\s+\((?P<en>[^/)\n]+)\s*/\s*(?P<alignment>[^)\n]+)\)\*\*[:：]\s*(?P<summary>.+?)\s*\n"
    r"\s+- \*特征\*[:：]\s*(?P<traits>.+?)\s*\n"
    r"\s+- \*介绍\*[:：]\s*(?P<intro>.*?)(?=\n- \*\*|\n## |\n---|\Z)",
    re.MULTILINE | re.DOTALL,
)


def _normalize_intro(value: str) -> str:
    return value.strip()


def _parse_lexicon_entries(markdown_text: str) -> tuple[PersonalityLexiconEntry, ...]:
    group_matches = list(GROUP_PATTERN.finditer(markdown_text))
    entries: list[PersonalityLexiconEntry] = []

    for index, group_match in enumerate(group_matches):
        section_start = group_match.end()
        section_end = (
            group_matches[index + 1].start()
            if index + 1 < len(group_matches)
            else len(markdown_text)
        )
        section_text = markdown_text[section_start:section_end]
        section_entries = list(ENTRY_PATTERN.finditer(section_text))

        if len(section_entries) != 3:
            raise ValueError(
                "Expected 3 personality entries under "
                f"{group_match.group('group_cn').strip()}, got {len(section_entries)}"
            )

        for entry_match in section_entries:
            entries.append(
                PersonalityLexiconEntry(
                    archetype_heading=(
                        f"## {group_match.group('index').strip()}. "
                        f"{group_match.group('group_cn').strip()} "
                        f"({group_match.group('group_en').strip()}) - 核心："
                        f"{group_match.group('core').strip()}"
                    ),
                    archetype_group_cn=group_match.group("group_cn").strip(),
                    archetype_group_en=group_match.group("group_en").strip(),
                    archetype_core=group_match.group("core").strip(),
                    personality_name=entry_match.group("cn").strip(),
                    personality_name_en=entry_match.group("en").strip(),
                    alignment=entry_match.group("alignment").strip().lower(),
                    summary=entry_match.group("summary").strip(),
                    traits=entry_match.group("traits").strip(),
                    intro=_normalize_intro(entry_match.group("intro")),
                )
            )

    if len(entries) != 36:
        raise ValueError(f"Expected 36 personality entries, got {len(entries)}")

    return tuple(entries)


def _build_analysis_lexicon_text(
    entries: tuple[PersonalityLexiconEntry, ...],
) -> str:
    lines: list[str] = []
    current_group: str | None = None

    for entry in entries:
        group = entry.archetype_heading
        if group != current_group:
            if lines:
                lines.append("")
            lines.append(entry.archetype_heading)
            current_group = group

        lines.append(entry.summary_line)
        lines.append(entry.traits_line)

    return "\n".join(lines)


class PromptLoader:
    """加载人格分析所需的本地提示词资源。"""

    def __init__(self, plugin_root: Path) -> None:
        self.plugin_root = plugin_root

    def load(self) -> PromptBundle:
        prompt_dir = self.plugin_root / "prompts"
        archetype_prompt = (prompt_dir / "人格提示词.md").read_text(encoding="utf-8")
        dimension_reference = (prompt_dir / "人格量化分析维度.md").read_text(
            encoding="utf-8"
        )
        lexicon_entries = _parse_lexicon_entries(archetype_prompt)
        names = tuple(item.personality_name for item in lexicon_entries)

        if len(names) != 36:
            raise ValueError(f"Expected 36 personality names, got {len(names)}")

        return PromptBundle(
            archetype_prompt=archetype_prompt,
            dimension_reference=dimension_reference,
            analysis_instruction=ANALYSIS_INSTRUCTION,
            allowed_personality_names=names,
            analysis_lexicon_text=_build_analysis_lexicon_text(lexicon_entries),
            lexicon_entries=lexicon_entries,
            persona_intro_map={
                item.personality_name: item.intro for item in lexicon_entries
            },
        )
