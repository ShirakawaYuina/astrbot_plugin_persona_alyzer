from __future__ import annotations

import json
import re
from dataclasses import dataclass

from astrbot.api import logger

from .message_sampler import SampledMessage
from .prompt_loader import PromptBundle

REQUIRED_DIMENSIONS = (
    "驱动极性",
    "逻辑载荷",
    "社交熵值",
    "变革烈度",
    "共情阈值",
)


def _render_samples(messages: list[SampledMessage | str]) -> str:
    rendered: list[str] = []
    for item in messages:
        if isinstance(item, SampledMessage):
            rendered.append(
                f"[{item.role}/{item.relation}][{item.sender_name}] {item.text}"
            )
        else:
            rendered.append(f"- {item}")
    return "\n".join(rendered)


def _replace_instruction_placeholders(
    instruction: str, *, sender_name: str, evidence_item_limit: int
) -> str:
    placeholder_pattern = re.compile(r"\{(sender_name|evidence_item_limit)\}")
    replacements = {
        "sender_name": sender_name,
        "evidence_item_limit": str(evidence_item_limit),
    }

    return placeholder_pattern.sub(
        lambda match: replacements[match.group(1)], instruction
    )


def _normalize_evidence_item_limit(value: int) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return 1
    return max(1, normalized)


def _count_sentences(text: str) -> int:
    pieces = [part.strip() for part in re.split(r"(?<=[。！？!?；;])\s*", text.strip())]
    return sum(1 for part in pieces if part)


def _truncate_text(value: str, limit: int = 400) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _serialize_debug_value(value) -> str:
    try:
        serialized = json.dumps(value, ensure_ascii=False)
    except TypeError:
        serialized = str(value)
    return _truncate_text(serialized)


@dataclass
class PersonalityAnalyzer:
    """调用固定 Provider 完成人格判定。"""

    bundle: PromptBundle
    provider: object

    async def analyze(
        self,
        *,
        sender_name: str,
        messages: list[SampledMessage | str],
        evidence_item_limit: int = 2,
    ) -> dict:
        evidence_item_limit = _normalize_evidence_item_limit(evidence_item_limit)
        lexicon_text = self.bundle.analysis_lexicon_text or self.bundle.archetype_prompt
        instruction = _replace_instruction_placeholders(
            self.bundle.analysis_instruction,
            sender_name=sender_name,
            evidence_item_limit=evidence_item_limit,
        )
        prompt = (
            f"{instruction}\n\n"
            f"证据上限：{evidence_item_limit}\n"
            "请从给定的 36 种人格中选择最匹配的一项，并严格输出 JSON。\n"
            "人格判断要优先依据[target/self]消息。\n"
            "[context/*] 只用于理解互动对象、语气和潜台词，不得把上下文说话者特征归因给分析对象。\n"
            f"人格库：\n{lexicon_text}\n\n"
            f"量化维度:\n{self.bundle.dimension_reference}\n\n"
            f"分析对象: {sender_name}\n"
            f"互动样本:\n{_render_samples(messages)}"
        )
        response = await self.provider.text_chat(
            contexts=[{"role": "user", "content": prompt}]
        )
        response_text = (response.completion_text or "").strip()
        payload = json.loads(response_text)
        try:
            self._validate(payload, evidence_item_limit=evidence_item_limit)
        except ValueError as exc:
            logger.warning(
                "[Personality] 人格分析结果校验失败 错误=%s 证据上限=%s 人格=%s 证据项=%s 原始返回=%s",
                str(exc),
                evidence_item_limit,
                str(payload.get("personality_name") or "").strip(),
                _serialize_debug_value(payload.get("evidence_items")),
                _truncate_text(response_text),
            )
            raise
        return payload

    def _validate(self, payload: dict, *, evidence_item_limit: int) -> None:
        if payload.get("personality_name") not in self.bundle.allowed_personality_names:
            raise ValueError("Unknown personality")

        evidence_items = payload.get("evidence_items")
        if not isinstance(evidence_items, list) or not evidence_items:
            raise ValueError("Missing evidence items")
        if len(evidence_items) > evidence_item_limit:
            raise ValueError("Too many evidence items")

        for item in evidence_items:
            if not isinstance(item, dict):
                raise ValueError("Invalid evidence item")

            quote = item.get("quote")
            analysis = item.get("analysis")
            if not isinstance(quote, str) or not quote.strip():
                raise ValueError("Invalid evidence item")
            if not isinstance(analysis, str) or not analysis.strip():
                raise ValueError("Invalid evidence item")
            if not 1 <= _count_sentences(analysis) <= 3:
                raise ValueError("Invalid evidence item")

        dimensions = payload.get("dimensions", {})
        for key in REQUIRED_DIMENSIONS:
            if key not in dimensions:
                raise ValueError(f"Missing dimension: {key}")
