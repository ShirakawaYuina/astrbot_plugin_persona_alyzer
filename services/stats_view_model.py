from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from .card_view_model import (
    PERSONA_ARCHETYPE_DISPLAY_MAP,
    PERSONA_ENGLISH_NAME_MAP,
    THEME_MAP,
    _build_dynamic_theme_from_image,
    _resolve_persona_image_path,
)

STATS_TOP_LIMIT = 10
CST_TIMEZONE = ZoneInfo("Asia/Shanghai")


def build_stats_view_model(stats: dict) -> dict:
    """把全局人格命中统计整理为统计图模板数据。

    排序、占比和进度条宽度都在这里算好，模板只负责排版。

    Args:
        stats: ResultStore.get_personality_stats 返回的原始统计结构。

    Returns:
        统计图模板所需的完整数据包。
    """
    total_count = max(0, int(stats.get("total_count", 0) or 0))
    raw_personality_counts = stats.get("personality_counts")
    if not isinstance(raw_personality_counts, dict):
        raw_personality_counts = {}

    personality_counts: list[tuple[str, int]] = []
    for raw_name, raw_count in raw_personality_counts.items():
        try:
            count = max(0, int(raw_count or 0))
        except (TypeError, ValueError):
            count = 0
        if count <= 0:
            continue
        personality_counts.append((str(raw_name).strip() or "未命名人格", count))

    # 次数降序，次数相同时按名称排序，保证多次渲染的榜单顺序稳定。
    ranked_counts = sorted(personality_counts, key=lambda item: (-item[1], item[0]))[
        :STATS_TOP_LIMIT
    ]
    top_count = ranked_counts[0][1] if ranked_counts else 0

    rank_items = [
        {
            "rank": rank,
            "name": name,
            "name_en": PERSONA_ENGLISH_NAME_MAP.get(name, "Unknown Archetype"),
            "archetype": PERSONA_ARCHETYPE_DISPLAY_MAP.get(name, "未归类原型"),
            "count": count,
            "share_text": f"{count / total_count * 100:.1f}%" if total_count else "-",
            "bar_percent": f"{count / top_count * 100:.1f}" if top_count else "0.0",
        }
        for rank, (name, count) in enumerate(ranked_counts, start=1)
    ]

    unlocked_count = len(personality_counts)
    total_archetypes = max(len(PERSONA_ENGLISH_NAME_MAP), unlocked_count)
    collection_percent = (
        f"{unlocked_count / total_archetypes * 100:.1f}" if total_archetypes else "0.0"
    )

    theme = dict(THEME_MAP["neutral"])
    if rank_items:
        persona_image_path = _resolve_persona_image_path(
            {"personality_name": rank_items[0]["name"]}
        )
        if persona_image_path is not None:
            theme = (
                _build_dynamic_theme_from_image(str(persona_image_path), "neutral")
                or theme
            )

    return {
        "theme": theme,
        "total_count": total_count,
        "unlocked_count": unlocked_count,
        "total_archetypes": total_archetypes,
        "collection_percent": collection_percent,
        "top_personality_name": rank_items[0]["name"] if rank_items else "暂无",
        "rank_items": rank_items,
        "generated_at": datetime.now(CST_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S CST"),
    }
