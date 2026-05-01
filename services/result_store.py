from __future__ import annotations

import json
from math import ceil
from pathlib import Path

from astrbot.core.star.star_tools import StarTools

from ..plugin_metadata import PLUGIN_NAME


class ResultStore:
    """管理最新一次分析结果与冷却信息。"""

    def __init__(self, plugin) -> None:
        self.plugin = plugin

    @staticmethod
    def _latest_key(platform_id: str, group_id: str, sender_id: str) -> str:
        return f"latest:{platform_id}:{group_id}:{sender_id}"

    @staticmethod
    def _personality_stats_key() -> str:
        return "personality_stats"

    @staticmethod
    def _personality_stats_filename() -> str:
        return "personality_stats.json"

    def _get_data_dir(self) -> Path:
        data_dir = getattr(self.plugin, "data_dir", None) or getattr(
            self.plugin,
            "plugin_data_dir",
            None,
        )
        if data_dir:
            return Path(data_dir)
        return StarTools.get_data_dir(PLUGIN_NAME)

    def _get_personality_stats_path(self) -> Path:
        return self._get_data_dir() / self._personality_stats_filename()

    def _read_personality_stats_payload(self) -> dict | None:
        stats_path = self._get_personality_stats_path()
        if not stats_path.exists():
            return None
        try:
            payload = json.loads(stats_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _write_personality_stats_payload(self, payload: dict) -> None:
        stats_path = self._get_personality_stats_path()
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = stats_path.with_suffix(f"{stats_path.suffix}.tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(stats_path)

    async def _safe_get_kv_data(self, key: str, default):
        """单测或早期初始化阶段可能尚未拿到插件存储上下文，此时回退默认值。"""
        try:
            return await self.plugin.get_kv_data(key, default)
        except AttributeError:
            return default

    async def _safe_put_kv_data(self, key: str, value) -> None:
        """只有插件存储可用时才落盘统计，避免非运行态环境误报。"""
        try:
            await self.plugin.put_kv_data(key, value)
        except AttributeError:
            return None

    async def save_latest(
        self, platform_id: str, group_id: str, sender_id: str, payload: dict
    ) -> None:
        await self._safe_put_kv_data(
            self._latest_key(platform_id, group_id, sender_id),
            payload,
        )

    async def get_latest(
        self, platform_id: str, group_id: str, sender_id: str
    ) -> dict | None:
        return await self._safe_get_kv_data(
            self._latest_key(platform_id, group_id, sender_id),
            None,
        )

    async def get_personality_stats(self) -> dict:
        """统一返回稀有度统计结构，避免上层到处处理缺省字段。"""
        payload = self._read_personality_stats_payload()
        if not isinstance(payload, dict):
            return {"total_count": 0, "personality_counts": {}}
        personality_counts = payload.get("personality_counts")
        if not isinstance(personality_counts, dict):
            personality_counts = {}
        total_count = int(payload.get("total_count", 0) or 0)
        return {
            "total_count": max(0, total_count),
            "personality_counts": {
                str(key): max(0, int(value or 0))
                for key, value in personality_counts.items()
            },
        }

    async def record_personality_occurrence(self, personality_name: str) -> dict:
        """每次分析成功后累计人格命中次数，并直接返回当前卡片所需的稀有度数据。"""
        stats = await self.get_personality_stats()
        normalized_name = str(personality_name or "").strip() or "未命名人格"
        personality_counts = dict(stats["personality_counts"])
        total_count = stats["total_count"] + 1
        personality_count = personality_counts.get(normalized_name, 0) + 1
        personality_counts[normalized_name] = personality_count
        payload = {
            "total_count": total_count,
            "personality_counts": personality_counts,
        }
        self._write_personality_stats_payload(payload)
        rarity_ratio = personality_count / total_count if total_count else 0.0
        return {
            **payload,
            "personality_name": normalized_name,
            "personality_count": personality_count,
            "rarity_ratio": rarity_ratio,
            "rarity_top_percent": ceil(rarity_ratio * 100) if total_count else 0,
        }
