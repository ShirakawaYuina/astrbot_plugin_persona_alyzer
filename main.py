from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from PIL import Image

from astrbot.api import AstrBotConfig, logger, star
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.core import astrbot_config, file_token_service
from astrbot.core.message.components import At, BaseMessageComponent
from astrbot.core.star.star_tools import StarTools

from .plugin_metadata import (
    PLUGIN_AUTHOR,
    PLUGIN_DESC,
    PLUGIN_NAME,
    PLUGIN_REPO,
    PLUGIN_ROOT,
    PLUGIN_VERSION,
)
from .services.analyzer import PersonalityAnalyzer
from .services.card_view_model import build_card_view_model
from .services.message_sampler import sample_interaction_messages
from .services.prompt_loader import PromptLoader
from .services.result_store import ResultStore

IMAGE_SEND_MODE_BASE64 = "base64"
IMAGE_SEND_MODE_URL = "url"


@star.register(
    PLUGIN_NAME,
    PLUGIN_AUTHOR,
    PLUGIN_DESC,
    PLUGIN_VERSION,
    PLUGIN_REPO,
)
class PersonalityPlugin(star.Star):
    """人格分析插件主入口。"""

    ANALYSIS_TIMEZONE = ZoneInfo("Asia/Shanghai")

    def __init__(
        self, context: star.Context, config: AstrBotConfig | None = None
    ) -> None:
        super().__init__(context)
        self.config = config or {}
        self.plugin_root = PLUGIN_ROOT
        self.data_dir = StarTools.get_data_dir(PLUGIN_NAME)
        self.prompt_loader = PromptLoader(self.plugin_root)
        self.result_store = ResultStore(self)

    def _get_config_value(self, key: str, default):
        """统一读取插件配置，避免缺字段时在主流程里散落兜底逻辑。"""
        if hasattr(self.config, "get"):
            return self.config.get(key, default)
        return default

    def _get_image_send_mode(self) -> str:
        """读取人格卡片发送模式，未知值回退到 base64 以保持旧行为。"""
        mode = str(self._get_config_value("image_send_mode", IMAGE_SEND_MODE_BASE64))
        clean_mode = mode.strip().lower()
        if clean_mode == IMAGE_SEND_MODE_URL:
            return IMAGE_SEND_MODE_URL
        return IMAGE_SEND_MODE_BASE64

    def _resolve_image_send_url_base(self) -> str:
        """解析 NapCat 可访问的 AstrBot 文件服务地址。"""
        configured_base = str(
            self._get_config_value("image_send_url_base", "") or ""
        ).strip()
        fallback_base = str(astrbot_config.get("callback_api_base", "") or "").strip()
        clean_base = (configured_base or fallback_base).rstrip("/")
        if not clean_base:
            raise RuntimeError(
                "URL 发送模式需要配置 image_send_url_base 或 AstrBot callback_api_base"
            )
        return clean_base

    async def _build_card_image_url(self, image_path: str) -> str:
        """注册人格卡片到文件服务，生成 OneBot/NapCat 可拉取的 HTTP URL。"""
        url_base = self._resolve_image_send_url_base()
        token = await file_token_service.register_file(str(Path(image_path)))
        return f"{url_base}/api/file/{token}"

    async def _send_card_image(
        self,
        event: AstrMessageEvent,
        image_path: str,
    ):
        """发送人格卡片；URL 模式会绕过 AstrBot 的 OneBot base64 转换。"""
        if self._get_image_send_mode() != IMAGE_SEND_MODE_URL:
            return event.image_result(image_path)

        bot = getattr(event, "bot", None)
        if bot is None:
            raise RuntimeError("URL 发送模式需要 aiocqhttp 事件暴露 bot 实例")

        image_message = [
            {
                "type": "image",
                "data": {"file": await self._build_card_image_url(image_path)},
            }
        ]
        group_id = str(event.get_group_id() or "").strip()
        if group_id.isdigit():
            await bot.send_group_msg(group_id=int(group_id), message=image_message)
            return None

        sender_id = str(event.get_sender_id() or "").strip()
        if sender_id.isdigit():
            await bot.send_private_msg(user_id=int(sender_id), message=image_message)
            return None

        raise RuntimeError("URL 发送模式缺少有效的群号或用户 ID")

    def _get_session_type(self, event: AstrMessageEvent) -> str:
        return "群聊" if str(event.get_group_id() or "").strip() else "私聊"

    def _log_analysis_event(self, phase: str, **fields) -> None:
        """统一格式化日志字段，便于群聊和私聊链路横向对比。"""
        serialized_fields = " ".join(
            f"{key}={str(value).replace(chr(10), ' ').strip()}"
            for key, value in fields.items()
        )
        logger.info(f"[Personality] {phase} {serialized_fields}".rstrip())

    def _build_summary_text(self, normalized: dict) -> str:
        """摘要文本用于群内快速浏览，和卡片承载不同的信息密度。"""
        return (
            f"今日人格：{normalized['personality_name']}\n"
            f"原型大类：{normalized['archetype_group']}\n"
            f"匹配度：{normalized['matching_rate']}%\n"
            f"判定依据：{normalized['description']}"
        )

    def _get_analysis_now(self) -> datetime:
        return datetime.now(self.ANALYSIS_TIMEZONE)

    def _get_analysis_window_start(
        self, current_time: datetime, day_span: int
    ) -> datetime:
        localized_now = current_time.astimezone(self.ANALYSIS_TIMEZONE)
        return localized_now.replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        ) - timedelta(days=max(day_span - 1, 0))

    def _find_rendered_content_right_edge(
        self, image: Image.Image, threshold: float = 4.0
    ) -> int:
        """根据右侧空白区与卡片主体的列差异，定位真实内容的右边界。"""
        width, height = image.size
        if width <= 0 or height <= 0:
            return width

        sample_y_positions = list(range(24, max(height - 24, 25), 24))
        if not sample_y_positions:
            sample_y_positions = [max(height // 2, 0)]

        for x_axis in range(width - 2, -1, -1):
            diffs = []
            for y_axis in sample_y_positions:
                pixel = image.getpixel((x_axis, y_axis))[:3]
                right_pixel = image.getpixel((x_axis + 1, y_axis))[:3]
                diffs.append(
                    sum(
                        abs(pixel[channel] - right_pixel[channel])
                        for channel in range(3)
                    )
                )

            if sum(diffs) / len(diffs) > threshold:
                return min(x_axis + 1, width)
        return width

    def _trim_rendered_image_width(self, image_path: str) -> str:
        """仅裁掉截图右侧的多余空白，保留高清渲染产生的真实内容宽度。"""
        path = Path(image_path)
        if not path.exists():
            return image_path

        try:
            with Image.open(path) as image:
                width, height = image.size
                content_right_edge = self._find_rendered_content_right_edge(image)
                if content_right_edge <= 0 or content_right_edge >= width:
                    return image_path

                cropped = image.crop((0, 0, content_right_edge, height))
                save_kwargs = {"format": image.format} if image.format else {}
                cropped.save(path, **save_kwargs)
        except Exception as exc:
            logger.warning(
                "[Personality] 渲染图片裁切失败 路径=%s 错误=%s",
                image_path,
                exc,
            )
        return image_path

    def _prepare_rendered_image_for_send(self, image_path: str) -> str | None:
        prepared_path = self._trim_rendered_image_width(image_path)
        path = Path(prepared_path)
        if not path.exists():
            logger.warning(
                "[Personality] 渲染图片不存在，取消发送 路径=%s",
                prepared_path,
            )
            return None

        try:
            with Image.open(path) as image:
                image.load()
        except Exception as exc:
            logger.warning(
                "[Personality] 渲染图片无效，取消发送 路径=%s 错误=%s",
                prepared_path,
                exc,
            )
            return None

        return prepared_path

    async def _render_personality_card_image(self, render_payload: dict) -> str | None:
        retry_count = max(int(self._get_config_value("render_retry_count", 0) or 0), 0)
        retry_interval_seconds = max(
            int(self._get_config_value("render_retry_interval_seconds", 10) or 10),
            0,
        )
        total_attempts = 1 + retry_count
        last_exception: Exception | None = None

        for attempt in range(1, total_attempts + 1):
            try:
                image_path = await self.html_render(
                    self._load_card_template(),
                    render_payload,
                    return_url=False,
                    options={
                        "full_page": True,
                        "type": "png",
                        "quality": 100,
                        "scale": "device",
                        "device_scale_factor_level": "high",
                    },
                )
                return self._prepare_rendered_image_for_send(image_path)
            except Exception as exc:
                last_exception = exc
                if attempt >= total_attempts:
                    break
                logger.info(
                    "[Personality] 人格卡片渲染失败，准备重试 当前次数=%s 总次数=%s 间隔秒数=%s 错误=%s",
                    attempt,
                    total_attempts,
                    retry_interval_seconds,
                    exc,
                )
                await asyncio.sleep(retry_interval_seconds)

        logger.warning(
            "[Personality] 人格卡片渲染失败 已尝试次数=%s 错误=%s",
            total_attempts,
            last_exception,
        )
        return None

    def _build_insufficient_message(
        self,
        *,
        day_span: int,
        min_msg_threshold: int,
        target_message_count: int,
    ) -> str:
        if day_span <= 1:
            return (
                "今日发言太少，无法支撑分析。"
                f"至少需要 {min_msg_threshold} 条，当前仅采集到 {target_message_count} 条。"
            )
        return (
            f"近 {day_span} 天发言太少，无法支撑分析。"
            f"至少需要 {min_msg_threshold} 条，当前仅采集到 {target_message_count} 条。"
        )

    def _build_sample_time_window(self, sampled_messages: list) -> tuple[str, str]:
        target_times = [
            item.created_at
            for item in sampled_messages
            if getattr(item, "role", "") == "target"
            and isinstance(getattr(item, "created_at", None), datetime)
        ]
        if not target_times:
            return "", ""
        return self._format_log_datetime(min(target_times)), self._format_log_datetime(
            max(target_times)
        )

    def _format_log_datetime(self, value: datetime | None) -> str:
        """统一把诊断日志里的时间格式化为 CST，避免排查时混入 UTC。"""
        if not isinstance(value, datetime):
            return ""
        normalized_value = value
        if normalized_value.tzinfo is None:
            normalized_value = normalized_value.replace(tzinfo=timezone.utc)
        return normalized_value.astimezone(self.ANALYSIS_TIMEZONE).isoformat()

    def _resolve_mentioned_target(
        self, event: AstrMessageEvent
    ) -> tuple[str, str] | None:
        """群聊里优先读取第一个 @ 目标，未命中时再回退到触发者本人。"""
        messages = event.get_messages() if hasattr(event, "get_messages") else []
        for component in messages:
            if isinstance(component, At):
                target_id = str(component.qq).strip()
                if not target_id or target_id == "all":
                    continue
                target_name = str(component.name or "").strip() or target_id
                return target_id, target_name

            if not isinstance(component, BaseMessageComponent) and isinstance(
                component, dict
            ):
                if str(component.get("type", "")).lower() != "at":
                    continue
                data = (
                    component.get("data")
                    if isinstance(component.get("data"), dict)
                    else {}
                )
                target_id = str(
                    data.get("qq") or data.get("user_id") or component.get("qq") or ""
                ).strip()
                if not target_id or target_id == "all":
                    continue
                target_name = str(
                    data.get("name") or component.get("name") or target_id
                ).strip()
                return target_id, target_name or target_id
        return None

    def _resolve_analysis_target(self, event: AstrMessageEvent) -> tuple[str, str, str]:
        """人格分析只支持群聊，私聊场景直接在入口层拦截。"""
        group_id = str(event.get_group_id() or "").strip()
        if group_id:
            mentioned_target = self._resolve_mentioned_target(event)
            if mentioned_target:
                target_id, target_name = mentioned_target
                return group_id, target_id, target_name

            sender_id = event.get_sender_id()
            sender_name = event.get_sender_name() or sender_id
            return group_id, sender_id, sender_name
        raise ValueError("请在目标群内使用该命令。")

    async def _load_group_history_within_range(
        self,
        *,
        platform_id: str,
        group_id: str,
        lower_bound: datetime,
        page_size: int = 200,
    ) -> list:
        """按时间范围持续翻页，直到覆盖分析窗口内的全部历史消息。"""
        page = 1
        records: list = []
        while True:
            logger.debug(
                "[Personality] 平台历史拉取 页码=%s 平台=%s 群组=%s 每页数量=%s",
                page,
                platform_id,
                group_id,
                page_size,
            )
            batch = await self.context.message_history_manager.get(
                platform_id,
                group_id,
                page=page,
                page_size=page_size,
            )
            if not batch:
                logger.debug(
                    "[Personality] 平台历史拉取为空 页码=%s 平台=%s 群组=%s",
                    page,
                    platform_id,
                    group_id,
                )
                break
            records.extend(batch)
            oldest = min(getattr(item, "created_at", lower_bound) for item in batch)
            logger.debug(
                "[Personality] 平台历史批次完成 页码=%s 批次数量=%s 最早时间=%s",
                page,
                len(batch),
                self._format_log_datetime(oldest)
                if isinstance(oldest, datetime)
                else oldest,
            )
            if oldest < lower_bound:
                break
            page += 1
        return records

    async def _fetch_onebot_group_history_batch(
        self,
        *,
        bot,
        group_id: str,
        page_size: int,
        message_seq: str | int | None = None,
    ) -> list[dict]:
        """群聊场景优先直接拉 OneBot 历史，避免依赖可能为空的平台消息表。"""
        normalized_group_id = int(group_id) if str(group_id).isdigit() else group_id

        async def _call_history_action(params: dict) -> dict | None:
            if hasattr(bot, "api") and hasattr(bot.api, "call_action"):
                return await bot.api.call_action("get_group_msg_history", **params)
            if hasattr(bot, "call_api"):
                return await bot.call_api("get_group_msg_history", **params)
            if hasattr(bot, "call_action"):
                return await bot.call_action("get_group_msg_history", **params)
            return None

        params = {
            "group_id": normalized_group_id,
            "count": page_size,
            "reverseOrder": True,
        }
        if message_seq is not None:
            params["message_seq"] = (
                int(message_seq) if str(message_seq).isdigit() else message_seq
            )

        try:
            response = await _call_history_action(params)
            if response is None:
                return []
        except Exception as exc:
            # 部分协议端首次空锚点会失败，此时先用 0 号锚点重试一次。
            if not message_seq:
                retry_params = {
                    "group_id": normalized_group_id,
                    "count": page_size,
                    "message_seq": 0,
                }
                logger.info(
                    "[Personality] OneBot历史首次拉取失败，尝试使用 0 号锚点重试 群组=%s 错误=%s",
                    group_id,
                    exc,
                )
                try:
                    retry_response = await _call_history_action(retry_params)
                    if retry_response is not None:
                        response = retry_response
                    else:
                        return []
                except Exception as retry_exc:
                    logger.warning(
                        "[Personality] OneBot历史拉取失败，准备回退平台历史表 群组=%s 消息游标=%s 错误=%s",
                        group_id,
                        message_seq or "",
                        retry_exc,
                    )
                    return []
            else:
                logger.warning(
                    "[Personality] OneBot历史拉取失败，准备回退平台历史表 群组=%s 消息游标=%s 错误=%s",
                    group_id,
                    message_seq or "",
                    exc,
                )
                return []

        if not isinstance(response, dict):
            return []
        response_data = (
            response.get("data") if isinstance(response.get("data"), dict) else {}
        )
        messages = response.get("messages", [])
        if not isinstance(messages, list):
            messages = response_data.get("messages", [])
        if not isinstance(messages, list):
            logger.warning(
                "[Personality] OneBot历史返回结构无法识别，已忽略 群组=%s 消息游标=%s 返回=%s",
                group_id,
                message_seq or "",
                response,
            )
            return []

        return [item for item in messages if isinstance(item, dict)]

    def _convert_onebot_history_record(self, raw_message: dict):
        """把 OneBot 历史消息转成采样器可消费的统一结构。"""
        sender = (
            raw_message.get("sender")
            if isinstance(raw_message.get("sender"), dict)
            else {}
        )
        created_at = datetime.fromtimestamp(
            int(raw_message.get("time", 0) or 0),
            tz=timezone.utc,
        )
        normalized_segments: list[dict] = []
        raw_segments = raw_message.get("message")
        if isinstance(raw_segments, str):
            text = raw_segments.strip()
            if text:
                normalized_segments.append({"type": "Plain", "text": text})
        else:
            for segment in raw_segments or []:
                if not isinstance(segment, dict):
                    continue
                segment_type = str(segment.get("type", "")).lower()
                data = (
                    segment.get("data") if isinstance(segment.get("data"), dict) else {}
                )
                if segment_type == "text":
                    normalized_segments.append(
                        {"type": "Plain", "text": str(data.get("text", "") or "")}
                    )
                    continue
                if segment_type == "reply":
                    normalized_segments.append(
                        {
                            "type": "reply",
                            "data": {
                                "message_id": str(
                                    data.get("message_id") or data.get("id") or ""
                                ).strip()
                            },
                        }
                    )
                    continue
                if segment_type == "at":
                    normalized_segments.append(
                        {
                            "type": "at",
                            "data": {"qq": str(data.get("qq") or "").strip()},
                        }
                    )
                    continue
                normalized_segments.append(segment)

        return SimpleNamespace(
            message_id=str(raw_message.get("message_id") or "").strip(),
            sender_id=str(sender.get("user_id") or "").strip(),
            sender_name=str(sender.get("card") or sender.get("nickname") or "").strip(),
            created_at=created_at,
            content={"type": "user", "message": normalized_segments},
        )

    async def _load_onebot_group_history_within_range(
        self,
        *,
        event: AstrMessageEvent,
        group_id: str,
        lower_bound: datetime,
        page_size: int = 200,
        target_sender_id: str | None = None,
    ) -> list:
        """仅群聊场景尝试直接从 OneBot 拉群历史。"""
        if not str(event.get_group_id() or "").strip():
            return []

        bot = getattr(event, "bot", None)
        if bot is None:
            return []

        records: list = []
        seen_message_ids: set[str] = set()
        message_seq: str | int | None = None
        fetch_round = 1
        normalized_lower_bound = lower_bound.astimezone(timezone.utc)
        while True:
            logger.debug(
                "[Personality] OneBot历史拉取 轮次=%s 群组=%s 每页数量=%s 消息游标=%s",
                fetch_round,
                group_id,
                page_size,
                message_seq or "",
            )
            batch = await self._fetch_onebot_group_history_batch(
                bot=bot,
                group_id=group_id,
                page_size=page_size,
                message_seq=message_seq,
            )
            if not batch:
                logger.debug(
                    "[Personality] OneBot历史拉取为空 轮次=%s 群组=%s",
                    fetch_round,
                    group_id,
                )
                break

            for item in batch:
                message_id = str(item.get("message_id") or "").strip()
                if message_id and message_id in seen_message_ids:
                    continue
                if message_id:
                    seen_message_ids.add(message_id)
                records.append(self._convert_onebot_history_record(item))
            oldest_ts = min(
                int(item.get("time", 0) or 0)
                for item in batch
                if isinstance(item, dict)
            )
            newest_ts = max(
                int(item.get("time", 0) or 0)
                for item in batch
                if isinstance(item, dict)
            )
            first_message = batch[0]
            last_message = batch[-1]
            if int(first_message.get("time", 0) or 0) <= int(
                last_message.get("time", 0) or 0
            ):
                chunk_earliest_message = first_message
            else:
                chunk_earliest_message = last_message

            next_message_seq = (
                chunk_earliest_message.get("message_seq")
                or chunk_earliest_message.get("real_id")
                or chunk_earliest_message.get("seq")
                or chunk_earliest_message.get("message_id")
            )
            logger.debug(
                "[Personality] OneBot历史批次完成 轮次=%s 批次数量=%s 最早时间戳=%s 下一游标=%s",
                fetch_round,
                len(batch),
                oldest_ts,
                next_message_seq or "",
            )
            oldest_dt = datetime.fromtimestamp(oldest_ts, tz=timezone.utc)
            newest_dt = datetime.fromtimestamp(newest_ts, tz=timezone.utc)
            normalized_target_sender_id = str(target_sender_id or "").strip()
            target_hit_count = sum(
                1
                for item in batch
                if (
                    str(
                        (
                            item.get("sender")
                            if isinstance(item.get("sender"), dict)
                            else {}
                        ).get("user_id")
                        or ""
                    ).strip()
                    == normalized_target_sender_id
                )
            )
            logger.info(
                "[Personality] OneBot历史批次诊断 轮次=%s 群组=%s 批次最早消息时间=%s 批次最晚消息时间=%s "
                "分析窗口起点=%s 下一游标=%s 目标用户命中数=%s 首次批次跨度分钟=%.2f",
                fetch_round,
                group_id,
                self._format_log_datetime(oldest_dt),
                self._format_log_datetime(newest_dt),
                self._format_log_datetime(normalized_lower_bound),
                next_message_seq or "",
                target_hit_count,
                max((newest_ts - oldest_ts) / 60, 0),
            )
            if (
                oldest_ts
                and datetime.fromtimestamp(oldest_ts, tz=timezone.utc) < lower_bound
            ):
                logger.info(
                    "[Personality] OneBot历史停止翻页 轮次=%s 群组=%s 原因=已覆盖分析窗口 最早消息时间=%s 分析窗口起点=%s",
                    fetch_round,
                    group_id,
                    self._format_log_datetime(oldest_dt),
                    self._format_log_datetime(normalized_lower_bound),
                )
                break
            if len(batch) < page_size:
                logger.info(
                    "[Personality] OneBot历史批次未满页 轮次=%s 群组=%s 批次数量=%s 页大小=%s 下一游标=%s",
                    fetch_round,
                    group_id,
                    len(batch),
                    page_size,
                    next_message_seq or "",
                )

            if next_message_seq in (None, "", message_seq):
                logger.info(
                    "[Personality] OneBot历史停止翻页 轮次=%s 群组=%s 原因=游标未推进 当前游标=%s 下一游标=%s",
                    fetch_round,
                    group_id,
                    message_seq or "",
                    next_message_seq or "",
                )
                break
            message_seq = next_message_seq
            fetch_round += 1
        return records

    async def _load_analysis_history(
        self,
        *,
        event: AstrMessageEvent,
        group_id: str,
        lower_bound: datetime,
        target_sender_id: str | None = None,
    ) -> tuple[list, str]:
        """群聊先尝试 OneBot 历史；失败或拿不到时再回退平台消息表。"""
        onebot_records = await self._load_onebot_group_history_within_range(
            event=event,
            group_id=group_id,
            lower_bound=lower_bound,
            target_sender_id=target_sender_id,
        )
        if onebot_records:
            return onebot_records, "OneBot接口"
        return (
            await self._load_group_history_within_range(
                platform_id=event.get_platform_id(),
                group_id=group_id,
                lower_bound=lower_bound,
            ),
            "平台历史表",
        )

    def _load_card_template(self) -> str:
        template_path = self.plugin_root / "templates" / "personality_card.html"
        return template_path.read_text(encoding="utf-8")

    @filter.command("persona")
    async def today_personality(self, event: AstrMessageEvent):
        """分析触发者在目标群内的近期文本发言，并返回人格卡片。"""
        event.should_call_llm(True)
        session_type = self._get_session_type(event)
        self._log_analysis_event(
            "收到人格分析命令",
            会话类型=session_type,
            平台=event.get_platform_id(),
            发送者=event.get_sender_id(),
            原始群组=str(event.get_group_id() or "").strip(),
            消息=str(event.get_message_str() or "").strip(),
        )

        try:
            target_group_id, target_sender_id, target_sender_name = (
                self._resolve_analysis_target(event)
            )
        except ValueError as exc:
            self._log_analysis_event(
                "解析分析目标失败",
                会话类型=session_type,
                发送者=event.get_sender_id(),
                原因=str(exc),
            )
            yield event.plain_result(str(exc))
            return

        self._log_analysis_event(
            "已解析分析目标",
            会话类型=session_type,
            平台=event.get_platform_id(),
            目标群组=target_group_id,
            目标发送者=target_sender_id,
            目标昵称=target_sender_name,
        )

        provider_id = self._get_config_value("llm_provider_id", "").strip()
        if not provider_id:
            self._log_analysis_event(
                "人格分析模型未配置",
                会话类型=session_type,
                目标群组=target_group_id,
                目标发送者=target_sender_id,
            )
            yield event.plain_result("请先在插件配置中选择人格分析模型。")
            return

        provider = self.context.get_provider_by_id(provider_id)
        if provider is None:
            self._log_analysis_event(
                "人格分析模型不存在",
                会话类型=session_type,
                模型提供商=provider_id,
                目标群组=target_group_id,
                目标发送者=target_sender_id,
            )
            yield event.plain_result("当前配置的人格分析模型不存在，请检查插件配置。")
            return

        local_now = self._get_analysis_now()
        day_span = max(1, int(self._get_config_value("analysis_day_span", 1)))
        context_window_size = max(
            0, int(self._get_config_value("context_window_size", 3))
        )
        min_msg_threshold = max(1, int(self._get_config_value("min_msg_threshold", 10)))
        lower_bound = self._get_analysis_window_start(local_now, day_span)

        records, history_source = await self._load_analysis_history(
            event=event,
            group_id=target_group_id,
            lower_bound=lower_bound,
            target_sender_id=target_sender_id,
        )
        self._log_analysis_event(
            "历史消息加载完成",
            会话类型=session_type,
            历史来源=history_source,
            目标群组=target_group_id,
            目标发送者=target_sender_id,
            历史条数=len(records),
            起始时间=self._format_log_datetime(lower_bound),
            分析天数=day_span,
        )
        sampled_conversation = sample_interaction_messages(
            records,
            sender_id=target_sender_id,
            local_now=local_now,
            day_span=day_span,
            context_window_size=context_window_size,
        )
        self._log_analysis_event(
            "消息采样完成",
            会话类型=session_type,
            目标群组=target_group_id,
            目标发送者=target_sender_id,
            目标消息数=sampled_conversation.target_message_count,
            采样消息数=sampled_conversation.sampled_message_count,
            上下文窗口=context_window_size,
            最低阈值=min_msg_threshold,
        )
        if sampled_conversation.target_message_count < min_msg_threshold:
            self._log_analysis_event(
                "消息不足无法分析",
                会话类型=session_type,
                目标群组=target_group_id,
                目标发送者=target_sender_id,
                目标消息数=sampled_conversation.target_message_count,
                最低阈值=min_msg_threshold,
                分析天数=day_span,
            )
            yield event.plain_result(
                self._build_insufficient_message(
                    day_span=day_span,
                    min_msg_threshold=min_msg_threshold,
                    target_message_count=sampled_conversation.target_message_count,
                )
            )
            return

        await event.send(event.plain_result("正在采样历史对话，构建人格模型…"))

        bundle = self.prompt_loader.load()
        analyzer = PersonalityAnalyzer(bundle=bundle, provider=provider)
        self._log_analysis_event(
            "人格分析开始",
            会话类型=session_type,
            目标群组=target_group_id,
            目标发送者=target_sender_id,
            采样消息数=sampled_conversation.sampled_message_count,
            模型提供商=provider_id,
        )
        evidence_quote_limit = int(self._get_config_value("evidence_quote_limit", 2))
        normalized = await analyzer.analyze(
            sender_name=target_sender_name,
            messages=sampled_conversation.sampled_messages,
            evidence_item_limit=evidence_quote_limit,
        )
        rarity_stats = await self.result_store.record_personality_occurrence(
            str(normalized.get("personality_name") or "").strip()
        )
        sample_time_window_start, sample_time_window_end = (
            self._build_sample_time_window(sampled_conversation.sampled_messages)
        )
        normalized.update(
            {
                "sender_name": target_sender_name,
                "sender_id": target_sender_id,
                "group_id": target_group_id,
                "platform_id": event.get_platform_id(),
                "analyzed_at": local_now.isoformat(),
                "target_message_count": sampled_conversation.target_message_count,
                "sampled_message_count": sampled_conversation.sampled_message_count,
                "sampled_messages": [
                    item.text for item in sampled_conversation.sampled_messages[:3]
                ],
                "sample_time_window_start": sample_time_window_start,
                "sample_time_window_end": sample_time_window_end,
                "rarity_label": "稀有度",
                "rarity_value": f"TOP {rarity_stats['rarity_top_percent']}%",
                "rarity_ratio": rarity_stats["rarity_ratio"],
                "rarity_top_percent": rarity_stats["rarity_top_percent"],
                "personality_count": rarity_stats["personality_count"],
                "personality_total_count": rarity_stats["total_count"],
                "persona_intro": bundle.persona_intro_map.get(
                    str(normalized.get("personality_name") or "").strip(), ""
                ),
                "persona_intro_map": bundle.persona_intro_map,
            }
        )

        card_data = build_card_view_model(
            normalized,
            evidence_quote_limit=evidence_quote_limit,
        )
        render_payload = {**card_data}
        image_path = await self._render_personality_card_image(render_payload)
        await self.result_store.save_latest(
            event.get_platform_id(),
            target_group_id,
            target_sender_id,
            normalized,
        )
        self._log_analysis_event(
            "人格分析完成",
            会话类型=session_type,
            目标群组=target_group_id,
            目标发送者=target_sender_id,
            人格名称=normalized.get("personality_name", ""),
            采样消息数=sampled_conversation.sampled_message_count,
            目标消息数=sampled_conversation.target_message_count,
            附带文本摘要=self._get_config_value("render_text_summary", True),
        )

        if image_path:
            image_result = await self._send_card_image(event, image_path)
            if image_result is not None:
                yield image_result
        else:
            yield event.plain_result("人格卡片渲染失败，请稍后重试。")
        if self._get_config_value("render_text_summary", True):
            yield event.plain_result(self._build_summary_text(normalized))
