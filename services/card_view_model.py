from __future__ import annotations

import base64
import colorsys
import json
import mimetypes
import random
import re
from datetime import datetime
from functools import lru_cache
from math import cos, pi, sin
from pathlib import Path
from zoneinfo import ZoneInfo

from PIL import Image

from astrbot.core.utils.image_ref_utils import resolve_file_url_path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
PERSONA_IMAGE_DIR = PLUGIN_ROOT / "persona_imgs"
DISPLAY_DATA_PATH = PLUGIN_ROOT / "persona_display_data.json"
PROMPT_DIR = PLUGIN_ROOT / "prompts"
IMAGE_MIME_FALLBACK = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".svg": "image/svg+xml",
}
PERSONA_KEY_ALIASES = {
    "the neighbor": "the good neighbor",
}
PERSONA_NAME_DISPLAY_OVERRIDES = {
    "好邻居": "The Neighbor",
}
PERSONA_GROUP_PATTERN = re.compile(
    r"^##\s+\d+\.\s+(?P<group_cn>[^(\n]+?)\s+\((?P<group_en>[^)]+)\)",
    re.MULTILINE,
)
PERSONA_ENTRY_PATTERN = re.compile(
    r"^- \*\*(?P<cn>[^(\n]+?)\s+\((?P<en>[^/)\n]+)\s*/\s*(?P<alignment>[^)\n]+)\)\*\*",
    re.MULTILINE,
)

DIMENSION_ORDER = (
    "驱动极性",
    "逻辑载荷",
    "社交熵值",
    "变革烈度",
    "共情阈值",
)

DIMENSION_META = {
    "驱动极性": {
        "meaning": "衡量个体行为是受内在理想驱动，还是受外部现实/规则驱动。",
        "high_label": "纯粹理想",
        "low_label": "绝对现实",
    },
    "逻辑载荷": {
        "meaning": "衡量言论中理性分析、数据支撑与感性表达、直觉判断的比例。",
        "high_label": "严丝合缝",
        "low_label": "全凭感觉",
    },
    "社交熵值": {
        "meaning": "衡量在群体中的能量散发程度。高分为制造波动，低分为维持静默。",
        "high_label": "能量中心",
        "low_label": "隐形屏障",
    },
    "变革烈度": {
        "meaning": "衡量对既有秩序的破坏欲望或重构倾向。",
        "high_label": "颠覆一切",
        "low_label": "固若金汤",
    },
    "共情阈值": {
        "meaning": "衡量吸收并处理他人情绪的能力与意愿。",
        "high_label": "情感黑洞",
        "low_label": "理性冰山",
    },
}

THEME_MAP = {
    "positive": {
        "background": "#F4F3F1",
        "accent": "#B58B53",
        "accent_soft": "#D8C2A4",
        "text_primary": "#191919",
        "text_secondary": "#6F6A64",
        "panel": "rgba(255,255,255,0.68)",
        "line": "rgba(27,27,27,0.08)",
        "radar_fill": "rgba(181,139,83,0.20)",
    },
    "neutral": {
        "background": "#EEF1F4",
        "accent": "#70879A",
        "accent_soft": "#B7C4CF",
        "text_primary": "#171B1E",
        "text_secondary": "#67727B",
        "panel": "rgba(255,255,255,0.64)",
        "line": "rgba(23,27,30,0.08)",
        "radar_fill": "rgba(112,135,154,0.20)",
    },
    "negative": {
        "background": "#F2F1EF",
        "accent": "#8E5361",
        "accent_soft": "#C4A7AF",
        "text_primary": "#161416",
        "text_secondary": "#6C6366",
        "panel": "rgba(255,255,255,0.62)",
        "line": "rgba(22,20,22,0.08)",
        "radar_fill": "rgba(142,83,97,0.18)",
    },
}

TAGLINE_MAP = {
    "导师": "把复杂问题讲明白",
    "学者": "让事实自己说话",
    "独裁家": "正确比温度更重要",
    "幼稚鬼": "任性也是求救信号",
}

LEXICON_PATH = PROMPT_DIR / "人格提示词.md"


def _build_radar_points(dimensions: dict[str, int]) -> str:
    """将五维分数转换为 SVG 多边形点位，便于模板直接渲染雷达图。"""
    center_x = 160
    center_y = 160
    radius = 102
    points: list[str] = []
    total = len(DIMENSION_ORDER)
    for index, key in enumerate(DIMENSION_ORDER):
        score = max(0, min(int(dimensions.get(key, 0)), 100))
        ratio = score / 100
        angle = (2 * pi * index / total) - (pi / 2)
        x = center_x + cos(angle) * radius * ratio
        y = center_y + sin(angle) * radius * ratio
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def _to_file_url(value: str | None) -> str:
    """将本地路径统一转为 file URL，方便模板稳定加载本地图片。"""
    if not value:
        return ""
    if value.startswith(("file://", "http://", "https://", "data:")):
        return value
    return Path(value).resolve().as_uri()


def _clamp_channel(value: float) -> int:
    return max(0, min(int(round(value)), 255))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def _mix_rgb(
    left: tuple[int, int, int], right: tuple[int, int, int], ratio: float
) -> tuple[int, int, int]:
    normalized_ratio = max(0.0, min(ratio, 1.0))
    return tuple(
        _clamp_channel(channel * (1 - normalized_ratio) + target * normalized_ratio)
        for channel, target in zip(left, right, strict=True)
    )


def _rgb_to_rgba_string(rgb: tuple[int, int, int], alpha: float) -> str:
    red, green, blue = rgb
    normalized_alpha = max(0.0, min(alpha, 1.0))
    return f"rgba({red},{green},{blue},{normalized_alpha:.2f})"


def _normalize_accent_color(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    red, green, blue = rgb
    hue, lightness, saturation = colorsys.rgb_to_hls(
        red / 255,
        green / 255,
        blue / 255,
    )
    normalized_lightness = min(max(lightness, 0.34), 0.58)
    normalized_saturation = min(max(saturation, 0.38), 0.72)
    normalized = colorsys.hls_to_rgb(
        hue,
        normalized_lightness,
        normalized_saturation,
    )
    return tuple(_clamp_channel(channel * 255) for channel in normalized)


def _collect_candidate_pixels(path: Path) -> list[tuple[int, int, int]]:
    with Image.open(path) as image:
        rgba_image = image.convert("RGBA")
        rgba_image.thumbnail((64, 64))
        pixels: list[tuple[int, int, int]] = []
        pixel_access = rgba_image.load()
        if pixel_access is None:
            return pixels
        width, height = rgba_image.size
        for y_axis in range(height):
            for x_axis in range(width):
                red, green, blue, alpha = pixel_access[x_axis, y_axis]
                if alpha < 24:
                    continue
                _, lightness, saturation = colorsys.rgb_to_hls(
                    red / 255,
                    green / 255,
                    blue / 255,
                )
                if lightness < 0.12 or lightness > 0.88:
                    continue
                if saturation < 0.15:
                    continue
                pixels.append((red, green, blue))
        return pixels


def _pick_dominant_accent(path: Path) -> tuple[int, int, int] | None:
    pixels = _collect_candidate_pixels(path)
    if not pixels:
        return None

    buckets: dict[tuple[int, int, int], int] = {}
    for red, green, blue in pixels:
        bucket = (
            (red // 24) * 24,
            (green // 24) * 24,
            (blue // 24) * 24,
        )
        buckets[bucket] = buckets.get(bucket, 0) + 1

    ranked = sorted(buckets.items(), key=lambda item: item[1], reverse=True)
    dominant_rgb, _ = ranked[0]
    return dominant_rgb


def _resolve_local_image_path(value: str | None) -> Path | None:
    if not value:
        return None
    lowered = value.lower()
    if lowered.startswith(("http://", "https://", "data:")):
        return None
    local_path = (
        resolve_file_url_path(value) if lowered.startswith("file://") else value
    )
    path = Path(local_path).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    return path if path.exists() else None


@lru_cache(maxsize=128)
def _build_dynamic_theme_from_image(path_str: str, alignment: str) -> dict | None:
    path = Path(path_str)
    if not path.exists():
        return None
    dominant_rgb = _pick_dominant_accent(path)
    if dominant_rgb is None:
        return None

    accent_rgb = _normalize_accent_color(dominant_rgb)
    accent_soft_rgb = _mix_rgb(accent_rgb, (255, 255, 255), 0.48)
    background_rgb = _mix_rgb(accent_rgb, (255, 255, 255), 0.90)
    line_rgb = _mix_rgb(accent_rgb, (32, 32, 32), 0.20)
    base_theme = dict(THEME_MAP.get(alignment, THEME_MAP["neutral"]))
    base_theme.update(
        {
            "background": _rgb_to_hex(background_rgb),
            "accent": _rgb_to_hex(accent_rgb),
            "accent_soft": _rgb_to_hex(accent_soft_rgb),
            "line": _rgb_to_rgba_string(line_rgb, 0.12),
            "radar_fill": _rgb_to_rgba_string(accent_rgb, 0.20),
        }
    )
    return base_theme


@lru_cache(maxsize=128)
def _to_data_image_url(value: str) -> str:
    """将本地人格图转成 data URL，避免远端 t2i 容器无法访问宿主文件。"""
    lowered = value.lower()
    if lowered.startswith(("http://", "https://", "data:")):
        return value

    local_path = (
        resolve_file_url_path(value) if lowered.startswith("file://") else value
    )
    path = Path(local_path).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    if not path.exists():
        return _to_file_url(value)
    mime_type = mimetypes.guess_type(path.name)[0] or IMAGE_MIME_FALLBACK.get(
        path.suffix.lower(),
        "application/octet-stream",
    )
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _resolve_user_avatar(result: dict) -> str:
    """优先使用显式头像；QQ 号恒为纯数字，因此可直接回退到 qlogo 头像服务。"""
    explicit_avatar = result.get("user_avatar")
    if explicit_avatar:
        return _to_file_url(explicit_avatar)

    sender_id = str(result.get("sender_id") or "").strip()
    if sender_id.isdigit():
        return f"https://q1.qlogo.cn/g?b=qq&nk={sender_id}&s=100"
    return ""


def _normalize_persona_key(value: str) -> str:
    """将人格英文名和文件名统一归一，避免大小写、编号和符号差异导致匹配失败。"""
    normalized = value.strip().lower().replace("_", " ")
    normalized = re.sub(r"^\d+\s*[_-]?\s*", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return PERSONA_KEY_ALIASES.get(normalized, normalized)


def _format_english_display_name(value: str) -> str:
    """统一英文展示名的大小写与历史别名，避免显示层和资源层分裂。"""
    normalized = _normalize_persona_key(value)
    if not normalized:
        return ""
    return normalized.title()


def _build_persona_image_map() -> dict[str, Path]:
    """根据 persona_imgs 目录中的英文文件名自动建立图片索引，减少手写映射维护成本。"""
    image_map: dict[str, Path] = {}
    for path in PERSONA_IMAGE_DIR.iterdir():
        if not path.is_file():
            continue
        normalized_stem = _normalize_persona_key(path.stem)
        image_map[normalized_stem] = path
    return image_map


def _build_persona_image_aliases() -> dict[str, str]:
    """从人格词库中提取中文人格名与英文人格名的对应关系，避免重复维护 36 条别名。"""
    alias_map: dict[str, str] = {}
    if not LEXICON_PATH.exists():
        return alias_map
    content = LEXICON_PATH.read_text(encoding="utf-8")
    for match in PERSONA_ENTRY_PATTERN.finditer(content):
        chinese_name = match.group("cn").strip()
        english_name = _normalize_persona_key(match.group("en"))
        if chinese_name and english_name:
            alias_map[chinese_name] = english_name
    return alias_map


def _build_persona_english_name_map() -> dict[str, str]:
    """优先使用词库中的英文人格名，必要时叠加显示层兼容别名。"""
    english_name_map: dict[str, str] = {}
    if not LEXICON_PATH.exists():
        return english_name_map
    content = LEXICON_PATH.read_text(encoding="utf-8")
    for match in PERSONA_ENTRY_PATTERN.finditer(content):
        personality_name = match.group("cn").strip()
        raw_english_name = match.group("en").strip()
        if not personality_name:
            continue
        english_name_map[personality_name] = PERSONA_NAME_DISPLAY_OVERRIDES.get(
            personality_name,
            _format_english_display_name(raw_english_name),
        )
    return english_name_map


def _build_persona_archetype_display_map() -> dict[str, str]:
    """按词库为每种人格补齐“中文 · English”原型展示文案。"""
    archetype_display_map: dict[str, str] = {}
    if not LEXICON_PATH.exists():
        return archetype_display_map
    content = LEXICON_PATH.read_text(encoding="utf-8")
    group_matches = list(PERSONA_GROUP_PATTERN.finditer(content))
    for index, group_match in enumerate(group_matches):
        section_start = group_match.end()
        section_end = (
            group_matches[index + 1].start()
            if index + 1 < len(group_matches)
            else len(content)
        )
        section_text = content[section_start:section_end]
        group_display = (
            f"{group_match.group('group_cn').strip()} · "
            f"{_format_english_display_name(group_match.group('group_en').strip())}"
        )
        for entry_match in PERSONA_ENTRY_PATTERN.finditer(section_text):
            personality_name = entry_match.group("cn").strip()
            if personality_name:
                archetype_display_map[personality_name] = group_display
    return archetype_display_map


def _build_persona_intro_map() -> dict[str, str]:
    """从人格词库中提取每种人格的介绍文案，作为卡片主文介绍的唯一来源。"""
    intro_map: dict[str, str] = {}
    if not LEXICON_PATH.exists():
        return intro_map
    content = LEXICON_PATH.read_text(encoding="utf-8")
    pattern = re.compile(
        r"- \*\*(?P<cn>[^（(]+?)\s+\((?P<en>[^/／)]+).*?\)\*\*：.*?"
        r"\n  - \*特征\*：(?P<traits>.*?)"
        r"\n  - \*介绍\*：(?P<intro>.*?)(?=\n\n- \*\*|\n---|\Z)",
        re.S,
    )
    for match in pattern.finditer(content):
        chinese_name = match.group("cn").strip()
        intro = re.sub(r"\s+", "", match.group("intro")).strip()
        if chinese_name and intro:
            intro_map[chinese_name] = intro
    return intro_map


def _format_archetype_group(value: str, personality_name: str) -> str:
    """优先使用词库映射，兜底兼容旧的“中文(English)”原型写法。"""
    mapped_value = PERSONA_ARCHETYPE_DISPLAY_MAP.get(personality_name)
    if mapped_value:
        return mapped_value

    normalized = value.strip()
    if not normalized:
        return "未归类原型"

    match = re.match(
        r"^(?P<group_cn>[^（(]+?)\s*[（(]\s*(?P<group_en>[^)）]+?)\s*[)）]\s*$",
        normalized,
    )
    if not match:
        return normalized
    return (
        f"{match.group('group_cn').strip()} · "
        f"{_format_english_display_name(match.group('group_en').strip())}"
    )


def _format_personality_name_en(personality_name: str, value: str) -> str:
    """展示层统一使用首字母大写英文名，并兼容历史命名。"""
    mapped_value = PERSONA_ENGLISH_NAME_MAP.get(personality_name)
    if mapped_value:
        return mapped_value
    return _format_english_display_name(value) or "Unknown Archetype"


def _load_persona_display_data() -> dict:
    """读取展示层专用数据，当前用于人格关键词池的独立维护。"""
    if not DISPLAY_DATA_PATH.exists():
        return {"default_keywords": [], "keyword_pools": {}}
    return json.loads(DISPLAY_DATA_PATH.read_text(encoding="utf-8"))


def _split_intro_paragraphs(intro: str) -> list[str]:
    """将长介绍按 1-2 句切分为阅读友好的短段，便于卡片排版。"""
    sentences = [
        f"{item.strip()}。"
        for item in intro.replace("\n", "").split("。")
        if item.strip()
    ]
    paragraphs: list[str] = []
    chunk: list[str] = []
    for sentence in sentences:
        chunk.append(sentence)
        if len(chunk) == 2:
            paragraphs.append("".join(chunk))
            chunk = []
    if chunk:
        paragraphs.append("".join(chunk))
    return paragraphs or [intro]


def _pick_persona_keywords(personality_name: str) -> list[str]:
    """为人格卡片底部随机抽取短关键词，保证每次预览和实际渲染都更有活气。"""
    keyword_pools = PERSONA_DISPLAY_DATA.get("keyword_pools", {})
    default_keywords = PERSONA_DISPLAY_DATA.get("default_keywords", [])
    pool = list(keyword_pools.get(personality_name, default_keywords))
    valid_pool = [item for item in pool if 2 <= len(item) <= 4]
    if len(valid_pool) < 3:
        valid_pool = [item for item in default_keywords if 2 <= len(item) <= 4]
    if not valid_pool:
        return []
    count = min(len(valid_pool), random.randint(3, 5))
    return random.sample(valid_pool, count)


PERSONA_IMAGE_MAP = _build_persona_image_map()
PERSONA_IMAGE_ALIASES = _build_persona_image_aliases()
PERSONA_ENGLISH_NAME_MAP = _build_persona_english_name_map()
PERSONA_ARCHETYPE_DISPLAY_MAP = _build_persona_archetype_display_map()
PERSONA_INTRO_MAP = _build_persona_intro_map()
PERSONA_DISPLAY_DATA = _load_persona_display_data()
MATCHING_RING_RADIUS = 46
MATCHING_RING_CIRCUMFERENCE = 2 * pi * MATCHING_RING_RADIUS
CST_TIMEZONE = ZoneInfo("Asia/Shanghai")


def _resolve_persona_image(result: dict) -> str:
    """优先使用上游显式传入的人格图；未传入时再根据人格结果映射本地资源。"""
    explicit_image = result.get("persona_image")
    if explicit_image:
        return _to_data_image_url(str(explicit_image))

    candidates = (
        PERSONA_IMAGE_ALIASES.get(
            str(result.get("personality_name") or "").strip(), ""
        ),
        str(result.get("personality_name_en") or "").strip().lower(),
        str(result.get("personality_id") or "").strip(),
    )
    for key in candidates:
        if not key:
            continue
        image_path = PERSONA_IMAGE_MAP.get(_normalize_persona_key(key))
        if image_path and image_path.exists():
            return _to_data_image_url(str(image_path.resolve()))
    return ""


def _resolve_persona_image_path(result: dict) -> Path | None:
    explicit_image = result.get("persona_image")
    if explicit_image:
        return _resolve_local_image_path(str(explicit_image).strip())

    candidates = (
        PERSONA_IMAGE_ALIASES.get(
            str(result.get("personality_name") or "").strip(), ""
        ),
        str(result.get("personality_name_en") or "").strip().lower(),
        str(result.get("personality_id") or "").strip(),
    )
    for key in candidates:
        if not key:
            continue
        image_path = PERSONA_IMAGE_MAP.get(_normalize_persona_key(key))
        if image_path and image_path.exists():
            return image_path.resolve()
    return None


def _format_analyzed_at(value) -> str:
    if not value:
        return ""

    if isinstance(value, datetime):
        analyzed_at = value
    else:
        try:
            analyzed_at = datetime.fromisoformat(str(value).strip())
        except ValueError:
            return str(value)

    if analyzed_at.tzinfo is None:
        analyzed_at = analyzed_at.replace(tzinfo=CST_TIMEZONE)
    analyzed_at = analyzed_at.astimezone(CST_TIMEZONE)
    return analyzed_at.strftime("%Y-%m-%d %H:%M:%S CST")


def _format_time_window_point(value) -> str:
    if not value:
        return ""

    if isinstance(value, datetime):
        point = value
    else:
        try:
            point = datetime.fromisoformat(str(value).strip())
        except ValueError:
            return ""

    if point.tzinfo is None:
        point = point.replace(tzinfo=CST_TIMEZONE)
    point = point.astimezone(CST_TIMEZONE)
    return point.strftime("%H：%M")


def _format_time_window_date(value) -> str:
    if not value:
        return ""

    if isinstance(value, datetime):
        point = value
    else:
        try:
            point = datetime.fromisoformat(str(value).strip())
        except ValueError:
            return ""

    if point.tzinfo is None:
        point = point.replace(tzinfo=CST_TIMEZONE)
    point = point.astimezone(CST_TIMEZONE)
    return point.strftime("%m-%d")


def _format_time_window_label(start_value, end_value) -> str:
    start_text = _format_time_window_point(start_value)
    end_text = _format_time_window_point(end_value)
    if not start_text or not end_text:
        return ""
    return f"{start_text} - {end_text}"


def build_card_view_model(result: dict, evidence_quote_limit: int = 2) -> dict:
    """将分析结果映射为卡片模板数据，避免模板直接依赖原始 LLM 输出。"""
    sanitized_result = {k: v for k, v in result.items() if k != "evidence_quotes"}
    alignment = result.get("alignment", "neutral")
    persona_image_path = _resolve_persona_image_path(result)
    theme = (
        _build_dynamic_theme_from_image(str(persona_image_path), alignment)
        if persona_image_path is not None
        else None
    ) or dict(THEME_MAP.get(alignment, THEME_MAP["neutral"]))
    dimensions = result.get("dimensions", {})
    sender_name = result.get("sender_name") or "匿名群友"
    personality_name = result.get("personality_name") or "未命名人格"
    personality_name_en = _format_personality_name_en(
        personality_name,
        str(result.get("personality_name_en") or "unknown archetype"),
    )
    archetype_group = _format_archetype_group(
        str(result.get("archetype_group") or "").strip(),
        personality_name,
    )
    matching_rate = result.get("matching_rate", 0)
    if isinstance(matching_rate, float) and 0 <= matching_rate <= 1:
        matching_rate = int(round(matching_rate * 100))
    else:
        matching_rate = int(matching_rate or 0)
    evidence_items: list[dict[str, str]] = []
    raw_evidence_items = result.get("evidence_items")
    if not isinstance(raw_evidence_items, list):
        raw_evidence_items = []
    for raw_item in raw_evidence_items:
        if not isinstance(raw_item, dict):
            continue
        quote = str(raw_item.get("quote") or "").strip()
        analysis = str(raw_item.get("analysis") or "").strip()
        if not quote or not analysis:
            continue
        evidence_items.append({"quote": quote, "analysis": analysis})
    evidence_items = evidence_items[: max(0, evidence_quote_limit)]
    persona_intro = (
        str(result.get("persona_intro") or "").strip()
        or PERSONA_INTRO_MAP.get(personality_name, "")
        or "此人格暂未补充详细介绍。"
    )
    return {
        **sanitized_result,
        "sender_name": sender_name,
        "personality_name": personality_name,
        "personality_name_en": personality_name_en,
        "archetype_group": archetype_group,
        "matching_rate": matching_rate,
        "matching_ring_radius": MATCHING_RING_RADIUS,
        "matching_ring_circumference": f"{MATCHING_RING_CIRCUMFERENCE:.2f}",
        "matching_ring_dashoffset": (
            f"{MATCHING_RING_CIRCUMFERENCE * (1 - matching_rate / 100):.2f}"
        ),
        "rarity_label": result.get("rarity_label") or "稀有度",
        "rarity_value": result.get("rarity_value") or "TOP --",
        "theme": theme,
        "tagline": result.get("tagline")
        or TAGLINE_MAP.get(personality_name, "从今天的发言气质中看见你"),
        "radar_points": result.get("radar_points") or _build_radar_points(dimensions),
        "user_avatar": _resolve_user_avatar(result),
        "persona_image": _resolve_persona_image(result),
        "persona_quote": result.get("persona_quote")
        or "情绪先一步出门，理智总在后面追。",
        "persona_intro": persona_intro,
        "persona_intro_paragraphs": _split_intro_paragraphs(persona_intro),
        "persona_keywords": result.get("persona_keywords")
        or _pick_persona_keywords(personality_name),
        "description": result.get("description")
        or "今天的表达中能感受到明显的情绪先行与安全感索取，呈现出带刺又脆弱的防御姿态。",
        "intensity_tag": result.get("intensity_tag") or "情绪外放",
        "evidence_items": evidence_items,
        "dimension_items": [
            {
                "name": key,
                "value": int(dimensions.get(key, 0)),
                "meaning": DIMENSION_META[key]["meaning"],
                "high_label": DIMENSION_META[key]["high_label"],
                "low_label": DIMENSION_META[key]["low_label"],
            }
            for key in DIMENSION_ORDER
        ],
        "summary_lines": [
            f"今日人格：{personality_name}",
            f"原型大类：{archetype_group}",
            f"匹配度：{matching_rate}%",
        ],
        "time_window_label": _format_time_window_label(
            result.get("sample_time_window_start"),
            result.get("sample_time_window_end"),
        ),
        "time_window_start_label": _format_time_window_point(
            result.get("sample_time_window_start")
        ),
        "time_window_end_label": _format_time_window_point(
            result.get("sample_time_window_end")
        ),
        "time_window_start_date_label": _format_time_window_date(
            result.get("sample_time_window_start")
        ),
        "time_window_end_date_label": _format_time_window_date(
            result.get("sample_time_window_end")
        ),
        "analyzed_at": _format_analyzed_at(result.get("analyzed_at")),
    }
