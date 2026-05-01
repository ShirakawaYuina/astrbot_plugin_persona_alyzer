from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class SampledMessage:
    message_id: str
    sender_id: str
    sender_name: str
    text: str
    created_at: datetime
    role: str
    relation: str


@dataclass(frozen=True)
class SampledConversation:
    sampled_messages: list[SampledMessage]
    target_message_count: int
    sampled_message_count: int


def _iter_message_parts(content: dict) -> list[dict]:
    parts = content.get("message", [])
    if not isinstance(parts, list):
        return []
    return [part for part in parts if isinstance(part, dict)]


def _extract_plain_text(content: dict) -> str:
    raw_message = content.get("message")
    if isinstance(raw_message, str):
        return raw_message.strip()

    texts: list[str] = []
    for part in _iter_message_parts(content):
        part_type = str(part.get("type", "")).lower()
        if part_type not in {"plain", "text"}:
            continue
        text = part.get("text")
        if not isinstance(text, str):
            text = (part.get("data") or {}).get("text")
        if isinstance(text, str):
            texts.append(text)
    return "".join(texts).strip()


def _extract_active_mention_ids(content: dict) -> tuple[str, ...]:
    mention_ids: list[str] = []
    for part in _iter_message_parts(content):
        if str(part.get("type", "")).lower() != "at":
            continue
        data = part.get("data") if isinstance(part.get("data"), dict) else {}
        candidate = part.get("qq") or data.get("qq")
        target_id = str(candidate or "").strip()
        if target_id and target_id != "all":
            mention_ids.append(target_id)
    return tuple(mention_ids)


def _extract_reply_message_id(record, content: dict) -> str:
    direct_reply = getattr(record, "reply", None)
    if isinstance(direct_reply, dict):
        reply_id = str(
            direct_reply.get("message_id") or direct_reply.get("id") or ""
        ).strip()
        if reply_id:
            return reply_id

    for part in _iter_message_parts(content):
        if str(part.get("type", "")).lower() != "reply":
            continue
        data = part.get("data") if isinstance(part.get("data"), dict) else {}
        reply_id = str(
            part.get("message_id")
            or part.get("id")
            or data.get("message_id")
            or data.get("id")
            or ""
        ).strip()
        if reply_id:
            return reply_id

    return ""


def _window_start(local_now: datetime, day_span: int) -> datetime:
    today_start = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    return today_start - timedelta(days=max(day_span - 1, 0))


def _record_sort_key(record, fallback: datetime) -> datetime:
    created_at = getattr(record, "created_at", fallback)
    if isinstance(created_at, datetime):
        return created_at
    return fallback


def _record_key(record, text: str) -> str:
    message_id = str(getattr(record, "message_id", "") or "").strip()
    if message_id:
        return f"id:{message_id}"

    sender_id = str(getattr(record, "sender_id", "") or "").strip()
    created_at = getattr(record, "created_at", None)
    created_text = created_at.isoformat() if isinstance(created_at, datetime) else ""
    return f"fallback:{sender_id}:{created_text}:{text}"


def _relation_priority(relation: str) -> int:
    if relation in {"reply_target", "mention_target"}:
        return 2
    if relation == "window_context":
        return 1
    return 0


def _resolve_mentioned_record(
    records: list, mention_ids: tuple[str, ...], sender_id: str
):
    if not mention_ids:
        return None

    target_ids = {item for item in mention_ids if item != str(sender_id)}
    if not target_ids:
        return None

    for record in reversed(records):
        if str(getattr(record, "sender_id", "")).strip() in target_ids:
            return record
    return None


def sample_interaction_messages(
    records: list,
    *,
    sender_id: str,
    local_now: datetime,
    day_span: int,
    context_window_size: int,
) -> SampledConversation:
    """按自然日范围抽取目标用户消息，并优先保留直接互动上下文。"""
    lower_bound = _window_start(local_now, day_span)
    in_range = [
        record
        for record in sorted(
            records, key=lambda item: _record_sort_key(item, lower_bound)
        )
        if _record_sort_key(record, lower_bound) >= lower_bound
    ]
    record_by_id = {
        str(getattr(record, "message_id", "") or "").strip(): record
        for record in in_range
        if str(getattr(record, "message_id", "") or "").strip()
    }
    sampled_messages: list[SampledMessage] = []
    sampled_index: dict[str, int] = {}
    normalized_sender_id = str(sender_id)
    target_message_count = 0

    def push(record, *, role: str, relation: str) -> None:
        content = getattr(record, "content", {}) or {}
        text = _extract_plain_text(content)
        if not text or text.startswith("/"):
            return

        item = SampledMessage(
            message_id=str(getattr(record, "message_id", "") or "").strip(),
            sender_id=str(getattr(record, "sender_id", "") or "").strip(),
            sender_name=str(getattr(record, "sender_name", "") or "").strip()
            or str(getattr(record, "sender_id", "") or "").strip(),
            text=text,
            created_at=_record_sort_key(record, lower_bound),
            role=role,
            relation=relation,
        )
        key = _record_key(record, text)
        existing_index = sampled_index.get(key)
        if existing_index is not None:
            existing = sampled_messages[existing_index]
            # 同一条消息被窗口命中后，如果又被识别为主动互动目标，需要升级关系语义。
            if _relation_priority(relation) > _relation_priority(existing.relation):
                sampled_messages[existing_index] = item
            return

        sampled_index[key] = len(sampled_messages)
        sampled_messages.append(item)

    for index, record in enumerate(in_range):
        if str(getattr(record, "sender_id", "")).strip() != normalized_sender_id:
            continue

        content = getattr(record, "content", {}) or {}
        target_text = _extract_plain_text(content)
        if not target_text or target_text.startswith("/"):
            continue

        target_message_count += 1

        reply_message_id = _extract_reply_message_id(record, content)
        if reply_message_id and reply_message_id in record_by_id:
            push(
                record_by_id[reply_message_id], role="context", relation="reply_target"
            )
            push(record, role="target", relation="self")
            continue

        mentioned_record = _resolve_mentioned_record(
            in_range[:index],
            _extract_active_mention_ids(content),
            normalized_sender_id,
        )
        if mentioned_record is not None:
            push(mentioned_record, role="context", relation="mention_target")
            push(record, role="target", relation="self")
            continue

        window_candidates = [
            item
            for item in in_range[max(0, index - max(context_window_size, 0)) : index]
            if str(getattr(item, "sender_id", "")).strip() != normalized_sender_id
        ]
        for candidate in window_candidates:
            push(candidate, role="context", relation="window_context")

        push(record, role="target", relation="self")

    return SampledConversation(
        sampled_messages=sampled_messages,
        target_message_count=target_message_count,
        sampled_message_count=len(sampled_messages),
    )


def sample_today_messages(
    records: list, *, sender_id: str, local_now: datetime
) -> list[str]:
    """兼容旧调用方，继续返回目标用户当天的纯文本发言。"""
    sampled = sample_interaction_messages(
        records,
        sender_id=sender_id,
        local_now=local_now,
        day_span=1,
        context_window_size=0,
    )
    return [item.text for item in sampled.sampled_messages if item.role == "target"]
