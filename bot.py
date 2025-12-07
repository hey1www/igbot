import json
import logging
import os
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from dotenv import load_dotenv
from instagrapi import Client
from instagrapi.exceptions import LoginRequired
from instagrapi.mixins.direct import DirectMessage

# 路径与文件常量
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROMPTS_DIR = BASE_DIR / "prompts"
SESSION_FILE = DATA_DIR / "session.json"
STATE_FILE = DATA_DIR / "state.json"
HISTORY_DIR = DATA_DIR / "history"
SUMMARY_DIR = DATA_DIR / "summary"
SYSTEM_PROMPT_FILE = PROMPTS_DIR / "system_prompt.txt"
SUMMARY_BATCH_SIZE = 50
MAX_HISTORY_KEEP = 1000

DEFAULT_SYSTEM_PROMPT = (
    "你是一个 Instagram 群聊里的成员，昵称叫 heyi bot，@hey1bot 就是在召唤你。\n"
    "你根据最近的聊天内容做出自然、有趣、简短的回复，像一个真实的群友，不要长篇大论。\n"
    "聊天记录按时间从旧到新排列，后面的消息更近。最近的消息以及上次被 @ 后的消息更重要，请优先回应最新的 @ 和其后的对话。\n"
    "聊天历史中的每一行格式为“时间 | 用户名: 内容”，请严格按照用户名判断说话人，不要混淆。"
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)


CONFIG_MENU_TEXT = """[heyi bot config · current chat]

Reply mode (reply mode)
  quiet       – never auto-reply (only system/config messages)
  @andreply   – reply when I’m @mentioned or someone replies directly to my message
  mention     – reply when I’m @mentioned, replied to, or people say “bot” / “ai” / “robot”
  all         – reply to every non-command message

Commands for reply mode:
  /mode quiet
  /mode @andreply
  /mode mention
  /mode all


Memory & history (this chat only)
  /mem trim          – drop the oldest 40 messages from the last 50
  /mem clear_recent  – clear all recent short-term memory (two-step confirm)
  /mem toggle_long   – turn long-term memory ON / OFF in replies
  /mem toggle_fish   – toggle “goldfish mode” (only see the last 10 messages)
  /mem clear_all     – delete ALL memory for this chat (short + long-term, two-step confirm)

Summary tools
  /summary last      – show the latest long-term summary for this chat
  /summary 50        – summarise the last 50 messages into a few lines
  /summary 1h        – summarise roughly the last 1 hour of conversation

Style settings
  /maxlen N          – set maximum reply length (e.g. /maxlen 300)
  /temp T            – set temperature 0.1–1.0 (higher = more playful, lower = calmer)

Config input helper
  In config commands, underscores "_" in arguments should be treated as spaces.
  For example, "hello_world" will be interpreted as "hello world".

Exit config
  /exit              – leave config mode and go back to normal behaviour

While this config menu is active I will poll every 1s.
After /exit I will return to the normal random polling interval.
""".strip()

REPLY_KEYWORDS = ["bot", "ai", "机器人", "人工智能"]


def normalize_arg_text(s: str) -> str:
    return s.replace("_", " ").strip()


def ensure_thread_state_defaults(thread_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backfill newly added thread state fields for compatibility.
    """
    thread_state = dict(thread_state) if thread_state else {}
    thread_state.setdefault("last_message_id", None)
    thread_state.setdefault("last_mention_id", None)
    thread_state.setdefault("last_summary_index", 0)
    thread_state.setdefault("total_messages_seen", 0)
    thread_state.setdefault("config_mode", False)
    thread_state.setdefault("reply_mode", "@andreply")
    thread_state.setdefault("long_memory_enabled", True)
    thread_state.setdefault("fish_mode", False)
    thread_state.setdefault("pending_memory_action", None)
    thread_state.setdefault("pending_memory_user_id", None)
    thread_state.setdefault("max_reply_len", 5120)
    thread_state.setdefault("temperature", 0.6)
    return thread_state


def _safe_int(value: Any, default: int = 0) -> int:
    """Best-effort int conversion with fallback."""
    try:
        return int(value)
    except Exception:
        return default


def ensure_history_entry_defaults(entry: Dict[str, Any]) -> None:
    """
    Backfill newly added history fields for compatibility with旧记录.
    """
    if entry is None:
        return
    if entry.get("id") is not None:
        entry["id"] = str(entry.get("id"))
    entry.setdefault("timestamp", "")
    entry.setdefault("user_id", None)
    entry.setdefault("username", None)
    entry.setdefault("item_type", "text")
    entry.setdefault("text", entry.get("text") or "")
    entry.setdefault("caption", None)
    entry.setdefault("media_type", None)

    # URL 字段兜底：确保为 str 或 None
    media_url = entry.get("media_url")
    if media_url is not None and not isinstance(media_url, str):
        try:
            entry["media_url"] = str(media_url)
        except Exception:
            entry["media_url"] = None

    link_url = entry.get("link_url")
    if link_url is not None and not isinstance(link_url, str):
        try:
            entry["link_url"] = str(link_url)
        except Exception:
            entry["link_url"] = None

    link_meta = entry.get("link_meta")
    if isinstance(link_meta, dict):
        for k, v in list(link_meta.items()):
            if v is not None and not isinstance(v, str):
                try:
                    link_meta[k] = str(v)
                except Exception:
                    link_meta[k] = None

    entry.setdefault("media_url", None)
    entry.setdefault("media_vl_summary", None)
    entry.setdefault("media_vl_tags", None)
    entry.setdefault("link_url", None)
    entry.setdefault("link_meta", entry.get("link_meta"))
    entry.setdefault("reply_to", entry.get("reply_to"))


def parse_timestamp(ts_str: str) -> datetime | None:
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except Exception:
        try:
            return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None


def load_state() -> Dict[str, Dict[str, Any]]:
    """
    从 state.json 读出每个 thread 的状态（兼容旧格式的 last_message_id）。
    """
    if not STATE_FILE.exists():
        logging.info("STATE_FILE 不存在，初始化为空状态")
        return {}
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logging.warning("读取 STATE_FILE 出错，将忽略：%s", exc)
        return {}

    if not isinstance(data, dict):
        logging.warning("STATE_FILE 内容不是字典，将忽略")
        return {}

    state: Dict[str, Dict[str, Any]] = {}
    for tid, raw in data.items():
        thread_state: Dict[str, Any] = {}
        if isinstance(raw, dict):
            last_msg = raw.get("last_message_id")
            last_mention = raw.get("last_mention_id")
            last_summary_index = _safe_int(raw.get("last_summary_index"), 0)
            total_seen = _safe_int(raw.get("total_messages_seen"), 0)
        else:
            last_msg = raw
            last_mention = None
            last_summary_index = 0
            total_seen = 0
        if last_msg is not None:
            thread_state["last_message_id"] = str(last_msg)
        if last_mention is not None:
            thread_state["last_mention_id"] = str(last_mention)
        thread_state["last_summary_index"] = last_summary_index
        thread_state["total_messages_seen"] = total_seen
        thread_state = ensure_thread_state_defaults(thread_state)
        state[str(tid)] = thread_state
    logging.info("加载状态成功，线程数=%d", len(state))
    return state


def save_state(state: Dict[str, Dict[str, Any]]) -> None:
    """
    将最新的 thread 状态写回磁盘。
    """
    try:
        STATE_FILE.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logging.info("已保存 STATE_FILE，线程数=%d", len(state))
    except Exception as exc:
        logging.warning("保存 STATE_FILE 出错：%s", exc)


def load_thread_history(tid: str) -> list:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    path = HISTORY_DIR / f"{tid}.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for entry in data:
                ensure_history_entry_defaults(entry)
            return data
    except Exception as exc:
        logging.warning("读取 history 文件失败：tid=%s, err=%s", tid, exc)
    return []


def save_thread_history(tid: str, records: list) -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    path = HISTORY_DIR / f"{tid}.json"
    try:
        path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logging.info("已保存 history：tid=%s, 条数=%d", tid, len(records))
    except Exception as exc:
        logging.warning("保存 history 文件失败：tid=%s, err=%s", tid, exc)


def delete_thread_history(tid: str) -> None:
    path = HISTORY_DIR / f"{tid}.json"
    try:
        path.unlink()
        logging.info("已删除 history 文件：tid=%s", tid)
    except FileNotFoundError:
        pass
    except Exception as exc:
        logging.warning("删除 history 文件失败：tid=%s, err=%s", tid, exc)


def load_thread_summaries(thread_id: str) -> list:
    """
    读取某个线程的长期摘要 data/summary/{thread_id}.json。
    返回列表，按文件中的顺序（通常就是按时间追加）。
    """
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    path = SUMMARY_DIR / f"{thread_id}.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        logging.warning("summary 文件格式异常：%s", path)
        return []
    except Exception as exc:
        logging.warning("读取 summary 文件出错：%s", exc)
        return []


def save_thread_summaries(tid: str, summaries: list) -> None:
    """
    将长期摘要列表写回磁盘。
    """
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    path = SUMMARY_DIR / f"{tid}.json"
    try:
        path.write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logging.info("已保存 summary：tid=%s, 条数=%d", tid, len(summaries))
    except Exception as exc:
        logging.warning("保存 summary 文件失败：tid=%s, err=%s", tid, exc)

def delete_thread_summaries(tid: str) -> None:
    path = SUMMARY_DIR / f"{tid}.json"
    try:
        path.unlink()
        logging.info("已删除 summary 文件：tid=%s", tid)
    except FileNotFoundError:
        pass
    except Exception as exc:
        logging.warning("删除 summary 文件失败：tid=%s, err=%s", tid, exc)


def load_system_prompt(default_text: str) -> str:
    try:
        prompt = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()
        if prompt:
            return prompt
        logging.warning("system_prompt.txt 为空，使用内置默认")
    except FileNotFoundError:
        logging.warning("system_prompt.txt not found, using built-in default")
    except Exception as exc:
        logging.warning("读取 system_prompt.txt 出错：%s，使用内置默认", exc)
    return default_text


def _detect_media_type(raw_item_type: str, media_obj: Any) -> str | None:
    """Best-effort media type detection."""
    if raw_item_type == "animated_media":
        return "sticker"
    if raw_item_type == "voice_media":
        return "audio"
    media_type_val = getattr(media_obj, "media_type", None)
    if media_type_val == 1:
        return "image"
    if media_type_val == 2:
        return "video"
    if media_type_val == 3:
        return "image"
    if getattr(media_obj, "video_url", None) or getattr(media_obj, "video_versions", None):
        return "video"
    if getattr(media_obj, "audio_url", None) or getattr(media_obj, "audio", None):
        return "audio"
    if getattr(media_obj, "thumbnail_url", None) or getattr(media_obj, "image_versions2", None):
        return "image"
    return None


def _extract_media_url(media_obj: Any) -> str | None:
    """Try multiple attributes to fetch downloadable media URL."""
    if media_obj is None:
        return None
    for attr in ("video_url", "audio_url", "thumbnail_url", "image_url"):
        val = getattr(media_obj, attr, None)
        if val:
            return val
    media_attr = getattr(media_obj, "media", None)
    if isinstance(media_attr, dict):
        for key in ("video", "video_url", "audio", "audio_url", "image", "image_url", "uri"):
            url = media_attr.get(key)
            if url:
                return url
    versions = getattr(media_obj, "video_versions", None)
    if versions:
        try:
            first = versions[0]
            if isinstance(first, dict):
                return first.get("url")
            return getattr(first, "url", None)
        except Exception:
            pass
    versions = getattr(media_obj, "image_versions2", None)
    if isinstance(versions, dict):
        candidates = versions.get("candidates") or []
        try:
            first = candidates[0]
            if isinstance(first, dict):
                return first.get("url")
            return getattr(first, "url", None)
        except Exception:
            pass
    return None


def parse_direct_message(
    msg: DirectMessage, thread_user_map: Dict[int, str]
) -> Dict[str, Any]:
    """Extract a normalized history entry from DirectMessage."""
    ts = getattr(msg, "timestamp", "")
    try:
        ts_str = ts.isoformat()
    except Exception:
        ts_str = str(ts)
    uid = getattr(msg, "user_id", None)
    username = thread_user_map.get(uid, f"user_{uid}") if uid is not None else "unknown"
    reply_obj = getattr(msg, "replied_to_message", None) or getattr(msg, "reply_to_message", None)
    reply_id = getattr(reply_obj, "id", None) if reply_obj else getattr(msg, "reply_to", None)
    reply_id_str = str(reply_id) if reply_id else None

    raw_item_type = getattr(msg, "item_type", None) or "text"
    item_type = raw_item_type or "text"
    text = msg.text or ""
    caption = None
    media_type = None
    media_url = None
    link_url = None
    link_meta = None

    if raw_item_type == "text":
        item_type = "text"
    elif raw_item_type in ("link", "shares"):
        item_type = "link"
        link_obj = getattr(msg, "link", None) or getattr(msg, "link_context", None)
        link_url = getattr(link_obj, "link_url", None) or getattr(link_obj, "url", None)
        link_meta = None
        if link_obj:
            link_meta = {
                "title": getattr(link_obj, "link_title", None) or getattr(link_obj, "title", None),
                "summary": getattr(link_obj, "link_summary", None) or getattr(link_obj, "summary", None),
                "site": getattr(link_obj, "link_site", None) or getattr(link_obj, "site_name", None),
            }
            if all(v is None for v in link_meta.values()):
                link_meta = None
        if not text:
            text = (link_meta or {}).get("title") or ""
    elif raw_item_type in ("media", "raven_media", "visual_media", "animated_media"):
        media_obj = getattr(msg, "media", None) or getattr(msg, "visual_media", None) or getattr(
            msg, "animated_media", None
        )
        caption = getattr(media_obj, "caption_text", None) or getattr(media_obj, "title", None)
        media_type = _detect_media_type(raw_item_type, media_obj)
        media_url = _extract_media_url(media_obj)
        item_type = media_type or ("sticker" if raw_item_type == "animated_media" else "mixed")
    elif raw_item_type == "voice_media":
        voice_obj = getattr(msg, "voice_media", None)
        media_type = "audio"
        media_url = None
        if voice_obj is not None:
            media_url = getattr(voice_obj, "audio", None) or getattr(voice_obj, "audio_url", None)
            if media_url is None:
                nested_media = getattr(voice_obj, "media", None)
                if isinstance(nested_media, dict):
                    media_url = (
                        nested_media.get("audio")
                        or nested_media.get("audio_src")
                        or nested_media.get("url")
                    )
        item_type = "audio"
    else:
        # fallback: keep raw item_type but still record text
        item_type = raw_item_type or "text"

    # 在构造 entry 之前，统一把 URL 类字段转换为 str，避免 HttpUrl 之类对象
    if media_url is not None and not isinstance(media_url, str):
        try:
            media_url = str(media_url)
        except Exception:
            media_url = None

    if link_url is not None and not isinstance(link_url, str):
        try:
            link_url = str(link_url)
        except Exception:
            link_url = None

    if isinstance(link_meta, dict):
        for k, v in list(link_meta.items()):
            if v is not None and not isinstance(v, str):
                try:
                    link_meta[k] = str(v)
                except Exception:
                    link_meta[k] = None

    entry = {
        "id": str(msg.id),
        "timestamp": ts_str,
        "user_id": uid,
        "username": username,
        "item_type": item_type,
        "text": text,
        "caption": caption,
        "media_type": media_type,
        "media_url": media_url,
        "media_vl_summary": None,
        "media_vl_tags": None,
        "link_url": link_url,
        "link_meta": link_meta,
        "reply_to": reply_id_str,
    }
    ensure_history_entry_defaults(entry)
    return entry


def update_history_for_thread(
    records: list,
    tid: str,
    new_msgs: List[DirectMessage],
    thread_user_map: Dict[int, str],
    self_user_id: int | None = None,
    self_username: str | None = None,
) -> bool:
    """
    追加线程历史（不再截断到 50 条）。
    """
    if not new_msgs:
        return False

    before_len = len(records)
    for entry in records:
        ensure_history_entry_defaults(entry)
    existing_ids = {entry.get("id") for entry in records}
    appended = False

    for msg in new_msgs:
        if msg.id in existing_ids:
            continue
        entry = parse_direct_message(msg, thread_user_map)
        records.append(entry)
        appended = True

    trimmed_records = records[-50:]

    # 统一把自己的用户名修正成标准形式
    if self_user_id is not None and self_username:
        for entry in trimmed_records:
            if entry.get("user_id") == self_user_id:
                entry["username"] = self_username
            if entry.get("username") == f"user_{self_user_id}":
                entry["username"] = self_username

    return appended or len(records) != before_len


def build_recent_context(
    thread_history: list,
    last_focus_id: str | None = None,
) -> str:
    """
    根据单个线程的 history 生成最近消息上下文（含 Focus 窗口）。
    只关心最近若干条消息（例如 50 条），不包含长期摘要。
    """
    if not thread_history:
        return "[Participants]\n- (unknown)\n[Recent messages]\n(暂无历史记录)"

    participants: List[str] = []
    seen = set()
    for entry in thread_history:
        name = entry.get("username") or f"user_{entry.get('user_id')}"
        if name not in seen:
            seen.add(name)
            participants.append(name)

    window = thread_history[-50:]
    id_to_entry = {entry.get("id"): entry for entry in window}
    focus_anchor_present = bool(
        last_focus_id and any(entry.get("id") == last_focus_id for entry in window)
    )
    focus_started = not focus_anchor_present
    focus_marker_inserted = False

    lines: List[str] = ["[Participants]"]
    for name in participants or ["(unknown)"]:
        lines.append(f"- {name}")
    lines.append("[Recent messages]")

    for idx, entry in enumerate(window):
        msg_id = entry.get("id")
        just_reached_anchor = False
        if not focus_started and last_focus_id and msg_id == last_focus_id:
            focus_started = True
            just_reached_anchor = True
        if focus_started and last_focus_id and not focus_marker_inserted and not just_reached_anchor:
            lines.append("---- Focus messages since last mention ----")
            focus_marker_inserted = True

        ts_str = entry.get("timestamp") or ""
        try:
            ts_fmt = datetime.fromisoformat(ts_str).strftime("%Y-%m-%d %H:%M")
        except Exception:
            ts_fmt = ts_str

        username = entry.get("username") or f"user_{entry.get('user_id')}"
        text_clean = (entry.get("text") or entry.get("caption") or "").replace("\n", " ").strip()
        lines.append(f"{ts_fmt} | {username}: {text_clean}")

        media_type = entry.get("media_type")
        if media_type:
            if media_type == "image":
                media_desc = "一条图片消息"
            elif media_type == "video":
                media_desc = "一条视频消息"
            elif media_type == "audio":
                media_desc = "一条语音消息"
            elif media_type == "sticker":
                media_desc = "一条贴纸/表情消息"
            else:
                media_desc = f"一条 {media_type} 消息"
            lines.append(f"  [媒体内容] {media_desc}")

        reply_to = entry.get("reply_to")
        if reply_to and reply_to in id_to_entry:
            replied_entry = id_to_entry[reply_to]
            replied_user = replied_entry.get("username") or f"user_{replied_entry.get('user_id')}"
            replied_text = (replied_entry.get("text") or "").replace("\n", " ").strip()
            if len(replied_text) > 50:
                replied_text = f"{replied_text[:50]}..."
            lines.append(f"  (reply to {replied_user}: {replied_text})")

        if text_clean:
            m = re.match(r"^@([A-Za-z0-9_.]+)\b", text_clean)
            if m:
                mention_name = m.group(1).lower()
                prev_entry = None
                for prev in reversed(window[:idx]):
                    prev_user = (prev.get("username") or "").lower()
                    if prev_user == mention_name:
                        prev_entry = prev
                        break
                if prev_entry:
                    prev_text = (prev_entry.get("text") or "").replace("\n", " ").strip()
                    if len(prev_text) > 50:
                        prev_text = f"{prev_text[:50]}..."
                    lines.append(f"  (可能是在回复 {prev_entry.get('username')}: {prev_text})")

    if focus_marker_inserted:
        lines.append("---- End focus window ----")

    return "\n".join(lines)


def format_summaries_for_context(thread_id: str, max_blocks: int = 3) -> str:
    """
    从 summary 文件中取最近几段摘要，压缩成一段「长期记忆」背景。
    只提供给模型一个大致印象，避免太长。
    """
    summaries = load_thread_summaries(thread_id)
    if not summaries:
        return ""

    # 只取最近 max_blocks 段摘要
    blocks = summaries[-max_blocks:]
    lines: List[str] = ["[Long-term memory]"]

    for blk in blocks:
        time_start = blk.get("time_start") or ""
        time_end = blk.get("time_end") or ""
        msg_count = blk.get("message_count") or 0

        # 参与者名字，最多 4 个
        participants = blk.get("participants") or []
        names: List[str] = []
        for p in participants:
            uname = p.get("username") or f"user_{p.get('user_id')}"
            if uname not in names:
                names.append(uname)
            if len(names) >= 4:
                break
        names_str = "、".join(names) if names else "若干群成员"

        topics = blk.get("topics") or []
        important = blk.get("important_points") or []

        topics_str = "；".join(topics[:2]) if topics else ""
        important_str = "；".join(important[:2]) if important else ""

        lines.append(
            f"- 时间段 {time_start} ~ {time_end}，约 {msg_count} 条消息，主要参与者：{names_str}"
        )
        if topics_str:
            lines.append(f"  · 主要话题：{topics_str}")
        if important_str:
            lines.append(f"  · 关键记忆点：{important_str}")

    return "\n".join(lines)


def build_context_with_memory(
    thread_id: str,
    thread_history: list,
    last_focus_id: str | None = None,
    fish_mode: bool = False,
    long_memory_enabled: bool = True,
) -> str:
    """
    最终发给 Qwen 的上下文：
    1. 前半部分是若干段长期摘要 [Long-term memory]
    2. 后半部分是最近消息窗口 [Participants] + [Recent messages]
    """
    if fish_mode:
        recent_window = thread_history[-10:]
        return build_recent_context(recent_window, last_focus_id)

    long_term = format_summaries_for_context(thread_id) if long_memory_enabled else ""
    recent = build_recent_context(thread_history, last_focus_id)

    if long_term:
        return long_term + "\n\n" + recent
    return recent


def ask_qwen(context_text: str, temperature: float = 0.7, max_tokens: int = 5120) -> str:
    """
    通过兼容 OpenAI 的 /chat/completions 接口调用 Qwen。
    """
    api_key = os.getenv("QWEN_API_KEY")
    api_url = os.getenv("QWEN_API_URL")
    model = os.getenv("QWEN_MODEL") or "qwen2.5-32b-instruct"

    if not api_key:
        raise RuntimeError("缺少 QWEN_API_KEY")
    if not api_url:
        raise RuntimeError("缺少 QWEN_API_URL")

    system_prompt = load_system_prompt(DEFAULT_SYSTEM_PROMPT)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": context_text,
            },
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
    except Exception as exc:
        logging.error("请求 Qwen 失败：%s", exc)
        raise

    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
    except Exception as exc:
        logging.error("解析 Qwen 返回值失败：%s，原始内容=%s", exc, resp.text)
        raise RuntimeError("Qwen 响应格式异常") from exc

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Qwen 返回内容为空")
    return content.strip()


def compute_batch_stats(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    计算一段消息的基础统计信息。
    """
    if not batch:
        return {
            "time_start": "",
            "time_end": "",
            "message_count": 0,
            "participants": [],
        }
    participants_count: Dict[Tuple[Any, Any], int] = defaultdict(int)
    for entry in batch:
        uid = entry.get("user_id")
        uname = entry.get("username") or f"user_{uid}"
        participants_count[(uid, uname)] += 1

    participants = [
        {"user_id": uid, "username": uname, "message_count": cnt}
        for (uid, uname), cnt in sorted(
            participants_count.items(), key=lambda kv: kv[1], reverse=True
        )
    ]

    time_start = batch[0].get("timestamp") or ""
    time_end = batch[-1].get("timestamp") or ""
    return {
        "time_start": time_start,
        "time_end": time_end,
        "message_count": len(batch),
        "participants": participants,
    }


def format_batch_for_summary(batch: List[Dict[str, Any]], stats: Dict[str, Any]) -> str:
    """
    把 50 条消息 + 预统计信息，整理成一个发给 Qwen 的文本。
    """
    lines: List[str] = [
        "【对话时间范围】",
        f"起始时间: {stats.get('time_start', '')}",
        f"结束时间: {stats.get('time_end', '')}",
        f"消息总数: {stats.get('message_count', len(batch))}",
        "",
        "【参与者及发言条数】",
    ]
    participants = stats.get("participants") or []
    if participants:
        for p in participants:
            lines.append(
                f"- 用户名: {p.get('username')} (user_id={p.get('user_id')}), 发言条数: {p.get('message_count')}"
            )
    else:
        lines.append("- (unknown)")

    lines.append("")
    lines.append("【原始对话记录】(按时间从旧到新)")
    for entry in batch:
        ts_str = entry.get("timestamp") or ""
        username = entry.get("username") or f"user_{entry.get('user_id')}"
        text = (entry.get("text") or entry.get("caption") or "").replace("\n", " ").strip()
        if not text:
            text = "(无文本，仅媒体)"
        lines.append(f"{ts_str} | {username}: {text}")
        if entry.get("media_type"):
            media_type = entry.get("media_type")
            if media_type == "image":
                media_desc = "一条图片消息"
            elif media_type == "video":
                media_desc = "一条视频消息"
            elif media_type == "audio":
                media_desc = "一条语音消息"
            elif media_type == "sticker":
                media_desc = "一条贴纸/表情消息"
            else:
                media_desc = f"一条 {media_type} 消息"
            lines.append(f"  [媒体内容] {media_desc}")
    return "\n".join(lines)


def ask_qwen_summary(summary_context: str) -> Dict[str, Any]:
    """
    调用 Qwen 文本模型生成结构化摘要。
    """
    api_key = os.getenv("QWEN_API_KEY")
    api_url = os.getenv("QWEN_API_URL")
    model = os.getenv("QWEN_SUMMARY_MODEL") or os.getenv("QWEN_MODEL") or "qwen2.5-32b-instruct"

    if not api_key:
        raise RuntimeError("缺少 QWEN_API_KEY")
    if not api_url:
        raise RuntimeError("缺少 QWEN_API_URL")

    system_prompt = (
        "你是一个对话分析助手。给你一段群聊记录（含时间、用户名、文本）和预统计信息，"
        "请用 JSON 格式输出："
        "1. topics: 本段对话涉及的主要话题列表（字符串数组）；"
        "2. personas: 每个用户的一句话性格/说话风格描述，数组，每个元素形如 "
        '{"username": "...", "description": "..."}; '
        "3. important_points: 本段对话中需要后续记住的重点事项或 TODO，字符串数组；"
        "4. raw_summary: 对这一段对话的详细自然语言总结（一两段话）。只输出 JSON，不要额外解释。"
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": summary_context},
        ],
        "temperature": 0.4,
        "max_tokens": 2048,
    }

    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=90)
        resp.raise_for_status()
    except Exception as exc:
        logging.error("请求 Qwen(summary) 失败：%s", exc)
        raise

    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
    except Exception as exc:
        logging.error("解析 Qwen(summary) 返回值失败：%s，原始内容=%s", exc, resp.text)
        raise RuntimeError("Qwen summary 响应格式异常") from exc

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Qwen summary 返回内容为空")
    content = content.strip()
    try:
        summary_obj = json.loads(content)
    except Exception:
        logging.warning("Qwen summary 返回非 JSON，使用兜底文本")
        summary_obj = {
            "topics": [],
            "personas": [],
            "important_points": [],
            "raw_summary": content,
        }
    return summary_obj


def summarize_and_append(thread_id: str, batch: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """
    对某个线程的一段消息做摘要，并追加到 summary 文件。
    """
    if not batch:
        return None
    stats = compute_batch_stats(batch)
    summary_context = format_batch_for_summary(batch, stats)
    try:
        llm_summary = ask_qwen_summary(summary_context)
    except Exception as exc:
        logging.error("生成摘要失败：%s", exc, exc_info=True)
        return None

    summaries = load_thread_summaries(thread_id)
    record = {
        "summary_id": len(summaries) + 1,
        "thread_id": thread_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "time_start": stats.get("time_start"),
        "time_end": stats.get("time_end"),
        "message_count": stats.get("message_count"),
        "participants": stats.get("participants"),
        "topics": llm_summary.get("topics") or [],
        "personas": llm_summary.get("personas") or [],
        "important_points": llm_summary.get("important_points") or [],
        "raw_summary": llm_summary.get("raw_summary") or "",
    }
    summaries.append(record)
    save_thread_summaries(thread_id, summaries)
    return record


def summarize_messages_inline(messages: List[Dict[str, Any]]) -> str | None:
    """
    生成即时摘要（不写入文件），返回 raw_summary 文本。
    """
    if not messages:
        return None
    stats = compute_batch_stats(messages)
    summary_context = format_batch_for_summary(messages, stats)
    try:
        summary_obj = ask_qwen_summary(summary_context)
    except Exception as exc:
        logging.error("即时摘要失败：%s", exc, exc_info=True)
        return None
    return summary_obj.get("raw_summary") or ""


def handle_mode_command(
    text: str,
    thread_state: Dict[str, Any],
    tid: str,
    client: Client,
) -> bool:
    normalized = normalize_arg_text(text)
    parts = normalized.split()
    if not parts or not parts[0].lower().startswith("/mode"):
        return False
    if len(parts) < 2:
        client.direct_answer(tid, "Usage: /mode <quiet|@andreply|mention|all>")
        return True
    mode = parts[1].lower()
    if mode not in ("quiet", "@andreply", "mention", "all"):
        client.direct_answer(tid, "Usage: /mode <quiet|@andreply|mention|all>")
        return True
    thread_state["reply_mode"] = mode
    client.direct_answer(tid, f"Reply mode set to: {mode}")
    return True


def handle_maxlen_command(
    text: str,
    thread_state: Dict[str, Any],
    tid: str,
    client: Client,
) -> bool:
    normalized = normalize_arg_text(text)
    parts = normalized.split()
    if not parts or not parts[0].lower().startswith("/maxlen"):
        return False
    if len(parts) < 2:
        client.direct_answer(tid, "Usage: /maxlen <number>, e.g. /maxlen 300")
        return True
    try:
        val = int(float(parts[1]))
    except Exception:
        client.direct_answer(tid, "Usage: /maxlen <number>, e.g. /maxlen 300")
        return True
    val = max(32, min(2048, val))
    thread_state["max_reply_len"] = val
    client.direct_answer(tid, f"Max reply length set to {val}.")
    return True


def handle_temp_command(
    text: str,
    thread_state: Dict[str, Any],
    tid: str,
    client: Client,
) -> bool:
    normalized = normalize_arg_text(text)
    parts = normalized.split()
    if not parts or not parts[0].lower().startswith("/temp"):
        return False
    if len(parts) < 2:
        client.direct_answer(tid, "Usage: /temp <0.1-1.0>, e.g. /temp 0.6")
        return True
    try:
        val = float(parts[1])
    except Exception:
        client.direct_answer(tid, "Usage: /temp <0.1-1.0>, e.g. /temp 0.6")
        return True
    val = max(0.1, min(1.0, val))
    thread_state["temperature"] = val
    client.direct_answer(tid, f"Temperature set to {val:.2f}.")
    return True


def handle_mem_command(
    msg: DirectMessage,
    thread_state: Dict[str, Any],
    thread_history: list,
    tid: str,
    client: Client,
) -> tuple[bool, list, bool]:
    text = msg.text or ""
    normalized = normalize_arg_text(text)
    parts = normalized.split()
    if not parts or not parts[0].lower().startswith("/mem"):
        return False, thread_history, False

    state_dirty = False
    if len(parts) < 2:
        client.direct_answer(
            tid,
            "Usage: /mem <trim|toggle_long|toggle_fish|clear_recent|clear_all>",
        )
        return True, thread_history, state_dirty

    if len(parts) >= 3 and parts[1].isdigit() and parts[2].lower() == "h":
        sub = f"{parts[1]}h"
    else:
        sub = parts[1].lower()
    if sub == "trim":
        if len(thread_history) < 10:
            client.direct_answer(tid, "Not enough messages to trim.")
            return True, thread_history, state_dirty
        if len(thread_history) >= 50:
            kept_prefix = thread_history[:-50]
            new_tail = thread_history[-10:]
            thread_history = kept_prefix + new_tail
        else:
            thread_history = thread_history[-10:]
        save_thread_history(tid, thread_history)
        thread_state["last_summary_index"] = min(
            thread_state.get("last_summary_index", 0), len(thread_history)
        )
        thread_state["total_messages_seen"] = len(thread_history)
        state_dirty = True
        client.direct_answer(tid, "Trimmed recent history for this chat.")
        return True, thread_history, state_dirty

    if sub == "toggle_long":
        thread_state["long_memory_enabled"] = not thread_state.get("long_memory_enabled", True)
        client.direct_answer(
            tid,
            "Long-term memory is now ON."
            if thread_state["long_memory_enabled"]
            else "Long-term memory is now OFF.",
        )
        state_dirty = True
        return True, thread_history, state_dirty

    if sub == "toggle_fish":
        thread_state["fish_mode"] = not thread_state.get("fish_mode", False)
        client.direct_answer(
            tid,
            "Goldfish mode is now ON (I will only look at the last 10 messages)."
            if thread_state["fish_mode"]
            else "Goldfish mode is now OFF.",
        )
        state_dirty = True
        return True, thread_history, state_dirty

    if sub == "clear_recent":
        thread_state["pending_memory_action"] = "clear_recent"
        thread_state["pending_memory_user_id"] = msg.user_id
        client.direct_answer(
            tid,
            'You are about to CLEAR recent memory for this chat.\nType "yes" to confirm, or "no" to cancel.',
        )
        state_dirty = True
        return True, thread_history, state_dirty

    if sub == "clear_all":
        thread_state["pending_memory_action"] = "clear_all"
        thread_state["pending_memory_user_id"] = msg.user_id
        client.direct_answer(
            tid,
            'WARNING: This will DELETE ALL memory for this chat (recent + long-term).\nType "yes" to confirm, or "no" to cancel.',
        )
        state_dirty = True
        return True, thread_history, state_dirty

    client.direct_answer(
        tid,
        "Usage: /mem <trim|toggle_long|toggle_fish|clear_recent|clear_all>",
    )
    return True, thread_history, state_dirty


def handle_summary_command(
    msg: DirectMessage,
    thread_history: list,
    tid: str,
    client: Client,
) -> bool:
    text = msg.text or ""
    normalized = normalize_arg_text(text)
    parts = normalized.split()
    if not parts or not parts[0].lower().startswith("/summary"):
        return False
    if len(parts) < 2:
        client.direct_answer(tid, "Usage: /summary <last|50|1h>")
        return True

    sub = parts[1].lower()
    if sub == "last":
        summaries = load_thread_summaries(tid)
        if not summaries:
            client.direct_answer(tid, "No long-term summary yet for this chat.")
            return True
        latest = summaries[-1]
        raw_summary = latest.get("raw_summary") or ""
        client.direct_answer(
            tid,
            f"Latest long-term memory for this chat:\n\n{raw_summary}",
        )
        return True

    if sub == "50":
        batch = thread_history[-50:]
        summary_text = summarize_messages_inline(batch)
        if summary_text:
            client.direct_answer(
                tid,
                f"Recap of the last 50 messages:\n\n{summary_text}",
            )
        else:
            client.direct_answer(tid, "Unable to summarise the last 50 messages.")
        return True

    m = re.match(r"^(\d+)h$", sub)
    if m:
        hours = int(m.group(1))
        cutoff = time.time() - hours * 3600
        filtered = []
        for entry in thread_history:
            ts_obj = parse_timestamp(entry.get("timestamp", ""))
            if ts_obj and ts_obj.timestamp() >= cutoff:
                filtered.append(entry)
        if not filtered:
            client.direct_answer(tid, f"No messages in the last {hours}h.")
            return True
        summary_text = summarize_messages_inline(filtered)
        if summary_text:
            client.direct_answer(
                tid,
                f"Recap of the last {hours}h:\n\n{summary_text}",
            )
        else:
            client.direct_answer(tid, f"Unable to summarise the last {hours}h.")
        return True

    client.direct_answer(tid, "Usage: /summary <last|50|1h>")
    return True


def send_typing_indicator(client: Client, thread_id: str) -> None:
    """
    尝试向指定 thread 发送“正在输入”状态，失败时仅记录日志。
    """
    try:
        client.private_request(
            "direct_v2/threads/broadcast/activity/",
            data={
                "activity_status": "typing",
                "thread_id": thread_id,
            },
        )
        logging.info("Sent typing indicator: thread=%s", thread_id)
    except Exception as exc:
        logging.debug("Failed to send typing indicator for thread %s: %s", thread_id, exc)


def mark_message_seen(
    client: Client,
    thread_id: str,
    msg: DirectMessage,
    thread_user_map: Dict[int, str],
    is_dm: bool,
) -> None:
    """
    尝试标记指定消息为已读。
    """
    if msg is None:
        return
    logging.info(
        "Sending seen: thread=%s, msg_id=%s, username=%s, is_dm=%s",
        thread_id,
        msg.id,
        thread_user_map.get(msg.user_id),
        is_dm,
    )
    try:
        client.direct_message_seen(thread_id, msg.id)
    except Exception as exc:
        logging.error(
            "direct_message_seen failed: thread=%s, msg_id=%s, err=%s",
            thread_id,
            msg.id,
            exc,
            exc_info=True,
        )


def login_client(username: str, password: str) -> Client:
    """
    封装登录逻辑：优先使用 IG_SESSIONID，其次复用 session.json，最后走用户名密码。
    """
    def _try_delete_session_file() -> None:
        try:
            SESSION_FILE.unlink(missing_ok=True)
            logging.info("已删除失效的 session 文件：%s", SESSION_FILE)
        except Exception as exc:
            logging.warning("删除旧 session 文件失败：%s", exc)

    sessionid_env = os.getenv("IG_SESSIONID")
    if sessionid_env:
        try:
            client = Client()
            client.login_by_sessionid(sessionid_env)
            client.dump_settings(SESSION_FILE)
            logging.info("使用 IG_SESSIONID 登录成功，已保存 session 到文件。")
            return client
        except Exception as exc:
            logging.warning("使用 IG_SESSIONID 登录失败，将尝试其它方式：%s", exc, exc_info=True)

    if SESSION_FILE.exists():
        try:
            logging.info("发现 session.json，尝试复用登录状态...")
            client.load_settings(SESSION_FILE)
            client.login(username, password)
            client.dump_settings(SESSION_FILE)
            logging.info("复用 session 登录成功，已更新 session 文件。")
            return client
        except LoginRequired as exc:
            logging.warning("缓存 session 已失效，需重新登录：%s", exc)
            _try_delete_session_file()
        except Exception as exc:
            logging.warning("复用 session 失败，将尝试重新登录：%s", exc, exc_info=True)
            _try_delete_session_file()

    logging.info("未找到有效 session，使用用户名密码重新登录...")
    client = Client()
    client.login(username, password)
    client.dump_settings(SESSION_FILE)
    logging.info("已保存新的 session 到 %s", SESSION_FILE)
    return client


def main_loop() -> None:
    load_dotenv()

    ig_username = os.getenv("IG_USERNAME")
    ig_password = os.getenv("IG_PASSWORD")
    mention_name = os.getenv("BOT_MENTION_NAME") or ig_username

    if not ig_username or not ig_password:
        raise RuntimeError("请在 .env 中配置 IG_USERNAME 和 IG_PASSWORD")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    state_raw = load_state()
    state = {tid: ensure_thread_state_defaults(st) for tid, st in state_raw.items()}

    cl = login_client(ig_username, ig_password)
    try:
        me = cl.user_info_by_username(ig_username)
    except LoginRequired as exc:
        logging.warning("首次读取用户信息时发现登录失效，刷新 session 并重试：%s", exc)
        try:
            SESSION_FILE.unlink(missing_ok=True)
        except Exception as exc_del:
            logging.warning("删除旧 session 文件失败：%s", exc_del)
        cl = login_client(ig_username, ig_password)
        me = cl.user_info_by_username(ig_username)
    my_pk = me.pk
    logging.info("登录成功，当前账号：%s (pk=%s)", me.username, my_pk)

    mention_token = f"@{(mention_name or '').lower()}"
    logging.info("监听被 @ 关键字：%s", mention_token)
    enable_seen = os.getenv("ENABLE_SEEN", "true").lower() == "true"
    logging.info("ENABLE_SEEN=%s", enable_seen)

    last_activity_ts = time.time()

    while True:
        try:
            had_new_messages = False
            state_dirty = False

            threads = cl.direct_threads(amount=10)
            for thread in threads:
                tid = thread.id
                participants = thread.users
                is_private = len(participants) == 1
                thread_user_map = {user.pk: user.username for user in participants}
                thread_user_map[my_pk] = me.username

                thread_history = load_thread_history(tid)

                messages = cl.direct_messages(tid, amount=20)
                if not messages:
                    continue

                messages_sorted = sorted(messages, key=lambda m: m.timestamp)
                thread_state = ensure_thread_state_defaults(state.get(tid, {}))
                state[tid] = thread_state
                last_seen_id = thread_state.get("last_message_id")
                last_focus_id = thread_state.get("last_mention_id")

                def _is_new(msg_id: str, last_seen: str | None) -> bool:
                    if last_seen is None:
                        return True
                    try:
                        return int(msg_id) > int(last_seen)
                    except Exception:
                        return str(msg_id) > str(last_seen)

                new_msgs = [
                    msg for msg in messages_sorted if _is_new(msg.id, last_seen_id)
                ]
                if not new_msgs:
                    continue

                had_new_messages = True
                changed = update_history_for_thread(
                    thread_history,
                    tid,
                    new_msgs,
                    thread_user_map,
                    self_user_id=my_pk,
                    self_username=me.username,
                )
                if changed:
                    save_thread_history(tid, thread_history)
                    thread_state["total_messages_seen"] = len(thread_history)
                    state_dirty = True

                other_new_msgs: List[DirectMessage] = []
                target_msg: DirectMessage | None = None
                focus_id_for_context = None

                for msg in new_msgs:
                    thread_state["last_message_id"] = msg.id
                    state[tid] = thread_state
                    state_dirty = True

                    text_lower = (msg.text or "").lower()
                    if msg.user_id != my_pk:
                        other_new_msgs.append(msg)
                    if msg.user_id == my_pk:
                        continue

                    text_raw = msg.text or ""
                    stripped_lower = text_raw.strip().lower()
                    is_explicit_mention = mention_token in text_lower if mention_token else False
                    reply_obj = getattr(msg, "replied_to_message", None) or getattr(msg, "reply_to_message", None)
                    reply_to_user_id = getattr(reply_obj, "user_id", None)
                    is_direct_reply_to_me = reply_to_user_id == my_pk
                    contains_keyword_mention = any(k in text_lower for k in REPLY_KEYWORDS)

                    pending_action = thread_state.get("pending_memory_action")
                    pending_user = thread_state.get("pending_memory_user_id")
                    if pending_action and (pending_user is None or pending_user == msg.user_id):
                        if stripped_lower in ("yes", "y"):
                            if pending_action == "clear_recent":
                                thread_history = []
                                save_thread_history(tid, thread_history)
                                thread_state["last_summary_index"] = 0
                                thread_state["total_messages_seen"] = 0
                                thread_state["last_mention_id"] = None
                                cl.direct_answer(tid, "Done. Recent memory cleared for this chat.")
                            elif pending_action == "clear_all":
                                delete_thread_history(tid)
                                delete_thread_summaries(tid)
                                thread_history = []
                                thread_state["last_summary_index"] = 0
                                thread_state["total_messages_seen"] = 0
                                thread_state["last_mention_id"] = None
                                cl.direct_answer(tid, "Done. All memory cleared for this chat.")
                            thread_state["pending_memory_action"] = None
                            thread_state["pending_memory_user_id"] = None
                            state_dirty = True
                            continue
                        if stripped_lower in ("no", "n", "cancel"):
                            cl.direct_answer(tid, "Cancelled. No memory was deleted.")
                            thread_state["pending_memory_action"] = None
                            thread_state["pending_memory_user_id"] = None
                            state_dirty = True
                            continue
                        cl.direct_answer(tid, 'Please reply "yes" to confirm, or "no" to cancel.')
                        continue

                    config_trigger = False
                    if is_private:
                        config_trigger = "/config" in text_lower
                    else:
                        config_trigger = is_explicit_mention and "/config" in text_lower
                    if config_trigger:
                        thread_state["config_mode"] = True
                        state_dirty = True
                        cl.direct_answer(tid, CONFIG_MENU_TEXT)
                        continue

                    if thread_state.get("config_mode"):
                        if stripped_lower in ("/exit", "exit", "quit", "q"):
                            thread_state["config_mode"] = False
                            state_dirty = True
                            cl.direct_answer(tid, "Config session closed. Back to normal mode.")
                            continue
                        if handle_mode_command(text_raw, thread_state, tid, cl):
                            state_dirty = True
                            continue
                        handled_mem, thread_history, dirty_mem = handle_mem_command(
                            msg, thread_state, thread_history, tid, cl
                        )
                        if handled_mem:
                            state_dirty = state_dirty or dirty_mem
                            continue
                        if handle_summary_command(msg, thread_history, tid, cl):
                            continue
                        if handle_maxlen_command(text_raw, thread_state, tid, cl):
                            state_dirty = True
                            continue
                        if handle_temp_command(text_raw, thread_state, tid, cl):
                            state_dirty = True
                            continue
                        continue

                    is_mem_cmd = (is_private and text_lower.strip().startswith("/mem")) or (
                        (not is_private) and is_explicit_mention and text_lower.strip().startswith("/mem")
                    )
                    if is_mem_cmd:
                        handled_mem, thread_history, dirty_mem = handle_mem_command(
                            msg, thread_state, thread_history, tid, cl
                        )
                        if handled_mem:
                            state_dirty = state_dirty or dirty_mem
                            continue

                    is_summary_cmd = (is_private and text_lower.strip().startswith("/summary")) or (
                        (not is_private) and is_explicit_mention and text_lower.strip().startswith("/summary")
                    )
                    if is_summary_cmd and handle_summary_command(msg, thread_history, tid, cl):
                        continue

                    is_mode_cmd = (is_private and text_lower.strip().startswith("/mode")) or (
                        (not is_private) and is_explicit_mention and text_lower.strip().startswith("/mode")
                    )
                    if is_mode_cmd and handle_mode_command(text_raw, thread_state, tid, cl):
                        state_dirty = True
                        continue

                    is_maxlen_cmd = (is_private and text_lower.strip().startswith("/maxlen")) or (
                        (not is_private)
                        and is_explicit_mention
                        and text_lower.strip().startswith("/maxlen")
                    )
                    if is_maxlen_cmd and handle_maxlen_command(text_raw, thread_state, tid, cl):
                        state_dirty = True
                        continue

                    is_temp_cmd = (is_private and text_lower.strip().startswith("/temp")) or (
                        (not is_private) and is_explicit_mention and text_lower.strip().startswith("/temp")
                    )
                    if is_temp_cmd and handle_temp_command(text_raw, thread_state, tid, cl):
                        state_dirty = True
                        continue

                    reply_mode = thread_state.get("reply_mode", "@andreply")
                    if is_private:
                        if reply_mode == "quiet":
                            continue
                        should_reply = True
                    else:
                        if reply_mode == "quiet":
                            should_reply = False
                        elif reply_mode == "@andreply":
                            should_reply = is_explicit_mention or is_direct_reply_to_me
                        elif reply_mode == "mention":
                            should_reply = (
                                is_explicit_mention
                                or is_direct_reply_to_me
                                or contains_keyword_mention
                            )
                        elif reply_mode == "all":
                            should_reply = True
                        else:
                            should_reply = is_explicit_mention or is_direct_reply_to_me
                    if not should_reply:
                        continue

                    target_msg = msg
                    focus_id_for_context = last_focus_id if not is_private else None

                # 长期摘要触发
                last_summary_index = _safe_int(thread_state.get("last_summary_index"), 0)
                if last_summary_index > len(thread_history):
                    last_summary_index = len(thread_history)
                    thread_state["last_summary_index"] = last_summary_index
                    state_dirty = True
                while len(thread_history) - last_summary_index >= SUMMARY_BATCH_SIZE:
                    batch = thread_history[
                        last_summary_index : last_summary_index + SUMMARY_BATCH_SIZE
                    ]
                    summary_record = summarize_and_append(tid, batch)
                    if summary_record:
                        last_summary_index += SUMMARY_BATCH_SIZE
                        thread_state["last_summary_index"] = last_summary_index
                        state_dirty = True
                    else:
                        break
                thread_state["total_messages_seen"] = len(thread_history)
                if len(thread_history) > MAX_HISTORY_KEEP:
                    trim_count = len(thread_history) - MAX_HISTORY_KEEP
                    del thread_history[:trim_count]
                    thread_state["last_summary_index"] = max(
                        0, thread_state.get("last_summary_index", 0) - trim_count
                    )
                    thread_state["total_messages_seen"] = len(thread_history)
                    state_dirty = True
                    save_thread_history(tid, thread_history)
                state[tid] = thread_state

                if enable_seen and is_private and other_new_msgs:
                    msg_to_seen = other_new_msgs[-1]
                    if msg_to_seen.user_id != my_pk:
                        mark_message_seen(cl, tid, msg_to_seen, thread_user_map, is_dm=True)

                if target_msg:
                    if enable_seen and not is_private and target_msg.user_id != my_pk:
                        mark_message_seen(cl, tid, target_msg, thread_user_map, is_dm=False)

                    fish_mode_enabled = bool(thread_state.get("fish_mode"))
                    long_memory_enabled = bool(thread_state.get("long_memory_enabled", True)) and not fish_mode_enabled
                    context = build_context_with_memory(
                        tid,
                        thread_history,
                        focus_id_for_context,
                        fish_mode=fish_mode_enabled,
                        long_memory_enabled=long_memory_enabled,
                    )
                    logging.info(
                        "准备回复：thread=%s, msg_id=%s, context_len=%d",
                        tid,
                        target_msg.id,
                        len(context),
                    )
                    send_typing_indicator(cl, tid)
                    time.sleep(random.uniform(1.0, 2.0))

                    try:
                        temperature = float(thread_state.get("temperature", 0.5))
                        max_tokens = int(thread_state.get("max_reply_len", 5120))
                        reply = ask_qwen(context, temperature=temperature, max_tokens=max_tokens)
                    except Exception as exc:
                        logging.error("调用 Qwen 出错，使用兜底文案：%s", exc)
                        reply = "网卡了，没收到消息，可以再说一遍吗？"

                    logging.info(
                        "发送回复：thread=%s, target_msg=%s, reply_preview=%r",
                        tid,
                        target_msg.id,
                        reply[:100],
                    )
                    cl.direct_answer(tid, reply)
                    if not is_private and target_msg:
                        thread_state["last_mention_id"] = target_msg.id
                        state_dirty = True

            if state_dirty:
                save_state(state)

            now = time.time()
            if had_new_messages:
                last_activity_ts = now

            idle = (now - last_activity_ts) > 20 * 60
            any_config_active = any(
                isinstance(ts, dict) and ts.get("config_mode") for ts in state.values()
            )
            if any_config_active:
                sleep_sec = 1.0
            else:
                if idle:
                    sleep_sec = random.uniform(60, 180)
                else:
                    sleep_sec = 2 + random.uniform(1, 10)

            logging.info(
                "End of loop: idle=%s, had_new_messages=%s, sleep=%.1f seconds",
                idle,
                had_new_messages,
                sleep_sec,
            )
            time.sleep(sleep_sec)

        except Exception as exc:
            # 捕获主循环异常，避免直接退出
            logging.error("主循环异常：%s", exc, exc_info=True)
            time.sleep(60)


if __name__ == "__main__":
    try:
        main_loop()
    except Exception as exc:
        logging.error("程序启动失败：%s", exc)
        raise
