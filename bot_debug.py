import json
import logging
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import requests
from dotenv import load_dotenv
from instagrapi import Client
from instagrapi.mixins.direct import DirectMessage

# 调试开关：True 时不再要求消息包含 @mention，也会回复（便于排查流水线）
DEBUG_IGNORE_MENTION = True

# 路径与文件常量
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROMPTS_DIR = BASE_DIR / "prompts"
SESSION_FILE = DATA_DIR / "session.json"
STATE_FILE = DATA_DIR / "state.json"
HISTORY_DIR = DATA_DIR / "history"
SYSTEM_PROMPT_FILE = PROMPTS_DIR / "system_prompt.txt"

DEFAULT_SYSTEM_PROMPT = (
    "你是一个 Instagram 群聊里的成员，昵称叫 QwenBot。\n"
    "你根据最近的聊天内容做出自然、有趣、简短的回复，一般控制在 1～3 句，不要长篇大论。\n"
    "聊天记录按时间从旧到新排列，后面的消息更近。最近的消息以及上次被 @ 后的消息更重要，请优先回应最新的 @ 和其后的对话。\n"
    "聊天历史中的每一行格式为“时间 | 用户名: 内容”，请严格按照用户名判断说话人，不要混淆。"
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)


def load_state() -> Dict[str, Dict[str, str]]:
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

    state: Dict[str, Dict[str, str]] = {}
    for tid, raw in data.items():
        thread_state: Dict[str, str] = {}
        if isinstance(raw, dict):
            last_msg = raw.get("last_message_id")
            last_mention = raw.get("last_mention_id")
        else:
            last_msg = raw
            last_mention = None
        if last_msg is not None:
            thread_state["last_message_id"] = str(last_msg)
        if last_mention is not None:
            thread_state["last_mention_id"] = str(last_mention)
        state[str(tid)] = thread_state
    logging.info("加载状态成功，线程数=%d", len(state))
    return state


def save_state(state: Dict[str, Dict[str, str]]) -> None:
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


def update_history_for_thread(
    records: list,
    new_msgs: List[DirectMessage],
    thread_user_map: Dict[int, str],
) -> bool:
    """
    追加线程历史，并保持滑动窗口。
    """
    if not new_msgs:
        return False

    before_len = len(records)
    existing_ids = {entry.get("id") for entry in records}
    appended = False

    for msg in new_msgs:
        if msg.id in existing_ids:
            continue
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

        records.append(
            {
                "id": str(msg.id),
                "timestamp": ts_str,
                "user_id": uid,
                "username": username,
                "text": msg.text or "",
                "reply_to": reply_id_str,
            }
        )
        appended = True

    trimmed = records[-50:]
    records[:] = trimmed
    return appended or len(trimmed) != before_len


def build_context(
    thread_history: list,
    last_focus_id: str | None = None,
) -> str:
    """
    把 per-thread 历史整理为模型可读的上下文。
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
        text_clean = (entry.get("text") or "").replace("\n", " ").strip()
        lines.append(f"{ts_fmt} | {username}: {text_clean}")

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


def ask_qwen(context_text: str) -> str:
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
        "temperature": 0.7,
        "max_tokens": 256,
    }

    logging.info("调用 Qwen：model=%s, payload_messages=%d", model, len(payload["messages"]))

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
    reply = content.strip()
    logging.info("Qwen 返回内容预览：%r", reply[:120])
    return reply


def login_client(username: str, password: str) -> Client:
    """
    封装登录逻辑：优先复用 session.json。
    """
    client = Client()
    if SESSION_FILE.exists():
        logging.info("发现 session.json，尝试复用登录状态...")
        client.load_settings(SESSION_FILE)
        client.login(username, password)
    else:
        logging.info("未找到 session.json，使用用户名密码登录...")
        client.login(username, password)
        client.dump_settings(SESSION_FILE)
        logging.info("已保存 session 到 %s", SESSION_FILE)
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
    state = load_state()

    cl = login_client(ig_username, ig_password)
    me = cl.user_info_by_username(ig_username)
    my_pk = me.pk
    logging.info("登录成功，当前账号：%s (pk=%s)", me.username, my_pk)

    mention_token = f"@{(mention_name or '').lower()}"
    logging.info("监听被 @ 关键字：%s，DEBUG_IGNORE_MENTION=%s", mention_token, DEBUG_IGNORE_MENTION)

    last_activity_ts = time.time()

    while True:
        try:
            had_new_messages = False
            state_dirty = False

            logging.info("---- 新一轮轮询开始 ----")
            threads = cl.direct_threads(amount=10)
            logging.info("本轮获取到线程数：%d", len(threads))

            for thread in threads:
                tid = thread.id
                participants = thread.users
                is_private = len(participants) == 1
                thread_user_map = {user.pk: user.username for user in participants}
                logging.info(
                    "线程调试：thread_id=%s, participants=%s, is_private=%s",
                    tid,
                    [(u.pk, u.username) for u in participants],
                    is_private,
                )

                thread_history = load_thread_history(tid)

                messages = cl.direct_messages(tid, amount=20)
                if not messages:
                    logging.info("线程 %s 无消息，跳过", tid)
                    continue

                messages_sorted = sorted(messages, key=lambda m: m.timestamp)
                thread_state = state.get(tid, {})
                last_seen_id = thread_state.get("last_message_id")
                last_focus_id = thread_state.get("last_mention_id")
                logging.info(
                    "线程调试：thread_id=%s, state_last_id=%r, messages_count=%d",
                    tid,
                    last_seen_id,
                    len(messages_sorted),
                )

                def _is_new(msg_id: str, last_seen: str | None) -> bool:
                    if last_seen is None:
                        return True
                    try:
                        return int(msg_id) > int(last_seen)
                    except Exception:
                        return str(msg_id) > str(last_seen)

                new_msgs = [msg for msg in messages_sorted if _is_new(msg.id, last_seen_id)]
                logging.info(
                    "线程调试：thread_id=%s, new_msgs_count=%d, new_ids=%s",
                    tid,
                    len(new_msgs),
                    [m.id for m in new_msgs],
                )
                if not new_msgs:
                    continue

                had_new_messages = True
                changed = update_history_for_thread(
                    thread_history, new_msgs, thread_user_map
                )
                if changed:
                    save_thread_history(tid, thread_history)

                mention_msgs: List[DirectMessage] = []
                reply_candidate: DirectMessage | None = None

                for msg in new_msgs:
                    thread_state["last_message_id"] = msg.id
                    state[tid] = thread_state
                    state_dirty = True

                    text_lower = (msg.text or "").lower()
                    if msg.user_id == my_pk:
                        logging.info("线程 %s 最新消息来自自己，跳过 msg_id=%s", tid, msg.id)
                        continue

                    if "/forgetall" in text_lower:
                        logging.info("收到 /forgetall：thread=%s, msg_id=%s", tid, msg.id)
                        delete_thread_history(tid)
                        thread_history = []
                        thread_state["last_mention_id"] = None
                        last_focus_id = None
                        state_dirty = True
                        cl.direct_answer(tid, "记忆已全部清空")
                        continue

                    if "/forget" in text_lower:
                        logging.info("收到 /forget：thread=%s, msg_id=%s", tid, msg.id)
                        thread_history = thread_history[-10:]
                        save_thread_history(tid, thread_history)
                        cl.direct_answer(tid, "已忘记较早的记录，仅保留最近几条")
                        continue

                    if not is_private and mention_token in text_lower:
                        mention_msgs.append(msg)

                    if DEBUG_IGNORE_MENTION or is_private or mention_token in text_lower:
                        reply_candidate = msg

                    logging.info(
                        "新消息调试：thread=%s, msg_id=%s, from_user_id=%s, username=%s, "
                        "timestamp=%s, text=%r, lower=%r",
                        tid,
                        msg.id,
                        msg.user_id,
                        thread_user_map.get(msg.user_id),
                        getattr(msg, "timestamp", None),
                        msg.text,
                        text_lower,
                    )

                target_msg: DirectMessage | None = None
                focus_id_for_context = None
                if DEBUG_IGNORE_MENTION:
                    target_msg = reply_candidate
                    focus_id_for_context = last_focus_id if not is_private else None
                elif is_private:
                    target_msg = reply_candidate
                elif mention_msgs:
                    target_msg = mention_msgs[-1]
                    focus_id_for_context = last_focus_id

                if target_msg:
                    context = build_context(thread_history, focus_id_for_context)
                    logging.info(
                        "构造上下文完成，长度=%d, target_msg=%s, mention_count=%d",
                        len(context),
                        target_msg.id,
                        len(mention_msgs),
                    )
                    # 随机等待，模拟“思考中”，降低触发风控概率
                    time.sleep(random.uniform(1.0, 3.0))

                    try:
                        reply = ask_qwen(context)
                    except Exception as exc:
                        logging.error("调用 Qwen 出错，使用兜底文案：%s", exc)
                        reply = "我刚刚有点卡顿，可以再说一遍吗？"

                    logging.info(
                        "准备回复：thread=%s, msg_id=%s, reply_preview=%r",
                        tid,
                        target_msg.id,
                        reply[:100],
                    )
                    cl.direct_answer(tid, reply)
                    if not is_private:
                        if mention_msgs:
                            thread_state["last_mention_id"] = mention_msgs[-1].id
                            state_dirty = True

            if state_dirty:
                save_state(state)

            now = time.time()
            if had_new_messages:
                last_activity_ts = now

            idle = (now - last_activity_ts) > 20 * 60
            if idle:
                sleep_sec = random.uniform(30, 90)
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
