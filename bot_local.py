# bot_local.py
import os
import time
import json
import random
import logging
from pathlib import Path
from typing import Dict, List

from instagrapi import Client
from instagrapi.mixins.direct import DirectMessage
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SESSION_FILE = DATA_DIR / "session.json"
STATE_FILE = DATA_DIR / "state.json"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)

def load_state() -> Dict[str, str]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logging.warning("读取 STATE_FILE 出错，将忽略：%s", e)
    return {}

def save_state(state: Dict[str, str]):
    try:
        STATE_FILE.write_text(json.dumps(state), encoding="utf-8")
    except Exception as e:
        logging.warning("保存 STATE_FILE 出错：%s", e)

def build_context(messages: List[DirectMessage], max_msgs: int = 10) -> str:
    """
    把最近若干条消息拼成一个多轮对话文本。
    这里只做最简单的 user_id 标记，后面接 Qwen 时可改得更漂亮。
    """
    selected = messages[-max_msgs:]
    lines = []
    for m in selected:
        text = m.text or ""
        text = text.replace("\n", " ")
        lines.append(f"user_{m.user_id}: {text}")
    return "\n".join(lines)

def main_loop():
    load_dotenv()
    IG_USERNAME = os.getenv("IG_USERNAME")
    IG_PASSWORD = os.getenv("IG_PASSWORD")
    BOT_MENTION_NAME = os.getenv("BOT_MENTION_NAME") or IG_USERNAME

    if not IG_USERNAME or not IG_PASSWORD:
        raise RuntimeError("请在 .env 中配置 IG_USERNAME / IG_PASSWORD")

    DATA_DIR.mkdir(exist_ok=True)
    state = load_state()

    cl = Client()
    if SESSION_FILE.exists():
        logging.info("加载现有 session...")
        cl.load_settings(SESSION_FILE)
        cl.login(IG_USERNAME, IG_PASSWORD)
    else:
        logging.info("未找到 session，进行首次登录...")
        cl.login(IG_USERNAME, IG_PASSWORD)
        cl.dump_settings(SESSION_FILE)
        logging.info("已保存 session 至 %s", SESSION_FILE)

    me = cl.user_info_by_username(IG_USERNAME)
    my_pk = me.pk
    logging.info("登录为 %s (pk=%s)", me.username, my_pk)

    mention_token = f"@{BOT_MENTION_NAME.lower()}"
    logging.info("监听被 @ 关键字：%s", mention_token)

    while True:
        try:
            threads = cl.direct_threads(amount=10)
            for thread in threads:
                tid = thread.id

                # 拿最近一些消息
                messages = cl.direct_messages(tid, amount=20)
                if not messages:
                    continue

                # 按时间排序（旧 -> 新）
                messages_sorted = sorted(messages, key=lambda m: m.timestamp)
                last_id_in_thread = state.get(tid)

                # 找出新消息
                new_msgs = []
                for m in messages_sorted:
                    if last_id_in_thread is None or m.id > last_id_in_thread:
                        new_msgs.append(m)
                if not new_msgs:
                    continue

                # 更新状态
                state[tid] = new_msgs[-1].id
                save_state(state)

                last_msg = new_msgs[-1]
                if last_msg.user_id == my_pk:
                    # 自己发的消息，忽略
                    continue

                text = (last_msg.text or "").lower()
                if mention_token not in text:
                    continue

                logging.info("在 thread %s 中检测到被 @，最新消息 id=%s", tid, last_msg.id)

                # 构造上下文（暂时只是调试用）
                ctx = build_context(messages_sorted, max_msgs=10)
                logging.info("上下文预览：\n%s", ctx)

                # 现在先做一个“复读机”：简单回复一句
                reply = f"我听到了，你刚刚说：{last_msg.text}"

                logging.info("准备回复：%s", reply)
                cl.direct_answer(tid, reply)

            # 随机 sleep，降低风控风险
            sleep_sec = 25 + random.randint(0, 10)
            logging.debug("本轮结束，sleep %s 秒", sleep_sec)
            time.sleep(sleep_sec)

        except Exception as e:
            logging.error("主循环异常：%s", e)
            time.sleep(60)

if __name__ == "__main__":
    main_loop()
