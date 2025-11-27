# list_threads.py
import os
from pathlib import Path

from instagrapi import Client
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SESSION_FILE = DATA_DIR / "session.json"

def main():
    load_dotenv()
    IG_USERNAME = os.getenv("IG_USERNAME")
    IG_PASSWORD = os.getenv("IG_PASSWORD")

    cl = Client()
    cl.load_settings(SESSION_FILE)
    cl.login(IG_USERNAME, IG_PASSWORD)

    me = cl.user_info_by_username(IG_USERNAME)
    print("当前账号：", me.username, " pk =", me.pk)

    threads = cl.direct_threads(amount=10)
    print(f"最近 10 个线程：")
    for t in threads:
        user_names = [u.username for u in t.users]
        print("-" * 40)
        print("thread_id:", t.id)
        print("参与者:", user_names)
        print("是否群聊:", t.thread_type)
        print("最新消息摘要:", getattr(t, "last_activity_at", None))

        # 取一些消息
        msgs = cl.direct_messages(t.id, amount=5)
        for m in reversed(msgs):
            print("   ", m.id, m.timestamp, ":", m.user_id, "->", m.text)

if __name__ == "__main__":
    main()
