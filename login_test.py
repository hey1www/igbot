# login_test.py
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

    if not IG_USERNAME or not IG_PASSWORD:
        raise RuntimeError("请在 .env 中配置 IG_USERNAME 和 IG_PASSWORD")

    DATA_DIR.mkdir(exist_ok=True)

    cl = Client()

    # 如果已经有 session，就优先加载
    if SESSION_FILE.exists():
        print("发现 session.json，尝试复用登录状态...")
        cl.load_settings(SESSION_FILE)
        cl.login(IG_USERNAME, IG_PASSWORD)
    else:
        print("首次登录，使用用户名 + 密码登录...")
        cl.login(IG_USERNAME, IG_PASSWORD)
        cl.dump_settings(SESSION_FILE)
        print(f"已保存 session 到 {SESSION_FILE}")

    user_info = cl.user_info_by_username(IG_USERNAME)
    print("登录成功，当前账号：", user_info.username, " pk =", user_info.pk)

if __name__ == "__main__":
    main()
