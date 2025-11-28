"""
离线修复历史/摘要中机器人用户名错误的小工具。
使用前请先停止运行 bot.py / bot_debug.py，避免并发写入。
运行示例：python fix_self_name.py
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
HISTORY_DIR = DATA_DIR / "history"
SUMMARY_DIR = DATA_DIR / "summary"

# 根据实际账号信息修改
SELF_ID = 78633320983  # 机器人自己的 pk
SELF_NAME = "hey1bot"  # 机器人自己的用户名


def fix_history() -> None:
    if not HISTORY_DIR.exists():
        print("history dir not found:", HISTORY_DIR)
        return
    for path in HISTORY_DIR.glob("*.json"):
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except Exception as exc:
            print("skip history (read error):", path, exc)
            continue

        if not isinstance(data, list):
            print("skip history (not list):", path)
            continue

        changed = False
        for entry in data:
            uid = entry.get("user_id")
            uname = entry.get("username")
            if uid == SELF_ID or uname == f"user_{SELF_ID}":
                entry["user_id"] = SELF_ID
                entry["username"] = SELF_NAME
                changed = True

        if changed:
            path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print("fixed history:", path.name)


def fix_summary() -> None:
    if not SUMMARY_DIR.exists():
        print("summary dir not found:", SUMMARY_DIR)
        return
    for path in SUMMARY_DIR.glob("*.json"):
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except Exception as exc:
            print("skip summary (read error):", path, exc)
            continue

        if not isinstance(data, list):
            print("skip summary (not list):", path)
            continue

        changed = False
        for block in data:
            participants = block.get("participants") or []
            # participants 通常是包含 user_id / username 的 dict 列表
            for p in participants:
                uid = p.get("user_id")
                uname = p.get("username")
                if uid == SELF_ID or uname == f"user_{SELF_ID}":
                    p["user_id"] = SELF_ID
                    p["username"] = SELF_NAME
                    changed = True

        if changed:
            path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print("fixed summary:", path.name)


if __name__ == "__main__":
    fix_history()
    fix_summary()
