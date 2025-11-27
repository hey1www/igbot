# igbot

中英双语说明 / Bilingual guide.

## 项目简介 / Overview
- 一个基于 Qwen 的 Instagram DM/群聊自动回复 Bot，使用 `instagrapi` 轮询消息，根据最近聊天上下文生成简短自然的回复。
- 复用本地 `data/` 目录保存会话状态与聊天片段，默认不随 Git 提交。

## 依赖 / Requirements
- Python 3.11+
- 访问 Qwen 兼容的 `chat/completions` API
- Instagram 账号凭据（用户名 + 密码）

## 环境变量 / Environment
在项目根目录创建 `.env`：
```
IG_USERNAME=your_ig_username
IG_PASSWORD=your_ig_password
QWEN_API_KEY=sk-...
QWEN_API_URL=https://api.your-qwen-endpoint/v1/chat/completions
QWEN_MODEL=qwen2.5-32b-instruct    # 可选
BOT_MENTION_NAME=hey1bot           # 可选，群聊被 @ 的昵称
```

## 本地运行 / Run Locally
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
首次登录可运行 `python login_test.py` 以生成 `data/session.json`（随后会复用）。随后启动：
```
python bot.py
```

## Docker 运行 / Docker
```
docker compose up --build -d
```
`docker-compose.yml` 会挂载 `./data` 以持久化会话与历史。

## 使用说明 / How It Works
- 私聊：直接回复最新消息。
- 群聊：仅在被 `@BOT_MENTION_NAME` 时回复。
- 命令：
  - `/forget`：仅保留最近少量历史。
  - `/forgetall`：清空当前线程记忆。
- 回复上下文来自 `data/history/*.json` 的最近 50 条消息，系统提示可在 `prompts/system_prompt.txt` 自定义。

## 数据与隐私 / Data & Privacy
- `.gitignore` 已忽略 `data/` 内的历史与会话（如 `session.json`, `state.json`, `history/*.json`）以及 `.env`。
- 发布到公共仓库前，确保未包含个人聊天内容或敏感密钥。

## 故障排查 / Troubleshooting
- 登录失败：确认 `.env` 的 IG 凭据正确，必要时删除损坏的 `data/session.json` 后重试。
- Qwen 请求异常：检查 `QWEN_API_URL` 与 `QWEN_API_KEY`，以及网络连通性。
