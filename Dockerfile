FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 预先复制依赖声明，利用 Docker 缓存
COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# 复制代码主体
COPY bot.py ./bot.py
COPY login_test.py ./login_test.py
COPY list_threads.py ./list_threads.py

# 确保数据目录存在（实际运行时会被 volume 覆盖）
RUN mkdir -p /app/data

CMD ["python", "bot.py"]
