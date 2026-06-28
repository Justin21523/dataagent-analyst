ARG PYTHON_IMAGE=python:3.12-slim-bookworm

FROM ${PYTHON_IMAGE}

LABEL org.opencontainers.image.title="DataAgent Analyst Backend"
LABEL org.opencontainers.image.description="FastAPI backend for DataAgent Analyst"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# 使用非 root user 執行 API，降低 container runtime 權限。
RUN groupadd \
      --gid 10001 \
      appgroup \
    && useradd \
      --uid 10001 \
      --gid appgroup \
      --create-home \
      --shell /usr/sbin/nologin \
      appuser

COPY requirements.txt ./requirements.txt

RUN python -m pip install \
      --no-cache-dir \
      -r requirements.txt

COPY --chown=appuser:appgroup backend ./backend
COPY --chown=appuser:appgroup data/samples ./data/samples

# Named volumes 首次掛載時會繼承這些目錄與檔案權限。
RUN mkdir -p \
      /app/data/raw \
      /app/data/processed \
      /app/data/reports \
      /app/data/models \
      /app/data/vector_store \
    && touch \
      /app/data/raw/.container-keep \
      /app/data/processed/.container-keep \
      /app/data/reports/.container-keep \
      /app/data/models/.container-keep \
      /app/data/vector_store/.container-keep \
    && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

HEALTHCHECK \
  --interval=15s \
  --timeout=5s \
  --start-period=20s \
  --retries=5 \
  CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"

CMD ["sh", "-c", "exec python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS:-1}"]
