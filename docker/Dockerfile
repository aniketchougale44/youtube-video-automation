# Single shared image for api / worker / scheduler — they differ only by CMD (see docker-compose.yml).
FROM python:3.11-slim

WORKDIR /app

# Split into two layers: the small one (curl/libpq-dev/build-essential) caches independently of
# ffmpeg's much larger dependency tree, so an interrupted build (flaky network, Docker Desktop
# restart mid-pull) doesn't force a full re-download of everything on retry.
RUN apt-get update -o Acquire::Retries=5 && apt-get install -y -o Acquire::Retries=5 --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update -o Acquire::Retries=5 && apt-get install -y -o Acquire::Retries=5 --no-install-recommends \
        ffmpeg \
        fonts-noto-core \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml alembic.ini ./
COPY agents ./agents
COPY graph ./graph
COPY api ./api
COPY scheduler ./scheduler
COPY db ./db
COPY tools ./tools
COPY worker ./worker
COPY core ./core

RUN pip install --no-cache-dir -e .

RUN mkdir -p /app/media

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
