# Single shared image for api / worker / scheduler — they differ only by CMD (see docker-compose.yml).
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        build-essential \
        libpq-dev \
        curl \
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
