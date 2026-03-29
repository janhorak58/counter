FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# dependency layer
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# application code
COPY src ./src
COPY configs ./configs
COPY tracker.bytetrack.yaml ./

# final install of the project itself
RUN uv sync --frozen --no-dev --no-editable

EXPOSE 8501

CMD ["uv", "run", "counter-ui", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]