# Shared base image for apin Harbor Agent Benchmark tasks.
# Build from the repository root:
#   docker build -f harbor/base.Dockerfile -t apin-harbor-base:latest .

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON=3.12

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .
# Strip hidden verifier, hidden rewind seeds/reference fixes, and Harbor task tests
# so the agent container cannot read reference answers or hidden verification seeds.
RUN rm -rf /app/harbor /app/autoresearch/lanes/S3 && mkdir -p /logs/verifier
