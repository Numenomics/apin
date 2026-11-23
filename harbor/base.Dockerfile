# Shared base image for the apin Harbor Agent Benchmark tasks.
#
# Do not build this file against the repository root. Build it with
#   uv run python scripts/build_harbor_base.py
# which stages a sealed build context (an allowlist: harness, models, data, codex,
# lane documents, the frozen starting code of every lane, pyproject.toml, uv.lock)
# and builds apin-harbor-base:<release> from it.

FROM python:3.12-slim

ARG APIN_BENCH_RELEASE=unset
LABEL org.numenomics.apin.bench-release=$APIN_BENCH_RELEASE

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON=3.12 \
    UV_PYTHON_DOWNLOADS=never \
    APIN_BENCH_RELEASE=$APIN_BENCH_RELEASE

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .
# Install the project itself, then pin uv to the finished environment: the agent and
# verifier phases run without open network, so `uv run` must use .venv as built and
# never resolve, rebuild or download (UV_NO_SYNC also keeps it from rebuilding apin
# with a build backend that is only in root's uv cache).
RUN uv sync --frozen && rm -f Dockerfile && mkdir -p /logs/verifier
ENV UV_OFFLINE=1 \
    UV_NO_SYNC=1
