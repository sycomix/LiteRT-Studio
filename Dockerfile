# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1 — Builder: install dependencies into an isolated layer
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=0 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /workspace

# Install build-time system deps (Rust for bitsandbytes, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl git ca-certificates pkg-config \
        && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first for layer caching
COPY pyproject.toml README.md ./
COPY src/litert_studio/ src/litert_studio/

# Install Python dependencies into a virtual environment
RUN python -m venv /opt/venv && . /opt/venv/bin/activate \
 && pip install --upgrade pip setuptools wheel \
 && pip install --no-compile ".[api,training,conversion,runtime]" \
 && pip install --no-compile ".[dev]"

# ---------------------------------------------------------------------------
# Stage 2 — Runtime: minimal image with only what's needed
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /bin/bash appuser \
 && mkdir -p /app/runs /app/workspace && chown -R appuser:appuser /app

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv
# Copy source (non-editable at runtime)
COPY --chown=appuser:appuser src/litert_studio/ litert_studio/

USER appuser

EXPOSE 7860/tcp

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

ENTRYPOINT ["litert-studio"]
CMD ["serve", "--workspace", "/app/workspace", "--port", "7860"]
