# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1 — Builder: install dependencies into an isolated layer
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04 AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /workspace

# Install build-time system deps (Rust for bitsandbytes, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl git ca-certificates pkg-config gcc g++ make rustc cargo \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first for layer caching
COPY pyproject.toml README.md ./
COPY src/litert_studio/ src/litert_studio/

# Create virtual environment and install core API + dev tools
RUN python -m venv /opt/venv \
 && . /opt/venv/bin/activate \
 && pip install --upgrade pip setuptools wheel \
 && pip install ".[api,dev]"

# Install CUDA PyTorch — this is the big download, separate layer for retries
RUN . /opt/venv/bin/activate \
 && pip install torch torchvision torchaudio

# Install training deps in one layer to save time
RUN . /opt/venv/bin/activate \
 && pip install "transformers>=4.57,<6" "peft>=0.12" "safetensors>=0.4" "accelerate>=1.0" "huggingface-hub>=0.26"

# Install LiteRT runtime (Linux-only, ignore failures)
RUN . /opt/venv/bin/activate \
 && pip install --ignore-installed "litert-lm==0.14.0; platform_system == 'Linux'" || true

# SafeTensors conversion deps
RUN . /opt/venv/bin/activate \
 && pip install "transformers>=4.57,<6" "safetensors>=0.4"

# Classic TF conversion (optional, ignore failures)
RUN . /opt/venv/bin/activate \
 && pip install --quiet --disable-pip-version-check "tensorflow>=2.17" || true

# ---------------------------------------------------------------------------
# Stage 2 — Runtime: minimal image with only what's needed
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04 AS runtime

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
