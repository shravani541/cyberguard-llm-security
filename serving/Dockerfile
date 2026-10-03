# ─────────────────────────────────────────────────────────────
# Dockerfile — CyberGuard vLLM Inference Server
# Base: NVIDIA CUDA 12.1 + Python 3.11
# GPU: T4 / A100 compatible
# ─────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Build-time args
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.11

# ── System Dependencies ───────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    python3-setuptools \
    git \
    curl \
    wget \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Use python3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# ── Working Directory ─────────────────────────────────────────
WORKDIR /app

# ── Python Dependencies ───────────────────────────────────────
COPY serving/requirements-serve.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-serve.txt

# ── Copy API Code ─────────────────────────────────────────────
COPY serving/api/ ./api/

# ── Model Mount Point ─────────────────────────────────────────
# Mount your fine-tuned model here at runtime:
# docker run -v /path/to/merged_model:/app/model ...
RUN mkdir -p /app/model

# ── Environment Defaults ──────────────────────────────────────
ENV MODEL_PATH=/app/model
ENV MAX_MODEL_LEN=4096
ENV GPU_MEMORY_UTIL=0.85
ENV TENSOR_PARALLEL_SIZE=1
ENV MAX_BATCH_SIZE=8
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── Health Check ──────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Expose Port ───────────────────────────────────────────────
EXPOSE 8000

# ── Startup ───────────────────────────────────────────────────
CMD ["python", "-m", "uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--log-level", "info", \
     "--workers", "1"]
