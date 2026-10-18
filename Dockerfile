# Native Linux x86_64 + NVIDIA GPU. Compilation tools are required for Triton.
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04@sha256:e542739fcaa4f45da5add8c4cf5769783a61628b2518304a5bbe4ace468b8c8f

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    TERN_CACHE_DIR=/cache/tern \
    HF_HOME=/cache/tern/hf-home \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev ca-certificates git ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv \
    && python3.12 -m venv /opt/uv \
    && /opt/uv/bin/pip install --no-cache-dir uv==0.12.1 \
    && ln -s /opt/uv/bin/uv /usr/local/bin/uv

WORKDIR /app
COPY requirements.lock ./
RUN uv pip sync --python /opt/venv/bin/python requirements.lock \
      --extra-index-url https://download.pytorch.org/whl/cu129 \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      --index-strategy unsafe-best-match --no-cache
COPY . .
RUN uv pip install --python /opt/venv/bin/python --no-deps . \
    && python -m tern check-runtime

EXPOSE 8000
ENTRYPOINT ["python", "-m", "tern"]
CMD ["serve", "--profile", "rtx-pro-6000", "--host", "0.0.0.0"]
